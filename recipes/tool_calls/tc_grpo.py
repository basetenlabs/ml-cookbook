## pip install -U bitsandbytes trl transformers datasets peft accelerate

import ast
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Load dataset
# ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn")
# ds = load_dataset("baseten-admin/CleanNousResearch_simple")
ds = load_dataset("Salesforce/xlam-function-calling-60k")
pattern = r'<tools>(.*?)</tools>'

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_example(x, add_generation_prompt=False):
    try:
        pattern = r'<tools>(.*?)</tools>'
        tools = json.loads(x['tools'])
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": x['query']}, 
            ], 
            tools=tools, 
            add_generation_prompt=add_generation_prompt, 
            tokenize=False
        )
        return {
            "prompt": text,
            # "text": text,
            # "tools": tools,
            "gt_generation": x['answers'],
            }
    except Exception as e:
        print(e)
        # Return None or a sentinel value to filter out later
        return None

formatted_dataset = ds.map(format_example).filter(lambda x: x is not None)
# formatted_dataset = formatted_dataset.remove_columns(["conversations", "category", "subcategory", "task"])

train_test_split = formatted_dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    # load_in_8bit=True
)

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "all_linear"],
    lora_dropout=0.05,
    bias="none",
    # task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Define reward function for tool calling

def get_all_available_tools_dict(prompt):
    pattern = r'<tools>(.*?)</tools>'
    try:
        data_string = re.findall(pattern, prompt, re.DOTALL)[-1].strip()
        json_objects = []
        for line in data_string.strip().split('\n'):
            if line:  # Skip empty lines
                json_objects.append(json.loads(line))
        tool_dict = {}
        for obj in json_objects:
            tool_dict[obj['name']] = obj
        return tool_dict
    except exception as e:
        print(f"Exception {e}")
        return {}

def get_selected_tools_list(generated_text):
    """Returns a dict of tool_name: list of tool_calls from the generated text in order"""
    pattern =r'<tool_call>(.*?)</tool_call>'
    tools = re.findall(pattern, generated_text, re.DOTALL)
    parsed_tools = []
    for tool in tools:
        try:
            # tool = ast.literal_eval(tool.encode().decode('unicode_escape').strip().replace('"', "'")) # escape chars and some other nonsense
            tool = json.loads(tool.encode().decode('unicode_escape').strip())
            parsed_tools.append(tool)
        except Exception as e:
            print(f"Exception in parsing generated tools for {tool}")
    return parsed_tools

# def reward_function(prompts, completions, references, tools_list):
def reward_function(prompts, completions, **kwargs):
    """
    Reward function that evaluates the quality of tool calls.
    Returns a list of reward scores (one per completion).
    """
    # print("All kwargs:")
    # for key in kwargs:
    #     print(f"  {key}: {type(kwargs[key])}")
    rewards = []
    # completions = kwargs.get("completions", [])
    references = kwargs.get("gt_generation", [])
    # references = kwargs.get("completion", [])
    tools = kwargs.get("tools", [])
    for prompts, completion, reference in zip(prompts, completions, references):
        available_tools = get_all_available_tools_dict(prompts) # dict of tool_name: tool_def
        reward = 0.0
        try:
            gen_tools = get_selected_tools_list(completion)
        except:
            # not parsable or correct structure 
            print("Count not parse")
            reward = -3.0
            rewards.append(reward)
            continue
        gt_tools = json.loads(reference)
        if gt_tools == gen_tools: # perfect match
            rewards.append(4.0)
            continue
        # lengths are unequal or something else is wrong
        PENALTY_NAME = 1.0
        PENALTY_MISSING_PARAM = 0.5
        PENALTY_WRONG_PARAM = 0.5
        PENALTY_EXTRA_PARAM = 0.25
        REWARD_CORRECT_CALL = 1.0
        reward = 0.0
        # Account for extra/missing tool calls
        if len(gen_tools) < len(gt_tools):
            reward -= PENALTY_NAME * (len(gt_tools) - len(gen_tools))
        elif len(gen_tools) > len(gt_tools):
            reward -= PENALTY_NAME * (len(gen_tools) - len(gt_tools))
        # Compare in order
        for gt_tool, gen_tool in zip(gt_tools, gen_tools):
            # tool name
            if gt_tool['name'] != gen_tool['name']:
                reward -= PENALTY_NAME
                continue
            # params
            gt_params = gt_tool.get('parameters', {})
            gen_params = gen_tool.get('parameters', {})
            # missing or wrong params
            for key, value in gt_params.items():
                if key not in gen_params:
                    reward -= PENALTY_MISSING_PARAM
                elif gen_params[key] != value:
                    reward -= PENALTY_WRONG_PARAM
            # extra params
            for key in gen_params:
                if key not in gt_params:
                    reward -= PENALTY_EXTRA_PARAM
            # reward for correct call (name + maybe params)
            # But you might choose: only if no param penalties
            if gt_params and not any(key not in gen_params or gen_params[key] != gt_params[key] for key in gt_params):
                # all params matched
                reward += REWARD_CORRECT_CALL
            elif not gt_params:
                # no params expected, name matches
                reward += REWARD_CORRECT_CALL
            else:
                # partial match: maybe +0 or some smaller reward
                reward += REWARD_CORRECT_CALL * 0.5
        rewards.append(reward)
    return rewards

# GRPO Configuration
grpo_config = GRPOConfig(
    output_dir="./qwen-tool-calling-1",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    # learning_rate=5e-4,
    learning_rate=1e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=10,
    bf16=True,
    max_grad_norm=0.3,
    
    # GRPO-specific parameters
    num_generations=4,  # Number of generations per prompt for group comparison
    temperature=0.9,  # Sampling temperature
    # max_new_tokens=512,
    # kl=0.05,  # KL divergence coefficient to prevent drift from reference
)

# Initialize GRPO Trainer
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # tokenizer=tokenizer,
    reward_funcs=[reward_function],
)

# Train the model
print("Starting GRPO training...")
trainer.train()

# Save the fine-tuned model
trainer.save_model("./qwen-tool-calling-grpo-1")
tokenizer.save_pretrained("./qwen-tool-calling-grpo-1")

print("Training complete!")

# Example inference
def test_tool_calling(prompt, tools):
    model.eval()
    
    formatted_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are an assistant capable of tool calls"}, 
            {"role": "user", "content": prompt}
        ],
        tools=tools,
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(response)

# Test the model
test_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

test_tool_calling("What's the weather in London?", test_tools)