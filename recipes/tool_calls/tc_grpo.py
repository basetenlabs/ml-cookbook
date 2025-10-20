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
ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn")
pattern = r'<tools>(.*?)</tools>'

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_example_grpo(x, tokenizer=tokenizer):
    """Format examples for GRPO - we need prompts and completions separately"""
    try:
        tools = [s['function'] for s in ast.literal_eval(re.findall(pattern, x['conversations'][0]['value'], re.DOTALL)[-1].strip())]
        # Create the prompt (without assistant response)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are an assistant capable of tool calls"}, 
                {"role": "user", "content": x['conversations'][-2]['value']}, 
            ], 
            tools=tools, 
            add_generation_prompt=True, 
            tokenize=False
        )
        # Get the ground truth completion (reference)
        completion = x['conversations'][-1]['value']
        return {
            "prompt": prompt,
            # "completion": completion,
            "references": completion,
            "tools": json.dumps(tools)
        }
    except Exception as e:
        return None

formatted_dataset = ds.map(format_example_grpo).filter(lambda x: x is not None)
formatted_dataset = formatted_dataset.remove_columns(["conversations", "category", "subcategory", "task"])

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
    load_in_8bit=True
)

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Define reward function for tool calling
# def reward_function(prompts, completions, references, tools_list):
def reward_function(prompts, **kwargs):
    """
    Reward function that evaluates the quality of tool calls.
    Returns a list of reward scores (one per completion).
    """
    # print("All kwargs:")
    # for key in kwargs:
    #     print(f"  {key}: {type(kwargs[key])}")
    rewards = []

    completions = kwargs.get("completions", [])
    references = kwargs.get("references", [])
    tools = kwargs.get("tools", [])

    for completion, reference in zip(completions, references):
        reward = 0.0
        
        # 1. Check if completion contains valid JSON tool call
        try:
            # Look for tool call patterns
            if '<tool_call>' in completion and '</tool_call>' in completion:
                reward += 2.0
                
                # Extract and validate JSON
                tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', completion, re.DOTALL)
                if tool_call_match:
                    tool_call_json = json.loads(tool_call_match.group(1))
                    reward += 2.0
                    
                    # Check if tool name matches reference
                    if 'name' in tool_call_json:
                        ref_tool_match = re.search(r'<tool_call>(.*?)</tool_call>', reference, re.DOTALL)
                        if ref_tool_match:
                            ref_tool = json.loads(ref_tool_match.group(1))
                            if tool_call_json.get('name') == ref_tool.get('name'):
                                reward += 3.0
                            
                            # Check if arguments overlap
                            if 'arguments' in tool_call_json and 'arguments' in ref_tool:
                                arg_overlap = len(set(tool_call_json['arguments'].keys()) & 
                                                set(ref_tool['arguments'].keys()))
                                reward += arg_overlap * 1.0
        except:
            reward -= 1.0
        
        # 2. Penalize if no tool call when reference has one
        if '<tool_call>' in reference and '<tool_call>' not in completion:
            reward -= 5.0
        
        # 3. Penalize if tool call when reference doesn't have one
        if '<tool_call>' not in reference and '<tool_call>' in completion:
            reward -= 2.0
        
        # 4. Length penalty (avoid overly verbose responses)
        if len(completion) > len(reference) * 2:
            reward -= 1.0
        
        rewards.append(reward)
    
    return rewards

# GRPO Configuration
grpo_config = GRPOConfig(
    output_dir="./qwen-tool-calling-grpo",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
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
trainer.save_model("./qwen-tool-calling-grpo-final")
tokenizer.save_pretrained("./qwen-tool-calling-grpo-final")

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
