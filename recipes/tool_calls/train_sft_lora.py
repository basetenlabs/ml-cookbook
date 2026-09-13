## pip install -U bitsandbytes

import ast
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# dataset = load_dataset("younissk/tool-calling-mix")
ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn")
pattern = r'<tools>(.*?)</tools>'

# 4. Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"  # or Qwen2.5-3B, Qwen2.5-1.5B
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_example(x):
    try:
        tools = [s['function'] for s in ast.literal_eval(re.findall(pattern, x['conversations'][0]['value'], re.DOTALL)[-1].strip())]
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are an assistant capable of tool calls"}, 
                {"role": "user", "content": x['conversations'][-2]['value']}, 
                {"role": "assistant", "content": x['conversations'][-1]['value']}
            ], 
            tools=tools, 
            add_generation_prompt=True, 
            tokenize=False
        )
        return {"text": text}
    except Exception as e:
        # Return None or a sentinel value to filter out later
        return None

formatted_dataset = ds.map(format_example).filter(lambda x: x is not None)
formatted_dataset = formatted_dataset.remove_columns(["conversations", "category", "subcategory", "task"])

train_test_split = formatted_dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

# Convert to dataset
# formatted_data = [format_training_example(ex) for ex in training_examples]
# dataset = Dataset.from_list(formatted_data)
# dataset = formatted_dataset
# print(" Dataset --- ", dataset)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True  # Use 8-bit quantization for memory efficiency
)

# 5. Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

"""
text = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    tokenize=False
)
"""

lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

sft_config = SFTConfig(
    output_dir="./qwen-tool-calling3",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",  # Enable evaluation
    eval_steps=30,  # Evaluate every 100 steps
    optim="paged_adamw_8bit",
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    # SFT-specific parameters
    dataset_text_field='text',
    max_length=2048,
)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=formatted_dataset['train'],
    eval_dataset=eval_dataset,  # Add validation dataset
)

trainer.train()


# 8. Train the model
print("Starting training...")
trainer.train()

# 9. Save the fine-tuned model
trainer.save_model("./qwen-tool-calling-final")
tokenizer.save_pretrained("./qwen-tool-calling-final")

print("Training complete!")

# 10. Example inference
def test_tool_calling(prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
test_prompt = f"""<|im_start|>system
You are a helpful assistant with access to tools: {json.dumps(tools)}<|im_end|>
<|im_start|>user
What's the weather in London?<|im_end|>
<|im_start|>assistant
"""

test_tool_calling(test_prompt)