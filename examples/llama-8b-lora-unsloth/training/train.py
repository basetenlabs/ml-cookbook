import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

max_seq_length = 4096
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-*
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    # mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)


def formatting_prompts_func(examples):
    texts = []
    # Handle batched examples correctly
    for question, solution in zip(examples["question"], examples["generated_solution"]):
        convo = [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": solution,
            },
        ]
        text = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


dataset = load_dataset(
    "nvidia/OpenMathInstruct-1", split="train[:20%]"
)  # Use a smaller split for quick training
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

checkpoint_dir = os.environ.get("BT_CHECKPOINT_DIR", "checkpoints")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=checkpoint_dir,
        report_to="none",  # Use this for wandb etc
        # push to hf
        push_to_hub=bool(
            os.environ.get("HF_WRITE_LOC", "")
        ),  # Set this to your HuggingFace repo name
        hub_model_id=os.environ.get("HF_WRITE_LOC", ""),
    ),
)

trainer_stats = trainer.train()
print("Done training!")
