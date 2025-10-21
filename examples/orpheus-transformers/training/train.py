import argparse

from unsloth import FastLanguageModel

from datasets import load_dataset

import locale
import os

from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

from audio_utils.utils import (
    CODES_LIST_NAME,
    add_codes,
    remove_duplicate_frames,
    create_input_ids,
)


def parse_args():
    output_dir = os.environ.get("BT_CHECKPOINT_DIR", "orpheus_voice_cloning_output")
    parser = argparse.ArgumentParser(
        description="Train a Whisper model for transcription."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="Directory to save the model.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="unsloth/orpheus-3b-0.1-ft",
        help="Name of the pre-trained Whisper model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="kchatr/fish-and-chips",
        help="Name of the dataset to use for training.",
    )
    parser.add_argument("--lora_rank", type=int, default=64, help="Rank for LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=64, help="Alpha for LoRA.")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.0, help="Dropout rate for LoRA."
    )
    parser.add_argument(
        "--lora_bias", type=str, default="none", help="Bias type for LoRA."
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        type=str,
        default="unsloth",
        help="True or 'unsloth' for very long context",
    )
    parser.add_argument(
        "--num_steps", type=int, default=5000, help="Number of training steps."
    )
    parser.add_argument(
        "--encoding_format",
        type=str,
        default="UTF-8",
        help="Encoding format for audio files.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=3407,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Max Sequence length for FastLanguageModel.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate for training."
    )
    parser.add_argument(
        "--optim", type=str, default="adamw_8bit", help="Optimizer to use for training."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training."
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        help="Evaluation strategy to use during training.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of steps between model saves.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Number of steps between evaluations.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=2, help="Number of steps between logging."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device for training.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=60, help="Maximum number of training steps."
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        help="Whether to pin memory in the dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per device for evaluation.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="baseten-admin/orpheus_voice_cloning1",
        help="Hugging Face Hub model ID to push the trained model.",
    )
    parser.add_argument(
        "--hub_strategy",
        type=str,
        default="end",
        help="Strategy for saving to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="Reporting tool to use (e.g., 'wandb', 'tensorboard').",
    )
    return parser.parse_args()


def main(args):
    # -- Load model + tokenizer --
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,  # Choose any for long context!
        dtype=None,  # Select None for auto detection
        load_in_4bit=False,  # Select True for 4bit which reduces memory usage,
    )
    model = model.to("cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Get PEFT model --
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,  # True or "unsloth" for very long context
        random_state=args.random_state,
        use_rslora=False,
        loftq_config=None,
    )

    dataset = load_dataset(args.dataset_name, split="train[:50%]")

    locale.getpreferredencoding = lambda: args.encoding_format
    ds_sample_rate = dataset[0]["audio"]["sampling_rate"]  # original freq

    # Create tokenized audio dataset
    print("Tokenizing audio dataset...")
    dataset = dataset.map(lambda x: add_codes(x, orig_freq=ds_sample_rate), remove_columns=["audio"])

    # Clean dataset
    print("Cleaning dataset...")
    dataset = dataset.filter(lambda x: x[CODES_LIST_NAME] is not None)
    dataset = dataset.filter(lambda x: len(x[CODES_LIST_NAME]) > 0)
    dataset = dataset.map(remove_duplicate_frames)

    # Convert dataset to format needed by the Trainer
    print("Converting dataset to inputs_ids, labels, and attention_mask...")
    # dataset = dataset.map(create_input_ids, remove_columns=["text", "codes_list"])
    dataset = dataset.map(
        lambda x: create_input_ids(x, tokenizer),
        remove_columns=["text", CODES_LIST_NAME],
        # num_proc=8, # Use multiple processes for faster mapping
    )
    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [
        col for col in dataset.column_names if col not in columns_to_keep
    ]
    dataset = dataset.remove_columns(columns_to_remove)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_seq_length,
    )

    # Initialize Trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            # lr_scheduler_type = "linear",
            seed=args.random_state,
            output_dir=args.output_dir,
            report_to=args.report_to,  # Use this for WandB etc
        ),
    )

    trainer_stats = trainer.train()
    print("Training complete")
    print("Training stats: ", trainer_stats)


if __name__ == "__main__":
    args = parse_args()
    print("All parsed arguments:")
    print(args)
    main(args)
