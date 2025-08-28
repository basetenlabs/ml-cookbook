import argparse
import os
from datasets import load_dataset, DatasetDict, Audio
from datasets import disable_progress_bars, enable_progress_bars
# from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

enable_progress_bars()

def load_and_process_common_accent_dataset(args):
    print("Loading and processing CommonAccent dataset...")
    common_accent = DatasetDict()
    common_accent["train"] = load_dataset(args.dataset_name, split="train", trust_remote_code=True)
    common_accent["test"] = load_dataset(args.dataset_name, split="test", trust_remote_code=True)
    common_accent = common_accent.cast_column("audio", Audio(sampling_rate=16000))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id, device_map="auto")
    tokenizer = WhisperTokenizer.from_pretrained(args.model_id, language=args.language, task=args.task) 

    print("Processing dataset...")
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    common_accent = common_accent.map(prepare_dataset, remove_columns=common_accent.column_names["train"], num_proc=1)
    return {
        "dataset": common_accent,
        "tokenizer": tokenizer,
        "feature_extractor": feature_extractor
    }


def load_and_process_dataset(args):
    # Used for dataset: mozilla-foundation/common_voice_11_0
    print("Loading and processing dataset...")
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(args.dataset_name, args.language_id, split="train+validation", trust_remote_code=True)
    common_voice["test"] = load_dataset(args.dataset_name, args.language_id, split="test", trust_remote_code=True)
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    print("Dataset loaded successfully.")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id, device_map="auto")
    tokenizer = WhisperTokenizer.from_pretrained(args.model_id, language=args.language, task=args.task) 

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    print("Processing dataset...")

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)
    return common_voice


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def prepare_compute_metrics(tokenizer, metric):
    def compute_metrics(pred):
        with torch.no_grad():
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            # replace -100 with the pad_token_id
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            # we do not want to group tokens when computing the metrics
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    return compute_metrics


def parse_args():
    output_dir = os.environ.get("BT_CHECKPOINT_DIR", "whisper-op")
    parser = argparse.ArgumentParser(description="Train a Whisper model for transcription.")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Directory to save the model.")
    parser.add_argument("--model_id", type=str, default="openai/whisper-small", help="Name of the pre-trained Whisper model.")
    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_11_0", help="Name of the dataset to use for training.")
    parser.add_argument("--language", type=str, default="Hindi", help="Language for the Whisper model.")
    parser.add_argument("--language_id", type=str, default="hi", help="Language for the Whisper model.")
    parser.add_argument("--task", type=str, default="transcribe", help="Task for the Whisper model.")
    parser.add_argument("--num_steps", type=int, default=5000, help="Number of training steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=4e-6, help="Learning rate for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument("--mixed_precision_fp16", action="store_true", help="Use mixed precision training.")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy to use during training.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between model saves.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of steps between evaluations.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Number of steps between logging.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum number of training steps.")
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Whether to pin memory in the dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation.")
    parser.add_argument("--generation_max_length", type=int, default=225, help="Maximum length for generation.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hugging Face Hub model ID to push the trained model.")
    parser.add_argument("--hub_strategy", type=str, default="end", help="Strategy for saving to the Hugging Face Hub.")
    parser.add_argument("--report_to", type=str, default=None, help="Reporting tool to use (e.g., 'wandb', 'tensorboard').")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of evaluations with no improvement to stop training.")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Minimum change to be considered an improvement for early stopping.")
    return parser.parse_args()

def main(args):

    print(f"Loading {args.model_id} model...")
    # load model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None

    print("Model loaded successfully.")

    metric = evaluate.load("wer")

    processor = WhisperProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)

    print("Loading dataset...")
    # common_voice = load_and_process_dataset(args)
    common_voice_dict = load_and_process_common_accent_dataset(args)
    common_voice = common_voice_dict["dataset"]
    tokenizer = common_voice_dict["tokenizer"]

    print("Dataset loaded successfully.")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    prepared_compute_metrics = prepare_compute_metrics(tokenizer, metric)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=False,
        fp16=args.mixed_precision_fp16,
        eval_strategy=args.eval_strategy,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=[args.report_to],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,
        dataloader_pin_memory=args.dataloader_pin_memory if hasattr(args, 'dataloader_pin_memory') else False,
    )
    print("Training arguments:", training_args)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,  # Stop if no improvement for n consecutive evaluations
        early_stopping_threshold=args.early_stopping_threshold, # Minimum change to be considered an improvement
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=prepared_compute_metrics,
        tokenizer=processor,
        callbacks=[early_stopping_callback],
    )
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    print("All parsed arguments:")
    print(args)
    main(args)