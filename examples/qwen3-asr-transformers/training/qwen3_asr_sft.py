# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from QwenLM/Qwen3-ASR's official finetuning/qwen3_asr_sft.py.
import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments


def resolve_model_path(path_or_repo_id: str) -> str:
    """Return a local model directory for checkpoint metadata copying."""
    if os.path.isdir(path_or_repo_id):
        return path_or_repo_id
    return snapshot_download(repo_id=path_or_repo_id)


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        match = _CKPT_RE.match(name)
        if not match:
            continue
        step = int(match.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16_000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example.get("prompt", "")
        prefix_messages = build_prefix_messages(prompt, None)
        prefix_text = processor.apply_chat_template(
            [prefix_messages], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": example["audio"],
            "target": example["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16_000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [feature["audio"] for feature in features]
        prefix_texts = [feature["prefix_text"] for feature in features]
        targets = [feature["target"] for feature in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [
            prefix + target + eos for prefix, target in zip(prefix_texts, targets)
        ]
        audios = [load_audio(path, sr=self.sampling_rate) for path in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for index, prefix_len in enumerate(prefix_lens):
            labels[index, :prefix_len] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for filename in required:
        src = os.path.join(src_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, filename))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if not os.path.isdir(checkpoint_dir):
            checkpoint_dir = kwargs.get("checkpoint", checkpoint_dir)
        copy_required_hf_files_for_qwen_asr(self.base_model_path, checkpoint_dir)
        return control


def parse_args():
    parser = argparse.ArgumentParser("Qwen3-ASR fine-tuning")

    parser.add_argument("--model_path", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--eval_file", default="")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--sr", type=int, default=16_000)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_acc", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--gradient_checkpointing", type=int, choices=(0, 1), default=0)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, choices=(0, 1), default=1)
    parser.add_argument("--persistent_workers", type=int, choices=(0, 1), default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    parser.add_argument("--save_strategy", choices=("steps", "epoch"), default="epoch")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--resume_from", default="")
    parser.add_argument("--resume", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def main():
    args_cli = parse_args()
    if not args_cli.train_file:
        raise ValueError(
            "TRAIN_FILE is required and must contain audio/text JSONL fields."
        )

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model_path = resolve_model_path(args_cli.model_path)
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        attn_implementation="flash_attention_2",
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    if args_cli.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    raw_dataset = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    dataset = raw_dataset.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in dataset:
        drop = [column for column in dataset[split].column_names if column not in keep]
        if drop:
            dataset[split] = dataset[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
    )
    has_eval = bool(args_cli.eval_file)
    strategy = args_cli.save_strategy
    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        per_device_eval_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=(
            args_cli.prefetch_factor if args_cli.num_workers > 0 else None
        ),
        save_strategy=strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy=strategy if has_eval else "no",
        eval_steps=args_cli.save_steps if strategy == "steps" else None,
        do_eval=has_eval,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=collator,
        processing_class=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(model_path)],
    )

    resume_from = args_cli.resume_from.strip()
    if not resume_from and args_cli.resume:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""
    if resume_from and trainer.args.process_index == 0:
        print(f"[resume] resume_from_checkpoint = {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from or None)

    # Always leave one stable, self-contained path even when save_steps exceeds
    # the total number of updates in a smoke test.
    final_dir = os.path.join(args_cli.output_dir, "final")
    trainer.save_model(final_dir)
    if trainer.args.process_index == 0:
        copy_required_hf_files_for_qwen_asr(model_path, final_dir)
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(final_dir)
        print(f"Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
