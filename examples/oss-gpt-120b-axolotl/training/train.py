from pathlib import Path
import os

from axolotl.utils.dict import DictDefault
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets
from axolotl.train import train

NODE_RANK = int(os.environ.get("BT_NODE_RANK", 0))

OUTPUT_DIR = Path(os.environ.get("BT_CHECKPOINT_DIR", "./outputs")) / f"gpt-oss-120b-lora-node{NODE_RANK}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    config = DictDefault(
        # Model
        base_model="axolotl-ai-co/gpt-oss-120b-dequantized",
        use_kernels=False,
        experimental_skip_move_to_device=True,

        # LoRA
        adapter="lora",
        lora_r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_target_modules="all-linear",

        # Dataset
        dataset_prepared_path="last_run_prepared",
        datasets=[
            {
                "path": "winglian/pirate-ultrachat-10k",
                "split": "train",
                "type": "chat_template",
            }
        ],
        val_set_size=0.05,

        # Packing / padding / seq len
        sequence_len=131072,
        sample_packing=True,
        pad_to_sequence_len=True,
        eval_sample_packing=True,

        # Training loop
        num_epochs=10,
        micro_batch_size=1,
        gradient_accumulation_steps=1,

        # Optim & schedule
        optimizer="adamw_torch_fused",
        lr_scheduler="constant_with_warmup",
        learning_rate=1e-4,
        warmup_ratio=0.1,


        # Precision / perf
        bf16=True,
        tf32=True,
        flash_attention=True,


        # Logging / saving
        logging_steps=1,
        save_strategy="no",
        save_only_model=True,
        output_dir=str(OUTPUT_DIR),

        # Special tokens
        eot_tokens=["<|end|>", "<|return|>"],


        # misc
        gc_steps=-1,

        # FSDP
        fsdp_version=2,
        fsdp_config={
            "offload_params": False,
            "state_dict_type": "FULL_STATE_DICT",
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": "GptOssDecoderLayer",
            "reshard_after_forward": True,
            "cpu_ram_efficient_loading": False,
            "activation_checkpointing": True,
        },

        # Plugins
        plugins=[
            "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
        ],
    )

    cfg = load_cfg(config)
    dataset_meta = load_datasets(cfg=cfg)
    model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)


if __name__ == "__main__":
    main()
