from pathlib import Path
import os

from axolotl.utils.dict import DictDefault
from axolotl.cli.config import load_cfg
from axolotl.cli.quantize import do_quantize
from axolotl.common.datasets import load_datasets
from axolotl.train import train

OUTPUT_DIR = Path(os.environ.get("BT_CHECKPOINT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    config = DictDefault(
        # Model
        base_model="openai/gpt-oss-20b",
        bf16=True,

        # Plugins
        plugins=[
            "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
        ],

        # Quantization
        model_quantization_config="Mxfp4Config",
        model_quantization_config_kwargs={
            "dequantize": True,
        },

        # Dataset / prepared cache / special tokens
        dataset_prepared_path="./outputs/last_run_prepared",
        datasets=[
            {
                "path": "winglian/pirate-ultrachat-10k",
                "split": "train",
                "type": "chat_template",
            }
        ],
        val_set_size=0.05,
        # YAML has both `special_tokens:` (empty) and `eot_tokens:`; in python we just set eot_tokens.
        eot_tokens=["<|end|>", "<|return|>"],

        # Packing / sequence length
        sequence_len=16384,
        sample_packing=True,
        eval_sample_packing=True,

        # Training loop
        micro_batch_size=2,
        num_epochs=1,
        gradient_accumulation_steps=1,
        evals_per_epoch=1,
        max_grad_norm=1.0,
        logging_steps=1,

        # Optim & schedule
        optimizer="adamw_torch",
        learning_rate=1e-4,
        warmup_ratio=0.1,

        # Attention
        flash_attention=True,

        # Deepspeed
        deepspeed=str(Path("deepspeed_configs/zero3_bf16.json")),
        gradient_checkpointing=True,

        # Saving
        save_strategy="no",
        save_only_model=True,

        # Output
        output_dir=str(OUTPUT_DIR / "gpt-oss-20b-deepspeed-zero3"),
    )

    cfg = load_cfg(config)
    dataset_meta = load_datasets(cfg=cfg)
    model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

if __name__ == "__main__":
    main()
