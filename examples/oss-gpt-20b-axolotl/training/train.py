import json
import os
from pathlib import Path

from axolotl.utils.dict import DictDefault
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets
from axolotl.train import train

OUTPUT_DIR = os.environ.get("BT_CHECKPOINT_DIR", "/workspace/checkpoints")


def patch_checkpoints(output_dir: str):
    """Patch FSDP architecture names in checkpoints for vLLM compatibility."""
    output_path = Path(output_dir)
    print(f"Patching checkpoints in: {output_path}")

    for config_path in output_path.rglob("config.json"):
        with open(config_path) as f:
            config = json.load(f)

        modified = False

        if "architectures" in config:
            new_archs = [a.replace("FSDP", "") for a in config["architectures"]]
            if new_archs != config["architectures"]:
                print(f"  {config_path}: {config['architectures']} -> {new_archs}")
                config["architectures"] = new_archs
                modified = True

        if "auto_map" in config:
            print(f"  {config_path}: removing auto_map")
            del config["auto_map"]
            modified = True

        if modified:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    print("Done patching checkpoints.")


def main():
    config = DictDefault(
        base_model="openai/gpt-oss-20b",

        # Model quantization
        use_kernels=False,
        model_quantization_config="Mxfp4Config",
        model_quantization_config_kwargs={
            "dequantize": True,
        },

        # Plugins
        plugins=[
            "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
        ],

        experimental_skip_move_to_device=True,  # prevent OOM by NOT putting model to GPU before sharding

        # Datasets
        datasets=[
            {
                "path": "HuggingFaceH4/Multilingual-Thinking",
                "type": "chat_template",
                "field_thinking": "thinking",
                "template_thinking_key": "thinking",
            }
        ],

        dataset_prepared_path="last_run_prepared",
        val_set_size=0.1,
        eval_steps=10,
        output_dir=OUTPUT_DIR,

        # Sequence and packing
        sequence_len=4096,
        sample_packing=True,

        # Training loop
        gradient_accumulation_steps=4,
        micro_batch_size=1,
        num_epochs=1,

        # Optimizer & schedule
        optimizer="adamw_torch_8bit",
        lr_scheduler="constant_with_warmup",
        learning_rate=2e-5,

        # Precision
        bf16=True,
        tf32=True,

        # Attention
        flash_attention=True,
        attn_implementation="kernels-community/vllm-flash-attn3",

        # Checkpointing & offloading
        gradient_checkpointing=True,
        activation_offloading=True,

        # Logging & saving
        logging_steps=1,
        saves_per_epoch=1,

        # Warmup & gradient
        warmup_ratio=0.03,
        max_grad_norm=1.0,

        # Special tokens
        special_tokens={},
        eot_tokens=["<|end|>", "<|return|>"],

        # FSDP configuration
        fsdp_version=2,
        fsdp_config={
            "offload_params": False,
            "state_dict_type": "FULL_STATE_DICT",
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": "GptOssDecoderLayer",
            "reshard_after_forward": True,
        },
    )

    cfg = load_cfg(config)

    dataset_meta = load_datasets(cfg=cfg)

    model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

    # Patch checkpoint names for vLLM compatibility
    patch_checkpoints(OUTPUT_DIR)


if __name__ == "__main__":
    main()
