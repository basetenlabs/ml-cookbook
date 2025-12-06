from pathlib import Path
import os

from axolotl.utils.dict import DictDefault
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets
from axolotl.train import train

OUTPUT_DIR = Path(os.environ.get("BT_CHECKPOINT_DIR", "outputs")) / "gemma3-27b-it-fsdp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    config = DictDefault(
        # Model
        base_model="google/gemma-3-27b-it",

        # Dataset / prepared cache
        dataset_prepared_path="./outputs/last_run_prepared",
        datasets=[
            {
                "path": "winglian/pirate-ultrachat-10k",
                "split": "train",
                "type": "chat_template",
            }
        ],
        eot_tokens=["<end_of_turn>"],
        val_set_size=0.05,

        # Packing
        sample_packing=True,
        eval_sample_packing=True,

        # Training loop
        num_epochs=1,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        evals_per_epoch=1,
        logging_steps=1,
        max_grad_norm=1.0,

        # Optim & schedule
        optimizer="adamw_torch_fused",
        sequence_len=16384,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,

        # FSDP
        fsdp_version=2,
        context_parallel_size=8,
        fsdp_config={
            "offload_params": False,
            "cpu_ram_efficient_loading": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": "Gemma3DecoderLayer",
            "state_dict_type": "FULL_STATE_DICT",
            "reshard_after_forward": True,
        },

        # Saving
        save_only_model=True,
        save_strategy="no",

        # Attention
        flash_attention=True,

        # Liger + CutCE plugins
        plugins=[
            "axolotl.integrations.liger.LigerPlugin",
            "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
        ],
        liger_rope=True,
        liger_rms_norm=True,
        liger_glu_activation=True,
        liger_layer_norm=True,
        liger_fused_linear_cross_entropy=False,
        liger_cross_entropy=False,

        # Misc
        output_dir=str(OUTPUT_DIR),
    )

    cfg = load_cfg(config)
    dataset_meta = load_datasets(cfg=cfg)
    model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

if __name__ == "__main__":
    main()
