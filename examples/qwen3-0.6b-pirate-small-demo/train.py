from pathlib import Path
import os

from axolotl.utils.dict import DictDefault
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets
from axolotl.train import train

OUTPUT_DIR = os.environ["BT_CHECKPOINT_DIR"] if "BT_CHECKPOINT_DIR" in os.environ else "outputs/qwen3-0.6b"

def main():
    config = DictDefault(
        adapter="qlora",
        base_model="Qwen/Qwen3-0.6B",
        bf16=True,
        # chat_template="tokenizer_default_fallback_chatml",

        # Data loader tweaks
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=8,

        datasets=[
            {
                "path": "winglian/pirate-ultrachat-10k",
                "type": "chat_template",
                "field_messages": "messages",
            }
        ],

        # Eval/val
        val_set_size=0.05,
        eval_steps=10,

        # LoRA / QLoRA
        load_in_4bit=True,
        lora_alpha=32,
        lora_r=64,
        lora_mlp_kernel=True,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj",
        ],

        # Optim & schedule
        optimizer="adamw_torch",
        learning_rate=1e-3,
        lr_scheduler="cosine",
        warmup_steps=5,
        max_grad_norm=0.1,

        # Training loop
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_epochs=1,
        max_steps=20,
        sequence_len=2048,

        # Misc
        embeddings_skip_upcast=True,
        logging_steps=1,
        output_dir=OUTPUT_DIR,
        saves_per_epoch=2,
        sample_packing=True,
        attn_implementation="flash_attention_2",

        # Plugins
        plugins=[
            "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
        ],

        # DeepSpeed (path can be relative)
        deepspeed=str(Path("deepspeed_configs/zero1.json")),
    )

    cfg = load_cfg(config)

    dataset_meta = load_datasets(cfg=cfg)

    model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

if __name__ == "__main__":
    main()