import os
from pathlib import Path

CACHE_DIR = os.environ.get("BT_PROJECT_CACHE_DIR")
if CACHE_DIR:
    os.environ["HF_HOME"] = CACHE_DIR

# Imported after HF_HOME is set so the hub cache constants resolve to it.
from huggingface_hub import HfApi
from axolotl.utils.dict import DictDefault
from axolotl.cli.config import load_cfg
from axolotl.common.datasets import load_datasets
from axolotl.train import train

OUTPUT_DIR = os.environ.get("BT_CHECKPOINT_DIR", "outputs/qwen3-0.6b")

MODEL_MOUNT_PATH = os.environ["MODEL_MOUNT_PATH"]
MODEL_ID = "Qwen/Qwen3-0.6B"


def seed_hf_cache_from_mount(model_id: str, mount_path: str) -> None:
    """Expose the BDN-mounted weights to HF under the canonical repo id.

    Lets from_pretrained(model_id) load the weights from the mount instead of
    downloading them, while still recording the HF repo id in the saved LoRA
    adapter — a deployable checkpoint needs base_model to be 'namespace/model'
    (resolved from the BDN mirror), not a local path. Recording it at train
    time avoids the post-hoc re-sync/re-discovery race a later rewrite hits."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = hub / f"models--{model_id.replace('/', '--')}"
    commit = HfApi().model_info(model_id).sha
    snapshot = repo_dir / "snapshots" / commit
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(commit)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.symlink_to(mount_path)


def main():
    seed_hf_cache_from_mount(MODEL_ID, MODEL_MOUNT_PATH)
    config = DictDefault(
        adapter="qlora",
        base_model=MODEL_ID,
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
        lora_target_modules="all-linear",

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
