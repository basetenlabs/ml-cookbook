# Qwen3.5-35B-A3B Long Context Fine-Tuning with MS-Swift

This example fine-tunes [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) — a 35B-total / 3B-active hybrid linear-attention MoE — using LoRA with the MS-Swift Megatron backend on Baseten. It uses the [LongAlign-10k](https://huggingface.co/datasets/zai-org/LongAlign-10k) dataset for long-context SFT.

## TL;DR — minimum compute by sequence length

The measurements below are split between **packed** (sequences truly fill `max_length`, the rigorous test) and **unpacked** (each LongAlign-10k sample at its actual length, ≤64K).

### Packed (real `max_length` budget — what you actually want)

| Seq length | Min compute | TP | PP | EP | CP | Recompute | Peak GiB (sw / nvidia-smi) | Steady s/iter |
|------------|-------------|----|----|----|----|-----------|----------------------------|---------------|
| 128K       | **1 × 8 H200** | 1 | 1 | 8 | 1 | **2** (full) | 131 / — | **~34** |
| 262K       | TBD          | — | — | — | — | — | — | — |

Recompute sweep on the same 1×8H200 setup (packed 128K):

| `recompute_num_layers` | Peak GiB (sw) | Steady s/iter | Notes |
|------------------------|---------------|---------------|-------|
| 4 (default in upstream Qwen3.5 doc) | OOM | — | Fails on 1 node |
| **2** | **131** | **34** | Sweet spot — fastest config that fits |
| 1 (every layer) | 122.5 | 50 | Most memory headroom; 1.5× slower |

For comparison, `2 × 8 H200 with PP=2 + recompute=4` also works at ~71 s/iter — but it's strictly worse than the 1-node config above because PP=2 pays pipeline-bubble + cross-node communication overhead (we're at `global_batch_size=8 / micro_batch_size=1`, so only 8 microbatches per pipeline schedule).

### Unpacked (sequences ≤ longest LongAlign-10k sample, ~64K in practice)

| Seq length (cfg) | Nodes × GPUs | Recompute | Peak GiB (sw / rank 0) | s/iter |
|------------------|--------------|-----------|-----------------------|--------|
| 8K  | 1 × 8 H200 | 4 (full) | 21.5 / — | 31 |
| 128K | 1 × 8 H200 | 4 (full) | 29.7 / 98 | 39 |
| 262K | 1 × 8 H200 | 4 (full) | 29.8 / 96 | ~16 |

Unpacked numbers are useful as a sanity baseline but **don't prove** training fits at the labeled `max_length` — actual sequence length comes from the data.

### Why CP isn't on the list

The natural way to scale long context is **context parallelism** (split the sequence across GPUs), but ms-swift's `mcore-bridge` 1.2.1 explicitly raises `AssertionError: Gated delta net does not support context parallel for now`. Per the [ms-swift Qwen3.5 best-practice doc](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html), CP-on-GDN requires `megatron-core` from git main; the stable 0.16.1 we install isn't enough. For now we substitute **PP** for the multi-node dim. Switching to git-main `megatron-core` would likely halve the min-node count for both 128K and 262K.

## About the model

Qwen3.5-35B-A3B is **not** standard transformer attention. Its 40 layers follow the
hybrid pattern `10 × [3 × Gated DeltaNet → MoE + 1 × Gated Attention → MoE]` —
30 GatedDeltaNet (linear-attention) sublayers and 10 GQA full-attention sublayers,
each followed by a 256-expert MoE block (8 routed + 1 shared). Native context is
**262K**, extensible to ~1M with YaRN. It's also a vision-language model, so
`--freeze_vit true --freeze_aligner true` is required for text-only LoRA SFT.

## Software stack (auto-installed by `run.sh`)

The base image (`baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3`) ships an older toolchain. `run.sh` upgrades the following into `$BT_PROJECT_CACHE_DIR/qwen3_5_packages` (cached across runs, takes ~3 min on first run, ~5 s thereafter):

| Package | Version | Why |
|---------|---------|-----|
| `ms-swift` | ≥4.1.0 | Adds `qwen3_5_moe` model registration |
| `transformers` | ==5.2.* | Has the Qwen3.5 modeling code |
| `huggingface_hub` | ≥1.3.0,<2.0 | transformers 5.2 needs `is_offline_mode` (1.x re-exports it) |
| `tokenizers` | 0.22.x–0.23.0 | transformers 5.2 dep |
| `safetensors` | ≥0.4.3 | transformers 5.2 dep |
| `accelerate` | ≥1.1.0 | transformers 5.2 dep |
| `peft` | ≥0.13 | LoRA |
| `liger-kernel` | latest | Recommended in upstream Qwen3.5 doc |
| `qwen_vl_utils` | ≥0.0.14 | VL preprocessing |
| `mcore-bridge` | ≥1.0.2 | ms-swift 4.1's `swift.megatron` hard-requires it |
| `torchao` | ≥0.16 | transformers 5.2 needs this for fp8 paths |
| `tilelang` | latest | FLA workaround for Triton ≥3.4 / Hopper bug ([fla #640](https://github.com/fla-org/flash-linear-attention/issues/640)) |
| `megatron-core` | ≥0.16 | mcore-bridge 1.2.x uses 0.16+ sharding APIs (`dp_reshardable`) |
| `flash-linear-attention` | git main | PyPI's 0.5.0 is missing `fla.ops.utils`; transformers' Qwen3.5 imports it |
| `causal-conv1d` | git main | GDN dependency, built from source (~5-10 min CUDA compile) |

### Why FLA from git, not PyPI

PyPI's `flash-linear-attention==0.5.0` ships without the `fla.ops.utils` submodule that transformers 5.2's Qwen3.5 implementation imports. Installing from `https://github.com/fla-org/flash-linear-attention.git` yields `0.5.1+` which has it.

### Why `megatron-core>=0.16` instead of the image's 0.14.1

`mcore-bridge` 1.2.x calls `distrib_optimizer.sharded_state_dict()` with `sharding_type='dp_reshardable'`, a code path added in megatron-core 0.16. The image's 0.14.1 raises `NotImplementedError`. We install 0.16.1 into `PKG_DIR` and rely on `PYTHONPATH=$PKG_DIR:...` to shadow the system version.

## Verified configs

The `config_*.py` files each set `node_count` and the parallelism env vars. The same `run.sh` is used for all of them — it reads `$EXP_TAG`, `$MAX_LENGTH`, `$TP/$PP/$EP/$CP`, etc. from the environment.

| File | Nodes | Seq | Packing | Status |
|------|-------|-----|---------|--------|
| `config.py` | 1 | n/a (hydrate) | n/a | ✅ pre-warms deps + model cache |
| `config_debug.py` | 1 | n/a (sleep ∞) | n/a | ✅ SSH-enabled debug pod (`session_provider=SSH`) |
| `config_debug_2node.py` | 2 | n/a (sleep ∞) | n/a | ✅ 2-node SSH debug pod (used to verify 2-node 128K) |
| `config_1node_128k.py` | 1 | 128K | false | ✅ runs (LongAlign samples ≤64K so 128K not exercised) |
| `config_2node_128k.py` | 2 | 128K | **true** | ✅ **verified packed 128K end-to-end** |

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI (≥0.17 for `interactive_session` SSH support):
   ```bash
   pip install -U "truss>=0.17"
   ```
3. Add your HuggingFace token as a Baseten secret named `hf_access_token` in your [workspace secrets](https://app.baseten.co/settings/secrets).

## Getting Started

**Single command — installs deps, downloads model, trains:**
```bash
truss train push training/config_1node_128k.py --team baseten-dogfood --remote baseten
```

`run.sh` checks the project cache first and only re-installs / re-downloads on a cold cache. First run is ~15 min (deps + 70 GB model + train); subsequent runs in the same project skip straight to training.

**Optional helpers** (only useful in specific scenarios):

- `config.py` — runs the dep install + model snapshot then exits (`HYDRATE_ONLY=1`). Use when fanning out **parallel** experiments — otherwise multiple jobs would race writing into the shared `$BT_PROJECT_CACHE_DIR`.
- `config_debug.py` — 1-node `sleep infinity` pod with SSH enabled, for interactive iteration:
  ```bash
  truss train push training/config_debug.py --team baseten-dogfood --remote baseten
  # wait for "Debug pod ready" in logs, then:
  ssh training-job-<JOB_ID>-0.ssh.baseten.co
  source ~/qwen35_env.sh
  ```

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
