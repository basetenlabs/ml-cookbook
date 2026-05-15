# Qwen3.6-35B-A3B Long Context Fine-Tuning with MS-Swift

This example fine-tunes [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — a 35B-total / 3B-active hybrid linear-attention MoE — using LoRA with the MS-Swift Megatron backend on Baseten. It uses the [LongAlign-10k](https://huggingface.co/datasets/zai-org/LongAlign-10k) dataset for long-context SFT.

> **Note on architecture:** Qwen3.6-35B-A3B and [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) share the same architecture exactly. The compute measurements in this README were collected primarily on Qwen3.5; we re-ran the 1n128k packed config on Qwen3.6-35B-A3B and confirmed identical memory footprint (~122 GiB sw) and similar throughput. To use Qwen3.5 instead, just swap the model ID in `run.sh`.

## TL;DR — minimum compute by sequence length

The measurements below are split between **packed** (sequences truly fill `max_length`, the rigorous test) and **unpacked** (each LongAlign-10k sample at its actual length, ≤64K).

### Packed (real `max_length` budget — what you actually want)

| Seq length | Min compute | TP | PP | EP | CP | Recompute | Peak GiB (sw / nvidia-smi) | Steady s/iter |
|------------|-------------|----|----|----|----|-----------|----------------------------|---------------|
| 128K       | **1 × 8 H200** | 1 | 1 | 8 | 1 | **2** (full) | 131 / — | **~34** |

Recompute sweep on the same 1×8H200 setup (packed 128K):

| `recompute_num_layers` | Peak GiB (sw) | Steady s/iter | Notes |
|------------------------|---------------|---------------|-------|
| 4 (default in upstream Qwen3.6 doc) | OOM | — | Fails on 1 node |
| **2** | **131** | **34** | Sweet spot — fastest config that fits |
| 1 (every layer) | 122.5 | 50 | Most memory headroom; 1.5× slower |


### Unpacked (sequences ≤ longest LongAlign-10k sample, ~64K in practice)

| Seq length (cfg) | Nodes × GPUs | Recompute | Peak GiB (sw / rank 0) | s/iter |
|------------------|--------------|-----------|-----------------------|--------|
| 8K  | 1 × 8 H200 | 4 (full) | 21.5 / — | 31 |
| 128K | 1 × 8 H200 | 4 (full) | 29.7 / 98 | 39 |
| 262K | 1 × 8 H200 | 4 (full) | 29.8 / 96 | ~16 |

Unpacked numbers are useful as a sanity baseline but **don't prove** training fits at the labeled `max_length` — actual sequence length comes from the data.

### Why CP isn't on the list

The natural way to scale long context is **context parallelism** (split the sequence across GPUs), but ms-swift's `mcore-bridge` 1.2.1 explicitly raises `AssertionError: Gated delta net does not support context parallel for now`. Per the [ms-swift Qwen3.5 best-practice doc](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html) (Qwen3.6 shares this architecture), CP-on-GDN requires `megatron-core` from git main; the stable 0.16.1 we install isn't enough. For now we substitute **PP** for the multi-node dim. Switching to git-main `megatron-core` would likely halve the min-node count for both 128K and 262K.

## About the model

Qwen3.6-35B-A3B is **not** standard transformer attention. Its 40 layers follow the
hybrid pattern `10 × [3 × Gated DeltaNet → MoE + 1 × Gated Attention → MoE]` —
30 GatedDeltaNet (linear-attention) sublayers and 10 GQA full-attention sublayers,
each followed by a 256-expert MoE block (8 routed + 1 shared). Native context is
**262K**, extensible to ~1M with YaRN. It's also a vision-language model, so
`--freeze_vit true --freeze_aligner true` is required for text-only LoRA SFT.

## Software stack

The base image (`baseten/megatron:py3.11.11-cuda12.8.1-torch2.9.1-fa2.8.3-megatron0.16.1-msswift4.1`) ships with everything the megatron sft training path needs — no in-script pip installs:

| Package | Version |
|---------|---------|
| `ms-swift` | 4.1.3 |
| `transformers` | 5.6.2 |
| `mcore-bridge` | 1.2.1 |
| `megatron-core` | 0.16.1 |
| `flash-linear-attention` | 0.5.0 |
| `peft` | 0.19.1 |
| `torch` | 2.9.1 (cu12.8) |
| `flash-attn` | 2.8.3 |

`run.sh` only pre-warms the HF model snapshot and then runs `megatron sft` directly.

## Verified configs

The `config_*.py` files each set `node_count` and the parallelism env vars. The same `run.sh` is used for all of them — it reads `$EXP_TAG`, `$MAX_LENGTH`, `$TP/$PP/$EP/$CP`, etc. from the environment.

| File | Nodes | Seq | Packing | Status |
|------|-------|-----|---------|--------|
| `config_1node_128k.py` | 1 | 128K | true | ✅ verified packed end-to-end |
| `config_debug.py` | 1 | n/a (sleep ∞) | n/a | SSH-enabled debug pod for interactive iteration |

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI (≥0.17 for `interactive_session` SSH support):
   ```bash
   pip install -U "truss>=0.17"
   ```
3. Add your HuggingFace token as a Baseten secret named `hf_access_token` in your [workspace secrets](https://app.baseten.co/settings/secrets).

## Getting Started

**Single command — downloads model, trains:**
```bash
truss train push training/config_1node_128k.py --team baseten-dogfood --remote baseten
```

`run.sh` is idempotent: cold first run is ~12 min (70 GB model download + train); subsequent runs in the same project skip the download and go straight to training.

**Optional helper:**

- `config_debug.py` — 1-node `sleep infinity` pod with SSH enabled, for interactive iteration:
  ```bash
  truss train push training/config_debug.py --team baseten-dogfood --remote baseten
  # wait for "Debug pod ready" in logs, then:
  ssh training-job-<JOB_ID>-0.ssh.baseten.co
  source ~/qwen36_env.sh
  ```

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
