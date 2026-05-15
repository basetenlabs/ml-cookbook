# Qwen3.6-27B Long Context Fine-Tuning with MS-Swift

This example fine-tunes [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) — a 27B-parameter **dense** hybrid linear-attention model — using LoRA with the MS-Swift Megatron backend on Baseten. It uses the [LongAlign-10k](https://huggingface.co/datasets/zai-org/LongAlign-10k) dataset for long-context SFT.

> **Note on architecture:** Qwen3.6-27B and [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) share the same architecture exactly. The compute measurements in this README were collected primarily on Qwen3.5; we re-ran the 1n128k packed config on Qwen3.6-27B and confirmed identical memory footprint (~28 GiB sw) and similar throughput. To use Qwen3.5 instead, just swap the model ID in `run.sh`.

## Architecture vs. Qwen3.6-35B-A3B

|  | Qwen3.6-35B-A3B | Qwen3.6-27B |
|---|---|---|
| Active params | 3B (256-expert MoE) | 27B (dense FFN) |
| Layers | 40 | **64** |
| Hidden | 2048 | **5120** |
| FFN intermediate | 512 (per expert × 256) | **17,408** (dense) |
| Attention pattern | Hybrid GDN + GQA | Hybrid GDN + GQA |
| Q/KV heads | 16 / 2 | **24 / 4** |
| GDN heads (V/QK) | 32 / 16 | **48 / 16** |
| Native context | 262K | 262K |

**Practical implication:** 27B is a heavier dense model — every GPU computes every parameter, no expert sparsity. EP doesn't help here. The natural single-node split is **TP=8** (sharding the 27B weights 8-way), not EP=8.

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

`run.sh` is idempotent: cold first run is ~10 min (50 GB model download + train); subsequent runs in the same project skip the download.

**Optional helper:**

- `config_debug.py` — SSH-enabled `sleep infinity` debug pod:
  ```bash
  truss train push training/config_debug.py --team baseten-dogfood --remote baseten
  # ssh training-job-<JOB_ID>-0.ssh.baseten.co
  # source ~/qwen36_27b_env.sh
  ```

## Verified compute (packed, 1× 8 H200)

| Seq length | TP | PP | EP | CP | Recompute | Peak GiB (sw / nvidia-smi) | Steady s/iter |
|------------|----|----|----|----|-----------|----------------------------|---------------|
| 128K       | 8  | 1  | 1  | 1  | 2 (full)  | 28.4 / 35   | ~88  |
| 262K       | 8  | 1  | 1  | 1  | 2 (full)  | 49.5 / 56   | ~180 |

Both verified end-to-end (5 iters + checkpoint save). Loss curve healthy (~1.17 → 0.88 over 5 iters at 262K).

**Single node is sufficient for both 128K and 262K** with packed sequences. Memory at 262K is only ~40% of H200 capacity — there's substantial headroom for larger batch, longer recompute groups (faster), or even longer context with YaRN.

CP isn't usable on the GDN sublayers without `megatron-core` from git main + `mcore-bridge>=1.1.0`. Since 1 node already handles 262K comfortably, CP isn't needed here.

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
