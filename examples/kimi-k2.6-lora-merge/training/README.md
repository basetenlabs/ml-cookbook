# Kimi-K2.6 LoRA Merge

Merges a Loops LoRA adapter into the quantized Kimi-K2.6 base, producing a
single deployable checkpoint that serves on **stock vLLM with no patches**.

The Loops sampler adapter targets attention projections + `lm_head`, which the
Kimi-K2.6 INT4 release stores as **BF16** (only the MoE experts are packed
INT4). So the merge happens **in place on CPU** — no dequantize/requantize — and
the experts and `quantization_config` are carried through untouched.

**Resources:** 1x H100 GPU. The GPU is idle; H100 is used only for its large
node-local disk to hold the ~555 GB base mirror. The merge runs on CPU and
peaks at ~50 GB RAM.

## What it does

1. `load_checkpoint_config` pulls the Loops **sampler** adapter via the
   user-facing checkpoint loader (`LoopsCheckpoint.from_checkpoint(...)`, which
   resolves to `bt://loops:...` server-side — no raw S3 URL).
2. `WeightsSource` mirrors the quantized `moonshotai/Kimi-K2.6` base to
   `/app/base`.
3. `merge.py` rewrites only the shards holding the adapted BF16 tensors
   (`W += (alpha/r)·BᵀA`) and copies everything else through.
4. The merged checkpoint is written to `$BT_CHECKPOINT_DIR/merged`.

## Prerequisites

1. A Baseten account and the Truss CLI (`pip install -U truss`).
2. An `hf_access_token` secret in your org for the base download.
3. Set `LOOPS_RUN_ID` / `LOOPS_CHECKPOINT_NAME` in `config.py` to your Loops run
   and its `sampler` checkpoint name.

## Getting started

```bash
truss train init --examples kimi-k2.6-lora-merge
cd kimi-k2.6-lora-merge
truss train push config.py
```

Monitor:

```bash
truss train logs --job-id <JOB_ID> --tail
```

When the job finishes, the merged checkpoint is at `$BT_CHECKPOINT_DIR/merged` —
a complete HF checkpoint (shards, `config.json` with the original
`quantization_config`, tokenizer) ready to deploy.

## Notes

- `merge.py` validates that every adapter target lands on a BF16 `.weight` and
  **fails loudly** if any target is a packed (quantized) tensor — that would
  mean the adapter touched the MoE experts and requires the slower
  dequantize → merge → requantize path instead.
- The merge is I/O-bound (reading/writing ~555 GB), not compute-bound; tune
  `--workers` in `run.sh` to your instance's CPU/RAM.
