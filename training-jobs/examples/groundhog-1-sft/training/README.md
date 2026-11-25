# Groundhog-1 decision-model SFT

Continues training [baselabs/groundhog-1](https://huggingface.co/baselabs/groundhog-1), a 26B mixture-of-experts decision model with about 4B active parameters per token, on your own multiple-choice, yes/no or score questions, with the recipe Groundhog-1 was trained on. For a cheaper run that fits the default quota, use the 9B [`elephant-1-sft`](../../elephant-1-sft/training) example instead. The example uses BANKING77 intent classification.

**Resources:** 8x H100 GPUs. A full fine-tune of the 26B model holds about 416 GB of weights, gradients and optimizer state, which 4 H100s cannot fit. The default self-serve training quota is 4 H100s; for a higher quota, [reach out to us](mailto:support@baseten.co).

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Baseten CLI and log in, following the [Training quickstart](https://docs.baseten.co/training/getting-started).

## Getting started

```bash
baseten train init --example groundhog-1-sft
cd groundhog-1-sft
baseten train push --config config.py
```

To train on your data, edit `prepare_data.py` to write your rows to `train.jsonl` and `test.jsonl`. The `Row` type in `prepare_data.py` defines the row format. Training uses the first `ROWS` rows (2,000 by default; set it in `config.py`), so shuffle your rows first.

## Training recipe

The example uses the recipe Groundhog-1 was trained with; the [model card](https://huggingface.co/baselabs/groundhog-1) covers the full methodology.

- Full fine-tune with FSDP, 1 epoch, learning rate 1e-6 with 3% warmup
- Loss: cross-entropy plus a Brier calibration term (weight 1.0)
- `--attention noncausal --readout slot_lm`: bidirectional attention in the full-attention layers, and answers read from the option-letter logits
- `--freeze-router --freeze-norms-embeds`: the expert routers, norms and token embeddings stay frozen, as in Groundhog-1's training; the vision encoder is always frozen
- Maximum sequence length 16,384 tokens

`run.sh` runs `setup.sh` from the [baselabs/groundhog-1](https://huggingface.co/baselabs/groundhog-1) Hugging Face repo. It installs [OneJev](https://github.com/OmniJev/OneJev), the training code, at a pinned commit, and applies `serve.patch`, which adds the decision readout, the attention mask and Gemma 4 support. `run.sh` then applies `fsdp_setup_memory.patch`, which stops OneJev holding a full copy of the model on every GPU during setup; without it the 26B model does not fit on H100s.

### Longer contexts

Memory grows quickly with sequence length. On one node of 8 GPUs, these lengths have been tested with full-length rows:

| GPUs | Tested maximum | Peak memory per GPU |
|---|---|---|
| 8x H100 | 24,576 tokens | 73.6 GB (needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) |
| 8x H200 | 65,536 tokens | 114.9 GB |

To train on longer rows, add `--max-len N --batch-tokens N` to the `torchrun` command in `run.sh`. To go past 16,384 tokens on H100s, also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `config.py`'s environment variables. Without it, 24,576 tokens runs out of memory. Groundhog-1 itself was trained at 131,072 tokens, which needs more than one node.

## Deploy

`deploy/config.yaml` serves the checkpoint on one H100 with the public [Infinity](https://github.com/michaelfeil/infinity) server.

1. Replace `abc123` in `deploy/config.yaml` (both places) with your training job ID.
2. Run `baseten model push --dir deploy`.

The endpoint takes the same `/v1/systemone` requests as the Groundhog-1 Model API.

## Details

- **Model:** [baselabs/groundhog-1](https://huggingface.co/baselabs/groundhog-1) (Gemma 4 26B A4B base)
- **Dataset:** [BANKING77](https://huggingface.co/datasets/mteb/banking77) (CC BY 4.0)
- **Evaluation:** accuracy, NLL and ECE on 500 test rows before and after training, in the job logs
- **Training time:** about 7 minutes on 8x H100 for the default 2,000 rows (15 steps), including setup. The first run takes about 9 minutes longer while it downloads the 53 GB model into the project cache.
