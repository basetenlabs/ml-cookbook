# Elephant-1 decision-model SFT

Continues training [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1), a 9B decision model, on your own multiple-choice, yes/no or score questions, with the recipe Elephant-1 was trained on. The example uses BANKING77 intent classification.

**Resources:** 4x H100 GPUs (the default self-serve training quota)

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Baseten CLI and log in, following the [Training quickstart](https://docs.baseten.co/training/getting-started).

## Getting started

```bash
baseten train init --example elephant-1-sft
cd elephant-1-sft
baseten train push --config config.py
```

To train on your data, edit `prepare_data.py` to write your rows to `train.jsonl` and `test.jsonl`. The `Row` type in `prepare_data.py` defines the row format. Set `ROWS` in `config.py` to train on more than the first 2,000 rows.

## Training recipe

The example uses the recipe Elephant-1 was trained with; the [technical report](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models) covers the full methodology.

- Full fine-tune with FSDP, 1 epoch, learning rate 1e-6 with 3% warmup
- Loss: cross-entropy plus a Brier calibration term (weight 1.0)
- `--attention noncausal --readout slot_lm`: bidirectional attention in the full-attention layers, and answers read from the option-letter logits
- Maximum sequence length 16,384 tokens

`run.sh` runs `setup.sh` from the [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1) Hugging Face repo. It installs [OneJev](https://github.com/OmniJev/OneJev), the training code, at a pinned commit, and applies `serve.patch`, which adds the decision readout and attention mask.

## Deploy

`deploy/config.yaml` serves the checkpoint on the same Infinity server as the Elephant-1 Model API.

1. Replace `abc123` in `deploy/config.yaml` (both places) with your training job ID.
2. Run `baseten model push --dir deploy`.

The endpoint takes the same `/v1/systemone` requests as the Elephant-1 Model API.

## Details

- **Model:** [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1) (Qwen3.5-9B base)
- **Dataset:** [BANKING77](https://huggingface.co/datasets/mteb/banking77) (CC BY 4.0)
- **Evaluation:** accuracy, NLL and ECE on 500 test rows before and after training, in the job logs
- **Training time:** about 11 minutes on 4x H100 for the default 2,000 rows (24 steps), including setup
