# Elephant-1 decision-model SFT

Continues training [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1), a 9B decision model, on your own multiple-choice, yes/no or score questions, with the recipe Elephant-1 was trained on. The example uses BANKING77 intent classification.

**Resources:** 4x H100 GPUs (the default self-serve training quota)

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI: `pip install -U truss`

## Getting started

```bash
truss train init --examples elephant-1-sft
cd elephant-1-sft
truss train push config.py
```

To train on your data, edit `prepare_data.py` to write your rows to `train.jsonl` and `test.jsonl`. Each row needs a `state` (string or JSON), a `question` (`choice` with `criteria` {label: description or null}, `noul` for yes/no, or `score` with a list of levels) and a `target` {label: probability}; soft targets are fine. Set `ROWS` in `config.py` to train on more than the first 2,000 rows.

## How it differs from a standard SFT

Elephant-1 reads each answer from the logits of the option letters in one forward pass. Its full-attention layers see the state and the question bidirectionally. `run.sh` trains with `--attention noncausal --readout slot_lm` and a Brier calibration term so the fine-tuned model matches how Elephant-1 was trained and is served.

The training code is [OneJev](https://github.com/OmniJev/OneJev) at a pinned commit, plus the `serve.patch` shipped in the model repo. `setup.sh` from the model repo installs both.

## Deploy

`deploy/config.yaml` serves the checkpoint on the same Infinity server as the Elephant-1 Model API.

1. Replace `abc123` in `deploy/config.yaml` (both places) with your training job ID.
2. Run `truss push deploy --publish`.

The endpoint takes the same `/v1/systemone` requests as the Elephant-1 Model API.

## Details

- **Model:** [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1) (Qwen3.5-9B base)
- **Dataset:** [BANKING77](https://huggingface.co/datasets/mteb/banking77) (CC BY 4.0)
- **Method:** full fine-tune with FSDP, cross-entropy plus Brier loss, learning rate 1e-6
- **Evaluation:** accuracy, NLL and ECE on 500 test rows before and after training, in the job logs
- **Training time:** about 11 minutes on 4x H100 for the default 2,000 rows (24 steps), including setup
