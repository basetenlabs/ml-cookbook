# Fine-tune Nemotron for support routing with Baseten Loops

An end-to-end guide to fine-tune [NVIDIA Nemotron 3.5 Super VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) for decision classification using LoRA on Baseten Loops.

## The problem and dataset

Language models can be effective tools in categorizing user queries to route requests to specialized agents and models. Open source language models can be fine-tuned to generate specifc labels for common queries with a small dataset and few GPU resources.

One specific use case is to route user queries for specialized financial services. A support router must distinguish similar requests—such as a pending transfer versus a failed transfer—and return a label that another application can use. The required output is only:

```json
{"intent":"transaction_charged_twice"}
```

This recipe uses **BANKING77**, a public dataset of banking-support queries with 77 intents. The original dataset contains 10,003 training and 3,080 test examples.[1] The prompt includes the full allowed-label inventory; training teaches the model the task-specific distinctions and exact JSON output contract.

Our preparation script removes four same-label duplicates before splitting the original training data:

| Split | Examples | Purpose |
|---|---:|---|
| Training | 9,229 | Optimizer updates |
| Validation | 770—ten per intent | Identical before/after evaluation |

The split uses seed 1111 and groups lowercase/whitespace-normalized duplicate text before splitting. Original text and label spelling are preserved. **The official test set is not downloaded or evaluated.**

## 0. Prerequisites

Prior to following this guide, ensure you have access to the [Baseten Loops](https://docs.baseten.co/loops/overview) platform and have a valid API key for the platform.

Optionally, you can also monitor the job with Weights and Biases (W&B). You will need a W&B API key to track the job on their platform.

## 1. Set up

In a local terminal, clone this repository and navigate to the source directory:

```bash
git clone https://github.com/basetenlabs/ml-cookbook
cd ml-cookbook/loops/sft/nemotron35_super_banking77
```

You should see the following three scripts in addition to this README.md file:

- [`prepare_banking77.py`](prepare_banking77.py): download and format data.
- [`train_lora.py`](train_lora.py): generic chat-JSONL LoRA training.
- [`evaluate_banking77.py`](evaluate_banking77.py): strict base/adapter intent evaluation.

Install [uv](https://docs.astral.sh/uv/) and use a Python 3.12+ environment. For a fresh checkout:

```bash
uv venv --python 3.12
uv pip install 'baseten-loops==0.28.0' 'transformers==5.19.0' \
  'jinja2==3.1.6' 'requests==2.34.2' 'wandb==0.30.0'
```

Skip `uv venv` if you already have an environment. Set `BASETEN_API_KEY` securely in your shell. For W&B, also configure `WANDB_API_KEY` or a valid cached W&B login. Never put credentials in the scripts or commit them.

Choose a model enabled for your Baseten workspace. This recipe assumes the following published Loops model ID:[2]

```bash
export MODEL=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16
```

No local GPU, model weights, or manual tokenizer download is required. The SDK loads tokenizer assets automatically; the client needs network access and must stay running. The tokenizer-only warning that PyTorch is absent is expected.

## 2. Prepare the data

```bash
uv run prepare_banking77.py
```

The script downloads checksum-verified sources at commit `57ec275d8078af65b7731c2a98be812d844a6d6b` and writes:

| File under `data/` | Contents |
|---|---|
| `banking77-train.jsonl` | Training chat messages, including the target assistant answer |
| `banking77-validation.jsonl` | Validation chat messages, including the reference answer |
| `banking77-validation-gold.jsonl` | The same validation IDs with original text and labels |
| `banking77-labels.json` | Exact allowed intent labels |
| `banking77-source/` | Cached source files and license |

Preparation is safe to repeat: identical outputs are verified, not rewritten; differing existing files are rejected. The evaluator reads references from the validation chat file, so the separate gold file is for inspection or another scorer. **It removes the final assistant answer before sending the prompt to the model.**

## 3. Evaluate the base model

Establish a baseline before training:

```bash
uv run evaluate_banking77.py \
  --base-only \
  --model "$MODEL" \
  --output runs/banking77-base-01
```

This provisions only a base sampler, runs all 770 validation questions, and shuts it down. Read `runs/banking77-base-01/scores.json` for:

- **Accuracy:** correct intent / all examples; invalid outputs count wrong.
- **Macro-F1:** equal-weight average across the 77 intents.
- **Valid JSON:** exactly one object with one `intent` key and an allowed label.

There is no constrained JSON decoding, answer repair, label normalization, or LLM judge. Temperature is zero, thinking is disabled, and each answer has a 256-token budget. Failed or truncated requests prevent a complete score rather than being silently omitted.

The Nemotron 3.5 Super VL model scores 76.49% accuracy out-of-the-box on this dataset which is good but post-training can take it even higher.

## 4. Train the LoRA adapter

After evaluating the base model, it's time to train a LoRA adapter on the dataset to teach the model the semantics and underlying decisions required to generate the labels. Launch a training job on the Baseten platform with:

```bash
uv run train_lora.py \
  --data data/banking77-train.jsonl \
  --model "$MODEL" \
  --output runs/banking77-train-01 \
  --wandb-project banking77-lora
```

Add `--wandb-entity YOUR_TEAM` if needed. Omit `--wandb-project` to disable tracking; an entity alone does not enable it.

Defaults are one epoch, batch size 8, rank 8, maximum sequence length 1,024, and a **constant learning rate of `1e-4`**. This dataset yields 1,154 optimizer updates, including a final partial batch. Only the final assistant response and native end marker receive supervision; system/user tokens are masked. The script rejects overlength examples rather than truncating them.

To change an experiment, pass `--learning-rate`, `--epochs`, `--batch-size`, `--rank`, or `--max-seq-len`. Choose a new output directory each time; existing outputs are never overwritten. This minimal recipe has no scheduler, warmup, checkpoint-selection sweep, or automatic resume.

## 5. Monitor training and confirm the save

The terminal prints `Training X/1154: loss=...`. With W&B enabled it also prints a run URL, where you can view **loss versus step** and learning rate. These are training curves, not held-out accuracy.

From another terminal, follow the local records:

```bash
tail -f runs/banking77-train-01/training.jsonl
```

After the final update the script saves resumable training state and sampler weights, prints both `bt://` URIs, and deactivates the run. Wait for **“Baseten shutdown verified.”**

Check the local artifacts:

- `run.json`: Baseten run/session IDs.
- `training.jsonl`: completed optimizer steps and actual losses.
- `checkpoints.json`: `state`, `weights`, and `run_id` for later use.
- `cleanup.json`: `verified: true` after shutdown readback.

**Checkpoints are saved at the end, not during the epoch.** If you stop early, no final checkpoint is guaranteed. A completed update counter is not confirmation that saving and cleanup finished.

## 6. Evaluate the trained adapter

```bash
uv run evaluate_banking77.py \
  --checkpoint runs/banking77-train-01/checkpoints.json \
  --model "$MODEL" \
  --output runs/banking77-comparison-01
```

For a matched comparison, this command **reruns the base**, shuts its sampler down, then loads and evaluates the adapter on the same questions. It does not retrain. Cold starts, weight loading, and shutdown can take time between the `base: 770/770` and adapter progress messages.

Read `runs/banking77-comparison-01/scores.json` for both arms, the accuracy difference, paired gains/regressions, and a 95% paired bootstrap interval. `base.jsonl` and `adapter.jsonl` preserve the raw predictions. Require both `base-cleanup.json` and `adapter-cleanup.json` to show `verified: true`.

### Measured example—not a guaranteed result

A completed run with this constant-LR setup produced the following on the 770-example validation split:

| Metric | Base | LoRA |
|---|---:|---:|
| Correct intents | 589 / 770 | 718 / 770 |
| Accuracy | 76.49% | 93.25% |
| Macro-F1 | 0.7578 | 0.9314 |
| Valid allowed-label JSON | 770 / 770 | 770 / 770 |

Accuracy improved **16.75 percentage points**, with 137 gains and eight regressions; the paired 95% interval was **+14.03 to +19.61 points**. The saved report is [`scores.json`](runs/eval-20261029-130532-197684/scores.json). Model/backend changes can change results.

These are **validation results, not official BANKING77 test accuracy**. Public benchmark contamination and ambiguous queries remain possible. BANKING77 has no out-of-scope class; this recipe does not establish safe abstention, financial advice, or production readiness. Keep the validation examples out of optimizer updates and use an independent final holdout before deployment.

## Attribution

BANKING77: PolyAI / Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020), **CC BY 4.0**.[1] This recipe derives a deduplicated split and chat formatting; source query text is unchanged. Review the model and platform terms separately.

## Sources

[1] https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b
[2] https://docs.baseten.co/loops/supported-models
[3] https://docs.baseten.co/loops/quickstart
