# Whisper Fine-Tuning with Transformers

This example fine-tunes OpenAI's Whisper speech recognition model using the HuggingFace Transformers framework on Baseten.

**Resources:** 1 node, 8x H100 GPUs

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   # pip
   pip install -U truss
   # or uv
   uv add truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples whisper-transformers
cd whisper-transformers
truss train push config.py
```

> **Note:** This example requires more than 4x H100 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.

## CI smoke test

The PR workflow submits `config.py` with `WHISPER_SMOKE_TEST=1` for this
example. It uses the same model and dataset but runs only two optimizer steps,
then evaluates and saves a checkpoint. The normal 5,000-step default is
unchanged. To opt in manually, run `WHISPER_SMOKE_TEST=1 truss train push config.py`.

## Checkpoint storage and optional export

By default, the job saves checkpoints to Baseten and does **not** upload them
to Hugging Face. See [Speech data handling](../../../speech-data-handling.md)
for retention and cleanup considerations.

To opt in to a Hugging Face export, set your approved destination and the name
of its token secret when submitting the job:

```bash
HF_HUB_MODEL_ID=your-org/your-private-model \
HF_TOKEN_SECRET_NAME=hf_access_token \
truss train push config.py
```

Use a private repository you control. The recipe requests private visibility
when creating a repository, but does not change the visibility of an existing
one. To authenticate only for downloads, set `HF_TOKEN_SECRET_NAME` without
setting `HF_HUB_MODEL_ID`.

## Serving smoke test

From the full cookbook checkout, open the example’s `inference/` directory and set `BASETEN_API_KEY`, `BASETEN_MODEL_ID`,
`BASETEN_DEPLOYMENT_ID`, and `SERVED_MODEL_NAME` for your deployment, then run
`python run_inference_example.py`. The bundled demo clip is used unless
`AUDIO_PATH` is set. Set `PRINT_TRANSCRIPT=1` only if you intend to display the
transcript on stdout.
