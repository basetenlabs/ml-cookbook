# Fine-tune NVIDIA Nemotron embedding models

NVIDIA's [Nemotron embedding fine-tuning recipe](https://github.com/NVIDIA-NeMo/Nemotron/tree/preview/embed-finetune-recipe-2607/docs/nemotron/embed)
adapts an embedding model to a domain-specific retrieval task. Pre-trained
embeddings often underperform on specialized corpora with their own terminology
and document structure; this recipe closes that gap and typically gains **5–20
points of nDCG@10** over the base model.

The full pipeline is maintained by NVIDIA in the NeMo repo. This page frames how
to run it on a Baseten GPU workstation.

## What it does

An end-to-end, six-stage pipeline:

1. **SDG** — generate synthetic query/answer pairs from your documents.
2. **Data prep** — build training data with hard-negative mining.
3. **Fine-tune** — train the embedding model with contrastive learning.
4. **Eval** — measure retrieval quality with BEIR metrics (nDCG, Recall).
5. **Export** — export the model for deployment when required.
6. **Deploy** — serve via a Retriever NIM or vLLM.

## Hardware

The fine-tune stage needs a single GPU with **80GB+ VRAM** (H100 or A100), 16+
CPU cores, 128GB+ RAM, and ~50GB of free storage. Python 3.12+.

## Run it on Baseten

Because this is a multi-stage, iterative pipeline (generate data, inspect it,
mine negatives, train, eval, repeat), the most common way to run it is
**interactively on an SSH-enabled GPU workstation** rather than as a one-shot
batch job.

**1. Set up SSH once (per machine):**

```bash
truss ssh setup
```

**2. Launch a single-H100 workstation:**

```bash
uvx truss train workstation --accelerator H100 --gpu-count 1
```

This provisions an interactive pod that stays alive until you stop it. Connect
with the SSH command printed in the output:

```bash
ssh training-job-<job-id>-0.ssh.baseten.co
```

**3. On the pod, follow NVIDIA's recipe** to install dependencies and run the
pipeline stages:

**→ [NVIDIA-NeMo/Nemotron · docs/nemotron/embed](https://github.com/NVIDIA-NeMo/Nemotron/tree/preview/embed-finetune-recipe-2607/docs/nemotron/embed)**

**4. Stop the workstation when you're done** — it bills per minute while up:

```bash
truss train stop
```

## Notes

- **Stage 0 (SDG) calls the NVIDIA API.** Export your NVIDIA API key in the pod
  shell (`export NVIDIA_API_KEY=...`) before running the SDG stage.
- **Persist your work to the cache.** The workstation's read-write cache survives
  across sessions — keep datasets, checkpoints, and the cloned recipe there so a
  stopped-and-restarted workstation picks up where you left off.
- The recipe lives on NVIDIA's `preview/embed-finetune-recipe-2607` branch — pin
  to that ref so the entrypoints and CLI flags stay stable.
