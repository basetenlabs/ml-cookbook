# Fine-tune NVIDIA Nemotron embedding models

NVIDIA's [Nemotron embedding fine-tuning recipe](https://github.com/NVIDIA-NeMo/Nemotron/tree/preview/embed-finetune-recipe-2607/docs/nemotron/embed)
adapts an embedding model to a domain-specific retrieval task. Pre-trained
embeddings often underperform on specialized corpora with their own terminology
and document structure; this recipe closes that gap and typically gains **5–20
points of nDCG@10** over the base model.

The full pipeline is maintained by NVIDIA in the NeMo repo. This page frames how
it maps onto a Baseten training job.

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

Follow NVIDIA's recipe for the pipeline itself:

**→ [NVIDIA-NeMo/Nemotron · docs/nemotron/embed](https://github.com/NVIDIA-NeMo/Nemotron/tree/preview/embed-finetune-recipe-2607/docs/nemotron/embed)**

To run the fine-tune stage as a Baseten training job, wrap the recipe's install
and training commands in a `config.py` / `run.sh` pair (see the
[`isaac-groot`](../isaac-groot) recipe for the pattern) on a single H100, then:

```bash
truss train push config.py
```

## Notes

- **Stage 0 (SDG) calls the NVIDIA API.** Set your NVIDIA API key as a
  [Baseten secret](https://app.baseten.co/settings/secrets) and map it in
  `config.py`.
- The recipe lives on NVIDIA's `preview/embed-finetune-recipe-2607` branch — pin
  to that ref so the entrypoints and CLI flags stay stable.
