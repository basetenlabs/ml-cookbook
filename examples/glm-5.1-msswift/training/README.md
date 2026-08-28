# GLM-5.1 Fine-Tuning with MS-Swift

This example fine-tunes the GLM-5.1 model using the MS-Swift framework with MegatronLM on Baseten.

**Resources:** 4 nodes, 8x H200 GPUs each (32 GPUs total)

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
truss train init --examples glm-5.1-msswift
cd glm-5.1-msswift
truss train push config.py
```

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.

## Notes

- GLM-5.1 is a 744B MoE model (256 routed experts) using DeepSeek Sparse Attention (DSA). It needs a `megatron-core >= 0.17.1` image — the tag pinned in `config.py` works as-is.
- The 256 experts shard cleanly only across power-of-2 GPU counts, so this recipe uses 4 nodes (`expert_model_parallel_size=32`). 24 GPUs (3 nodes) cannot fit the weights.
- `max_length` is set to **16384**. Longer sequences are currently limited by the DSA indexer, which materializes an O(seqlen²) score matrix; context-parallel support for DSA (the fix for long context) is not yet available upstream.
