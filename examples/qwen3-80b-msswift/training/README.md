# Qwen3-Coder-Next Long Context Fine-Tuning with MS-Swift

This example fine-tunes the [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) (80B MoE) model using LoRA with the MS-Swift framework and MegatronLM on Baseten. It uses the [LongAlign-10k](https://huggingface.co/datasets/zai-org/LongAlign-10k) dataset for long-context supervised fine-tuning.

**Default configuration:** 1 node, 8x H200 GPUs, 32K sequence length

## Tested Configurations

| Nodes | GPUs | Seq Length | TP | PP | EP | LoRA Rank | Recompute Layers | Peak Memory (GiB) | Time/Iter |
|-------|------|------------|----|----|----|-----------|------------------|--------------------|-----------|
| 1     | 8    | 16K        | —  | —  | 8  | 64        | 4                | 105                | ~35s      |
| 1     | 8    | 32K        | —  | —  | 8  | 8         | 2                | 121                | ~40s      |
| 4     | 32   | 48K        | 2  | —  | 16 | 8         | 1                | 98                 | ~270s     |
| 4     | 32   | 64K        | 2  | 2  | 4  | 8         | 1                | 118                | ~455s     |

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
truss train init --examples qwen3-80b-msswift
cd qwen3-80b-msswift
truss train push training/config.py
```

### Scaling to longer sequences

To train at longer sequence lengths, increase `node_count` in `config.py` and adjust parallelism flags in `run.sh`. For example, for 64K on 4 nodes:

```python
# config.py
node_count=4
```

```bash
# run.sh — key flags to change
--tensor_model_parallel_size 2
--pipeline_model_parallel_size 2
--expert_model_parallel_size 4
--global_batch_size 32
--recompute_num_layers 1
--max_length 64000
```

> **Note:** This example requires H200 GPUs. You may need to [contact Baseten](https://www.baseten.co/contact) to get approval for this instance type before running the job.
