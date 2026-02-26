# Long Context Training with Qwen3-30B-A3B Multinode

[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) is a powerful model with the capability to many complex tasks. In this example, we provide a working script that leverages multinode training to scale out the sequence length, allowing you to train on incredibly large inputs. 

## Details 
This script uses [Megatron-SWIFT](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html), or ms-swift, which builds on top of optimizations that come as part of Nvidia's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). 

We train Qwen3-30B-A3B with a sequence length of 64000 on the [LongAlign dataset](https://huggingface.co/datasets/zai-org/LongAlign-10k) on 2 H100 nodes (16 GPUs). Doing this helps the model complete tasks that require comprehension across larger inputs, like document understanding or information extraction. 

The base image used is home-rolled by Baseten, which makes ms-swift easier to use off the shelf. 

## Running the example

### Install `truss` 
Use the appropriate command for your package manager
```bash
# pip
pip install -U truss
# uv
uv add truss && uv sync --upgrade-package truss
```

### Create the workspace for your training project

```bash
# for the multinode example
truss train init --examples qwen3-30b-mswift-multinode && cd qwen3-30b-mswift-multinode
```

### Kick off the job

Make sure you've plugged in proper secrets (e.g. Hugging Face token) via Baseten Secrets and Environment Variables, and kick off your job

```bash
truss train push config.py
```

For more details, take a look at the [docs](https://docs.baseten.co/training/overview)

## Access
Reach out to Baseten to get access to Multinode Training!