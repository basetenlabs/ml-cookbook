# Qwen3-30B-A3B Multinode Training

Qwen3 30B is a powerful model with the capability to many complex tasks. In this example, we provide a working script that leverages multinode training to scale out the sequence length, allowing you to train on incredibly large inputs. 

## Details 
This script uses [Megatron-SWIFT](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html), or ms-swift, which builds on top of optimizations that come as part of Nvidia's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). 

We traint Qwen3-30B-A3B with a sequence length of 32000 on the [LongAlign dataset](https://huggingface.co/datasets/zai-org/LongAlign-10k) on 2 H100 nodes (16 chips). Doing this helps the model complete tasks that require comprehension across larger inputs, like document understanding or information extraction. 

The base image used is home-rolled by Baseten, which makes ms-swift easier to use off the shelf. 

## Access
Reach out to Baseten to get access to Multinode Training!