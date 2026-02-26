# Qwen3 0.6B Classifier with HuggingFace Trainer

This example trains a classifier based on the Qwen3 0.6B model using the HuggingFace Trainer on Baseten.

**Resources:** 1 node, 1x H100 GPU

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   pip install -U truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples qwen3-0.6b-classifier-hf-trainer
cd qwen3-0.6b-classifier-hf-trainer
truss train push config.py
```
