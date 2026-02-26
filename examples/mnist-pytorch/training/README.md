# MNIST Digit Classifier with PyTorch

This example trains an MNIST digit classifier using PyTorch on Baseten.

**Resources:** CPU only (4 vCPU, 16Gi memory)

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   pip install -U truss
   ```

## Getting Started

Initialize the example, navigate into the directory, and push the training job:

```bash
truss train init --examples mnist-pytorch
cd mnist-pytorch
truss train push config.py
```
