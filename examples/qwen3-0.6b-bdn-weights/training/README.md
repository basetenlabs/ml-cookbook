# Qwen3 0.6B fine-tuning with BDN weight loading

Fine-tune Qwen3 0.6B on Baseten using BDN (Baseten Delivery Network) to load model weights. Weights are mirrored and cached through BDN so they're available in your container before training starts. You aren't billed for compute time during data loading.

**Resources:** 1 node, 1x H100 GPU

## How BDN weight loading works

Instead of downloading model weights inside your training script, declare them in `config.py` using `WeightsSource`. BDN mirrors the weights once and caches them close to your training container. On subsequent jobs, cached weights load faster.

```python
weights=[
    WeightsSource(
        source="hf://Qwen/Qwen3-0.6B",
        mount_location="/app/models/Qwen/Qwen3-0.6B",
    ),
]
```

Your training script loads directly from the mount path:

```python
model = AutoModelForCausalLM.from_pretrained("/app/models/Qwen/Qwen3-0.6B")
```

BDN supports Hugging Face (`hf://`), S3 (`s3://`), GCS (`gs://`), and R2 (`r2://`). See the [BDN docs](https://docs.baseten.co/development/model/bdn) for authentication and configuration options.

### Loading training data from S3

To load training data from S3, add another `WeightsSource` entry. For private buckets, add authentication:

```python
WeightsSource(
    source="s3://my-bucket/training-data",
    mount_location="/app/data/training-data",
    auth={"auth_method": "CUSTOM_SECRET", "auth_secret_name": "aws_credentials"},
)
```

The `aws_credentials` secret should contain `{"aws_access_key_id": "...", "aws_secret_access_key": "...", "aws_region": "us-west-2"}`. You can also use [OIDC authentication](https://docs.baseten.co/organization/oidc) to avoid long-lived credentials.

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   # pip
   pip install -U truss
   # or uv
   uv add truss
   ```

## Getting started

Clone, navigate, and push:

```bash
truss train init --examples qwen3-0.6b-bdn-weights
cd qwen3-0.6b-bdn-weights
truss train push config.py
```
