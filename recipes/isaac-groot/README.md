# Fine-tune NVIDIA Isaac-GR00T N1 on Baseten

[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) is a ~3B
Vision-Language-Action foundation model for humanoid robots. This recipe
fine-tunes it on a LeRobot-format dataset as a Baseten training job.

Unlike Isaac Lab (which drives Isaac Sim's RTX renderer), GR00T fine-tuning is
pure PyTorch on pre-recorded demonstrations — no simulator, Omniverse, or
rendering stack — so it runs on any GPU with no RT-core requirement.

## Hardware

The default is a single **A10G (24GB)**, using `--no-tune_diffusion_model` so the
fine-tune fits in 24GB. For a full fine-tune that also tunes the diffusion head,
use a 40GB+ GPU (**L40S** or **H100**) and drop that flag, or use LoRA
(`--lora_rank 64`).

## How it works

- **`config.py`** defines the training job: a public PyTorch 2.7 / CUDA 12.8 base
  image, A10G, the read-write cache, and checkpointing.
- **`run.sh`** installs Isaac-GR00T and runs the fine-tune. GR00T pins Python 3.10
  while the base image ships 3.11, so `run.sh` uses `uv` (GR00T's documented env
  manager) to build a 3.10 environment, then installs GR00T and `flash-attn`.
- The recipe trains on the robotics demo dataset bundled in the GR00T repo
  (`demo_data/robot_sim.PickNPlace`). Point `--dataset-path` at your own LeRobot
  v2 dataset (must include `meta/modality.json`).

## Run it

```bash
truss train push config.py
```

For gated Hugging Face datasets or models, add an `hf_access_token` secret in
[Baseten Secrets](https://app.baseten.co/settings/secrets) and map it to
`HF_TOKEN` in `config.py` (a commented example is included). The public GR00T
base model needs no token.

## Notes

- **Pin a version.** `run.sh` clones `main`. Pin to a release tag and match the
  command: N1.5 uses `scripts/gr00t_finetune.py`, N1.7 uses
  `gr00t/experiment/launch_finetune.py`.
- Isaac-GR00T is currently NVIDIA **Early Access** (no commercial deployment until GA).
