# Fine-tune NVIDIA Isaac-GR00T N1 (Vision-Language-Action) on Baseten

Fine-tunes [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), a ~3B
Vision-Language-Action foundation model for humanoid robots, on a LeRobot-format
dataset. Training is pure PyTorch on pre-recorded demonstrations, so no simulator
or rendering stack is required.

**Resources:** 1 node, 1x L40S GPU (48GB).

## Prerequisites

1. [Create a Baseten account](https://baseten.co/signup) if you don't already have one.
2. Install the Truss CLI:
   ```bash
   # pip
   pip install -U truss
   # or uv
   uv add truss
   ```
3. (Optional) For gated Hugging Face datasets or models, add an `hf_access_token`
   secret in [Baseten Secrets](https://app.baseten.co/settings/secrets). The
   public GR00T base model needs no token.

## Getting Started

```bash
truss train init --examples isaac-groot-n1-finetune
cd isaac-groot-n1-finetune
truss train push config.py
```

## Notes

- **Python 3.10.** Isaac-GR00T pins Python 3.10. The base image ships 3.11, so
  `run.sh` uses `uv` to build a 3.10 environment before installing GR00T.
- **Dataset.** The example trains on the robotics demo dataset bundled in the
  GR00T repo (`demo_data/robot_sim.PickNPlace`). Point `--dataset-path` in
  `run.sh` at your own LeRobot v2 dataset (must include `meta/modality.json`).
- **GPU / VRAM.** A full fine-tune needs a 40GB+ GPU (L40S or H100). On a 24GB GPU
  (A10G/L4), switch `run.sh` to LoRA (`--lora_rank 64`) or
  `--no-tune_diffusion_model`.
- **Pin a version.** `run.sh` clones `main`. Pin to a release tag and match the
  command: N1.5 uses `scripts/gr00t_finetune.py`, N1.7 uses
  `gr00t/experiment/launch_finetune.py`.
- Isaac-GR00T is currently NVIDIA **Early Access** (no commercial deployment until GA).
