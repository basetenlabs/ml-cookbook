# Fine-tune NVIDIA Isaac-GR00T N1 on Baseten

[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) is a ~3B
Vision-Language-Action foundation model for humanoid robots. This recipe
fine-tunes it on a LeRobot-format dataset as a Baseten training job.

Unlike Isaac Lab (which drives Isaac Sim's RTX renderer), GR00T fine-tuning is
pure PyTorch on pre-recorded demonstrations — no simulator, Omniverse, or
rendering stack — so it runs on any GPU with no RT-core requirement.

## Hardware

The default is a single **H100 (80GB)**, which runs a full fine-tune of GR00T's
projector and diffusion action head and has ample host RAM for the full 3B-model
save. To fine-tune on a smaller GPU such as an **A10G (24GB)**, add `--lora-rank 64`
in `run.sh` (training fits, but A10G's 16GB host RAM can OOM on a full-model save,
so LoRA's small-adapter save is the safer path there).

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

- **Pinned to n1.5.** `run.sh` clones the `n1.5-release` tag, whose
  `scripts/gr00t_finetune.py` entrypoint and `--no-tune-diffusion-model` flag are
  stable. (`main` has moved to N1.7, which uses
  `gr00t/experiment/launch_finetune.py` instead.)
