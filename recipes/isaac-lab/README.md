# Isaac Lab RL on Baseten

[NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) is a GPU-accelerated robot-learning
framework built on Isaac Sim. This recipe runs headless reinforcement-learning training as a
Baseten training job — installing Isaac Sim + Isaac Lab into a standard PyTorch image and
training a policy with [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl).

The default configuration trains the `Isaac-Cartpole-v0` task on a single A10G. The same
recipe runs any Isaac Lab task (e.g. `Isaac-Velocity-Rough-Anymal-C-v0` for legged locomotion)
— set `TASK` and `MAX_ITERATIONS` in `run.sh`.

## Hardware requirement: use an RT-core GPU

> **Isaac Lab does not run on H100, A100, or B200.** These are compute-only SKUs without RT
> cores, and their datacenter driver cannot create a Vulkan device. Isaac Sim's RTX renderer
> initializes Vulkan even in `--headless` mode and requires hardware ray tracing, so it fails
> at GPU-foundation init on those GPUs (`vkCreateInstance failed: ERROR_INCOMPATIBLE_DRIVER`).

Use a GPU with RT cores: **A10G** (default), **L4**, or **T4** (NVIDIA's documented minimum).
For most Isaac Lab RL workloads the simulator is the bottleneck rather than raw FLOPs, so these
are well-matched. (If you need H100-class compute, you would have to decouple simulation from
learning across machines — Isaac Lab couples them on one GPU by design.)

## How it works

`config.py` defines the training job. Two environment variables are essential:

- **`NVIDIA_DRIVER_CAPABILITIES=all`** — the container default (`compute,utility`) injects only
  CUDA libraries. Setting `all` makes the NVIDIA container runtime also inject the graphics /
  Vulkan stack (`libGLX_nvidia`, `nvidia_icd.json`, `libnvidia-rtcore`) that the renderer needs.
- **`OMNI_KIT_ACCEPT_EULA=YES`** — accepts the Omniverse EULA non-interactively; otherwise
  `import isaacsim` blocks on a prompt and the job hangs.

`run.sh` then installs the userspace GL/Vulkan libraries, `isaacsim[all,extscache]==5.1.0`,
clones Isaac Lab, installs `rsl_rl`, and launches headless training.

## Prerequisites

1. A [Baseten account](https://baseten.co/signup) with access to an RT-core GPU (A10G/L4/T4).
2. The Truss CLI installed and configured (`pip install -U truss && truss login`).

If you need GPU access or a higher quota, [reach out to us](mailto:support@baseten.co).

## Getting started

```bash
git clone https://github.com/basetenlabs/ml-cookbook.git
cd ml-cookbook/recipes/isaac-lab
truss train push config.py
```

Monitor the job:

```bash
truss train logs --job-id <JOB_ID> --tail
```

## What it trains

The default run trains a Cartpole balancing policy for 150 `rsl_rl` iterations. Outputs
(checkpoints + TensorBoard logs) are written under `/tmp/checkpoints` and synced as Baseten
training checkpoints. To train a different task or longer, edit the variables at the top of
`run.sh`:

```bash
TASK="Isaac-Velocity-Rough-Anymal-C-v0"   # any Isaac Lab task id
MAX_ITERATIONS=1500
```

## Notes

- **First run is slow.** Isaac Sim downloads Kit extensions and compiles RTX shaders on first
  launch (several minutes before training starts). The shader cache lives under `~/.cache`,
  which is on the persistent cache mount, so subsequent runs are much faster.
- The base image (`pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`) is chosen because its
  Ubuntu 22.04 / Python 3.11 / torch 2.7.0+cu128 stack matches Isaac Sim 5.1 exactly — no
  Python or torch reinstall required.
