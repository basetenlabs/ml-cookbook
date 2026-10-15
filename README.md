  <a href="https://www.baseten.co/">
    <img alt="Baseten" src="https://github.com/user-attachments/assets/1e342a9e-56a5-4919-b776-228a5fc1288e">
  </a>

<p align="center">
  <strong><a href="https://docs.baseten.co/loops/overview">Loops docs</a> | <a href="https://docs.baseten.co/training/overview">Training Jobs docs</a> | <a href="https://docs.baseten.co/examples/deploy-your-first-model">Inference docs</a></strong>
</p>

Ready-to-use recipes for training models on Baseten. The cookbook covers Baseten's two ways to train:

| | [**Loops**](loops/) | [**Training Jobs**](training-jobs/) |
| --- | --- | --- |
| What it is | An SDK for online training: you write the training loop in plain Python and Baseten runs the trainer and a paired sampler | Batch jobs: you package a training script and `config.py`, then `truss train push` runs it on GPUs to completion |
| Best for | RL (GRPO, async and multi-turn), agent training, and SFT you want to drive step by step | SFT, DPO, full fine-tunes, and multi-node runs on the framework you choose (Axolotl, TRL, MS-Swift, VeRL, Megatron, TorchTitan, ...) |
| You manage | The loop: data, rewards, when to step and sample | The script and the GPU config |
| Start here | [`loops/README.md`](loops/README.md) | [`training-jobs/README.md`](training-jobs/README.md) |

## Repository layout

```
loops/                      # Loops recipes: SFT, GRPO (sync + async), multi-turn RL, rollout dashboard
training-jobs/
  examples/                 # runnable model/framework projects, launch with `truss train push`
  recipes/                  # implementation guides and patterns (RL, SFT, robotics, programmatic API, ...)
tools/
  experiment-dashboard/     # self-hosted run dashboard for both Loops and Training Jobs
.examples-index.json        # where each example lives; read by `truss train init --examples`
```

## Loops

Loops is Baseten's SDK for online training. You write the training loop in plain Python: forward-backward passes, optimizer steps, and weight sync to a live sampler. Baseten provisions and runs the trainer and sampler. The recipes cover SFT, GRPO (sync, and async with bounded off-policy sampling), and multi-turn RL.

```bash
cd loops
export BASETEN_API_KEY="your-api-key"
uv sync
uv run rl/train_grpo.py
```

See [`loops/README.md`](loops/README.md) for every recipe and the [Loops docs](https://docs.baseten.co/loops/overview) for concepts.

## Training Jobs

Training jobs run a training script you package with a `config.py` on Baseten GPUs, through `truss train`. Pull any example into a fresh directory and launch it:

```bash
pip install -U truss
truss train init --examples qwen3-8b-lora-dpo-trl && cd qwen3-8b-lora-dpo-trl
truss train push config.py
```

Or clone this repo and push from the example's directory:

```bash
git clone https://github.com/basetenlabs/ml-cookbook.git
cd ml-cookbook/training-jobs/examples/qwen3-8b-lora-dpo-trl/training
truss train push config.py
```

See [`training-jobs/README.md`](training-jobs/README.md) for the full list of examples and recipes, and the [Training Jobs docs](https://docs.baseten.co/training/overview) for concepts.

## Tools

[`tools/experiment-dashboard`](tools/experiment-dashboard/README.md) is a local, self-hosted dashboard over an S3 experiment store. It syncs runs down and serves loss-curve overlays, hyperparameter filtering, and cross-run comparison, using only the Python standard library. It works with any training source that writes its three-file run contract, including Loops and Training Jobs.

## Prerequisites

- A Baseten account. [Sign up here](https://baseten.co/signup) if you don't have one.
- API keys and tokens your runs need (for example, a [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens) for gated models), saved as [Baseten secrets](https://app.baseten.co/settings/secrets).
- Python 3.12+ for Training Jobs tooling, and Python 3.13+ with [uv](https://docs.astral.sh/uv/) for Loops.

## Contributing

Contributions are welcome. Please open issues or submit pull requests. When you add, rename, or move a Training Jobs example, update [`.examples-index.json`](.examples-index.json) so `truss train init --examples` can find it; `bin/check_examples_index.py` checks it in CI.

## License

[MIT License](LICENSE), except [`loops/`](loops/), which is adapted from tinker-cookbook and licensed under Apache 2.0 (see [`loops/LICENSE`](loops/LICENSE)).
