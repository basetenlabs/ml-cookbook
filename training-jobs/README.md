# Training Jobs

Training jobs run a training script you package with a `config.py` on Baseten GPUs, through [`truss train`](https://docs.baseten.co/training/overview). For online RL and step-by-step training from Python, see [Loops](../loops/README.md) instead.

- **[`examples/`](examples/)** are runnable, model- and framework-specific projects. Launch one directly with `truss train push config.py`.
- **[`recipes/`](recipes/)** are implementation guides and patterns that help you choose an approach and adapt it to your own project.

## Run an example

Install truss, pull the example into a fresh directory, and push it:

```bash
pip install -U truss
truss train init --examples <example-name> && cd <example-name>
truss train push config.py
```

Or push straight from a clone of this repo:

```bash
cd training-jobs/examples/qwen3-8b-lora-dpo-trl/training
truss train push config.py
```

Before you push, add any secrets the example needs (such as `hf_access_token`) under [Baseten secrets](https://app.baseten.co/settings/secrets), and check `config.py` and `run.sh` for model and dataset IDs or output repos to change. The CLI prints the job ID and the commands to follow it:

```
✨ Training job successfully created!
🪵 View logs for your job via `truss train logs --job-id e3m512w [--tail]`
🔍 View metrics for your job via `truss train metrics --job-id e3m512w`
```

See [managing and monitoring jobs](https://docs.baseten.co/training/management) for more. If checkpointing is enabled in `config.py`, list the checkpoints with:

```bash
truss train get_checkpoint_urls --job-id $JOB_ID
```

## Recipes

| Recipe | What it covers |
| --- | --- |
| [Programmatic Training API](recipes/programmatic-training-api/README.md) | Launch and manage training jobs from Python code rather than the CLI. |
| [Long-context SFT](recipes/sft/long_context/README.md) | Multi-node supervised fine-tuning of Qwen3-30B-A3B on long sequences. |
| [SFT framework options](recipes/sft/framework_options/README.md) | The same task at three layers of abstraction: raw PyTorch, the Hugging Face Trainer, and Axolotl. |
| [OCaml specialist (RL)](recipes/rl/ocaml_specialist/README.md) | RL with a compiler-in-the-loop reward, as full fine-tune and LoRA with VeRL. |
| [Improve tool calling (RL)](recipes/rl/improve_tool_calling/qwen3-8b-instruct-lora-trl) | LoRA GRPO with TRL to improve Qwen3-8B tool calling. |
| [Autoresearch fine-tuning](recipes/autoresearch-finetune/README.md) | An AI agent iterates on fine-tuning experiments within the bounds you set. |
| [Isaac Lab RL](recipes/isaac-lab/README.md) | Headless robot-learning RL with NVIDIA Isaac Lab. Needs an RT-core GPU (A10G, L4, or T4); does not run on H100, A100, or B200. |
| [Isaac-GR00T N1 fine-tuning](recipes/isaac-groot/README.md) | Fine-tune NVIDIA's ~3B vision-language-action robotics model on a LeRobot-format dataset. |

## Examples

Every example below can be pulled with `truss train init --examples <name>`. The mapping from name to directory lives in [`.examples-index.json`](../.examples-index.json).

| Name | Directory |
| --- | --- |
| `autoresearch-finetune` | [`recipes/autoresearch-finetune/training`](recipes/autoresearch-finetune/training) |
| `gemma-27b-axolotl` | [`examples/gemma-27b-axolotl/training`](examples/gemma-27b-axolotl/training) |
| `glm-4-6-msswift` | [`examples/glm-4-6-msswift/training`](examples/glm-4-6-msswift/training) |
| `glm-4.7-flash-msswift` | [`examples/glm-4.7-flash-msswift/training`](examples/glm-4.7-flash-msswift/training) |
| `glm-4.7-msswift` | [`examples/glm-4.7-msswift/training`](examples/glm-4.7-msswift/training) |
| `isaac-groot` | [`recipes/isaac-groot`](recipes/isaac-groot) |
| `isaac-lab` | [`recipes/isaac-lab`](recipes/isaac-lab) |
| `kimi-k2.6-lora-merge` | [`examples/kimi-k2.6-lora-merge/training`](examples/kimi-k2.6-lora-merge/training) |
| `llama-8b-lora-unsloth` | [`examples/llama-8b-lora-unsloth/training`](examples/llama-8b-lora-unsloth/training) |
| `minimax2_5-multinode` | [`examples/minimax2_5-multinode`](examples/minimax2_5-multinode) |
| `mnist-pytorch` | [`examples/mnist-pytorch/training`](examples/mnist-pytorch/training) |
| `nemotron3-30b-torchtitan` | [`examples/nemotron3-30b-torchtitan/training`](examples/nemotron3-30b-torchtitan/training) |
| `omnivoice-tts-finetuning` | [`examples/omnivoice-tts-finetuning`](examples/omnivoice-tts-finetuning) |
| `orpheus-transformers` | [`examples/orpheus-transformers/training`](examples/orpheus-transformers/training) |
| `oss-gpt-120b-axolotl` | [`examples/oss-gpt-120b-axolotl/training`](examples/oss-gpt-120b-axolotl/training) |
| `oss-gpt-20b-axolotl` | [`examples/oss-gpt-20b-axolotl/training`](examples/oss-gpt-20b-axolotl/training) |
| `oss-gpt-20b-fft-trl` | [`examples/oss-gpt-20b-fft-trl/training`](examples/oss-gpt-20b-fft-trl/training) |
| `oss-gpt-20b-lora-trl` | [`examples/oss-gpt-20b-lora-trl/training`](examples/oss-gpt-20b-lora-trl/training) |
| `qwen3-0.6b-axolotl` | [`examples/qwen3-0.6b-axolotl/training`](examples/qwen3-0.6b-axolotl/training) |
| `qwen3-0.6b-classifier-hf-trainer` | [`examples/qwen3-0.6b-classifier-hf-trainer/training`](examples/qwen3-0.6b-classifier-hf-trainer/training) |
| `qwen3-0.6b-classifier-pytorch` | [`examples/qwen3-0.6b-classifier-pytorch/training`](examples/qwen3-0.6b-classifier-pytorch/training) |
| `qwen3-0.6b-pytorch` | [`examples/qwen3-0.6b-pytorch/training`](examples/qwen3-0.6b-pytorch/training) |
| `qwen3-235b-mswift` | [`examples/qwen3-235b-mswift/training`](examples/qwen3-235b-mswift/training) |
| `qwen3-30b-mswift-multinode` | [`examples/qwen3-30b-mswift-multinode/training`](examples/qwen3-30b-mswift-multinode/training) |
| `qwen3-30b-mswift-singlenode` | [`examples/qwen3-30b-mswift-singlenode/training`](examples/qwen3-30b-mswift-singlenode/training) |
| `qwen3-32b-fft-verl-multinode` | [`examples/qwen3-32b-fft-verl-multinode/training`](examples/qwen3-32b-fft-verl-multinode/training) |
| `qwen3-4b-lora-sft-trl` | [`examples/qwen3-4b-lora-sft-trl/training`](examples/qwen3-4b-lora-sft-trl/training) |
| `qwen3-80b-msswift` | [`examples/qwen3-80b-msswift/training`](examples/qwen3-80b-msswift/training) |
| `qwen3-8b-fft-verl` | [`examples/qwen3-8b-fft-verl/training`](examples/qwen3-8b-fft-verl/training) |
| `qwen3-8b-instruct-lora-trl-tool-calling` | [`recipes/rl/improve_tool_calling/qwen3-8b-instruct-lora-trl`](recipes/rl/improve_tool_calling/qwen3-8b-instruct-lora-trl) |
| `qwen3-8b-lora-dpo-trl` | [`examples/qwen3-8b-lora-dpo-trl/training`](examples/qwen3-8b-lora-dpo-trl/training) |
| `qwen3-8b-lora-verl` | [`examples/qwen3-8b-lora-verl/training`](examples/qwen3-8b-lora-verl/training) |
| `qwen3-asr-transformers` | [`examples/qwen3-asr-transformers/training`](examples/qwen3-asr-transformers/training) |
| `qwen3-tts-transformers` | [`examples/qwen3-tts-transformers`](examples/qwen3-tts-transformers) |
| `qwen3.6-27b-msswift` | [`examples/qwen3.6-27b-msswift/training`](examples/qwen3.6-27b-msswift/training) |
| `qwen3.6-35b-msswift` | [`examples/qwen3.6-35b-msswift/training`](examples/qwen3.6-35b-msswift/training) |
| `whisper-transformers` | [`examples/whisper-transformers/training`](examples/whisper-transformers/training) |

The diarization examples ([`parakeet-nemotron-diarization`](examples/parakeet-nemotron-diarization/README.md), [`qwen3-asr-nemotron-diarization`](examples/qwen3-asr-nemotron-diarization/README.md)) are inference clients for Model Library deployments rather than training projects, so they aren't in the index.

