# Adapted from tinker-cookbook (https://github.com/thinking-machines-lab/tinker-cookbook),
# Copyright 2025 Thinking Machines Lab, licensed under Apache-2.0. Modified by Baseten.

"""Async math RL on DeepMath.

As in train_grpo_async.py, rollouts and optimizer steps run concurrently
instead of alternating, with `max_steps_off_policy` bounding how stale a
trajectory group may be. DeepMath solutions are long and vary widely in
length, so the trainer never idles waiting for the slowest one.

Set BASETEN_API_KEY before running:

    uv run rl/train_math_async.py

Any CLIConfig field can be overridden on the command line, e.g.:

    uv run rl/train_math_async.py max_steps_off_policy=4 groups_per_batch=32
"""

import asyncio
import os

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl.math_env import DeepMathDatasetBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, main

# Provision the paired sampler alongside the trainer instead of at first use.
os.environ.setdefault("LOOPS_WARM_START_SAMPLER", "true")


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32
    renderer_name: str | None = None

    group_size: int = 16
    groups_per_batch: int = 64
    learning_rate: float = 4e-5
    # DeepMath solutions are long; don't cap them like the GSM8K recipes do.
    max_tokens: int = 2048

    # Max optimizer steps a trajectory group may lag the current policy.
    max_steps_off_policy: int = 2

    log_path: str = "/tmp/loops-cookbook/math_async"
    wandb_project: str | None = None
    eval_every: int = 0
    save_every: int = 20
    max_steps: int | None = None
    seed: int = 0

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_config(cli_config: CLIConfig) -> Config:
    renderer_name = (
        cli_config.renderer_name
        or model_info.get_recommended_renderer_name(cli_config.model_name)
    )
    builder = DeepMathDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        seed=cli_config.seed,
    )

    return Config(
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        renderer_name=renderer_name,
        log_path=cli_config.log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_steps=cli_config.max_steps,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        ),
    )


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    asyncio.run(main(config))
