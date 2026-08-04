# Adapted from tinker-cookbook (https://github.com/thinking-machines-lab/tinker-cookbook),
# Copyright 2025 Thinking Machines Lab, licensed under Apache-2.0. Modified by Baseten.

"""Async multi-turn RL: twenty questions with concurrent rollouts and training.

Same environment as train_twenty_questions.py — the player (the policy being
trained) asks yes/no questions and a frozen answerer model served by a
separate Loops sampler responds — but instead of alternating between playing
a full batch of games and training on it, rollout workers play games
continuously while the training loop takes optimizer steps.

Multi-turn episodes make async mode pay off more than single-turn tasks: a
game takes many sequential sampler calls (one per question), so a synchronous
loop leaves the trainer idle for the duration of the slowest game in every
batch. It also makes off-policy staleness more likely, since a game spans
more wall-clock time than a single completion. `max_steps_off_policy` bounds
that staleness: every trajectory group is tagged with the policy version it
was sampled from, and groups more than `max_steps_off_policy` optimizer steps
behind the current trainer step are requeued instead of trained on.

Set BASETEN_API_KEY and LOOPS_PROJECT_ID before running:

    uv run multiturn_rl/train_twenty_questions_async.py

Any CLIConfig field can be overridden on the command line, e.g.:

    uv run multiturn_rl/train_twenty_questions_async.py max_steps_off_policy=4 batch_size=32
"""

import asyncio
import os
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

from env import TwentyQuestionsDatasetBuilder

# Provision the paired sampler in parallel with the trainer instead of at
# the first sampling request.
os.environ.setdefault("LOOPS_WARM_START_SAMPLER", "true")


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    # Non-thinking renderer: questions are capped at max_tokens=20, which a
    # thinking renderer would spend entirely on truncated reasoning.
    renderer_name: str | None = "qwen3_5_disable_thinking"
    group_size: int = 8
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-5
    max_tokens: int = 20
    eval_every: int = 5
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None
    answerer_base_model: str = "Qwen/Qwen3.5-0.8B"

    # Trajectory groups sampled more than this many optimizer steps behind
    # the current policy are requeued rather than trained on.
    max_steps_off_policy: int = 2

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    renderer_name = (
        cli_config.renderer_name
        or model_info.get_recommended_renderer_name(cli_config.model_name)
    )

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{model_name}-{cli_config.group_size}group-{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/loops-cookbook/twenty-questions-async/{run_name}"
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = TwentyQuestionsDatasetBuilder(
        batch_size=cli_config.batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        num_epochs=cli_config.num_epochs,
        answerer_base_model=cli_config.answerer_base_model,
    )

    return train.Config(
        model_name=model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        max_steps=cli_config.max_steps,
        async_config=train.AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.batch_size,
        ),
    )


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    asyncio.run(train.main(config))
