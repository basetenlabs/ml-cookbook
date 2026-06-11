"""Multi-turn RL: train a model to play twenty questions against another LLM.

Each episode is a full game — the player (the policy being trained) asks
yes/no questions and the answerer (a frozen model served by a separate Loops
sampler) responds, until the player guesses the secret word or runs out of
turns. The environment lives in env.py and is the part you'd adapt to build
your own multi-turn task.

Set BASETEN_API_KEY and LOOPS_PROJECT_ID before running:

    uv run multiturn_rl/train_twenty_questions.py

Any CLIConfig field can be overridden on the command line, e.g.:

    uv run multiturn_rl/train_twenty_questions.py batch_size=32 group_size=4
"""

import asyncio
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

from env import TwentyQuestionsDatasetBuilder


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
        log_path = f"/tmp/loops-cookbook/twenty-questions/{run_name}"
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
    )


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    asyncio.run(train.main(config))
