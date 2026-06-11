"""GRPO on GSM8K math problems.

Uses tinker-cookbook's RL training loop, running against a Baseten Loops
trainer/sampler pair via the baseten-loops-tinker shim. Set BASETEN_API_KEY
and LOOPS_PROJECT_ID before running:

    uv run rl/train_grpo.py

Any Config field can be overridden on the command line, e.g.:

    uv run rl/train_grpo.py learning_rate=1e-5 max_tokens=512
"""

import asyncio
import sys

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
from tinker_cookbook.rl import train


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "Qwen/Qwen3.5-4B"
    # Non-thinking renderer: with the thinking renderer, completions need a much
    # larger max_tokens budget or every answer truncates mid-reasoning.
    renderer_name = "qwen3_5_disable_thinking"
    builder = Gsm8kDatasetBuilder(
        batch_size=128,
        group_size=16,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "renderer_name": renderer_name,
            "log_path": "/tmp/loops-cookbook/grpo",
            "dataset_builder": builder,
            "learning_rate": 4e-5,
            "max_tokens": 256,
            "eval_every": 0,
        }
    )


def main(config: train.Config):
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
