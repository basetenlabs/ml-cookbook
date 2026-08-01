"""Supervised fine-tuning of GLM-5.2 on Baseten Loops.

GLM-5.2 (`zai-org/GLM-5.2-FP8`) is supported by Loops trainers, but
tinker-cookbook has no built-in renderer or tokenizer entry for it. This
recipe shows the bring-your-own-model pattern: define a renderer that matches
the model's chat template exactly, register it (plus a tokenizer factory)
in tinker-cookbook's custom registries, and train through the standard
supervised loop. The same pattern works for any Loops-supported model the
cookbook doesn't know about — check your workspace's live list at
https://api.baseten.co/v1/loops/capabilities.

Two GLM-specific wrinkles this file handles:

1. The FP8 weights repo ships no tokenizer files, so the tokenizer is loaded
   from `zai-org/GLM-5.2` and registered under the FP8 model name.
2. GLM renders assistant turns as `<|assistant|><think>reasoning</think>text`.
   This recipe trains in the template's native *no-thinking* mode
   (`enable_thinking=False`): the empty `<think></think>` prefix is part of
   the untrained header, loss lands only on the response text plus the
   `<|user|>` terminator. Serve the result with
   `chat_template_kwargs: {"enable_thinking": false}` for exact train/serve
   parity. `verify_template_parity` below asserts token-level equality with
   `tokenizer.apply_chat_template` before any GPU time is spent, so template
   drift fails fast.

Trains on the HuggingFaceH4/no_robots chat dataset. Set BASETEN_API_KEY and
LOOPS_PROJECT_ID before running:

    uv run sft/train_sft_glm.py

Any Config field can be overridden on the command line, e.g.:

    uv run sft/train_sft_glm.py learning_rate=5e-5 num_epochs=2

A GLM-5.2 Loops session provisions a multi-node trainer — it bills until you
deactivate it (`truss loops view` / `truss loops deactivate <id>`), so don't
leave it idle after training finishes.
"""

import asyncio
import functools
import sys

import chz
import tinker

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    ensure_text,
    parse_response_for_stop_token,
    remove_thinking,
)
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import Tokenizer, register_tokenizer

MODEL_NAME = "zai-org/GLM-5.2-FP8"  # what Loops trains
TOKENIZER_NAME = "zai-org/GLM-5.2"  # the FP8 repo has no tokenizer files
RENDERER_NAME = "glm52_nothink"


class GLM52NoThinkRenderer(Renderer):
    """
    Renderer for GLM-5.2 matching its chat template's no-thinking mode.

    Produces identical tokens to HuggingFace's apply_chat_template with
    enable_thinking=False (asserted by verify_template_parity):

        [gMASK]<sop><|system|>You are helpful.<|user|>Hi<|assistant|><think></think>Hello!

    Format notes:
    - `[gMASK]<sop>` opens every sequence (BOS).
    - Role markers `<|system|>` / `<|user|>` / `<|assistant|>` introduce turns;
      content follows immediately with no newlines or end-of-turn token.
    - Every assistant turn carries a forced empty `<think></think>` prefix.
      It lives in the header so it never receives loss.
    - A turn ends where the next role marker begins, so the stop signal is the
      `<|user|>` token itself. It overlaps the next message's header — the
      stop_overlap field appends it only after the final message, teaching the
      model to terminate (`<|user|>` is a configured eos in GLM's
      generation_config, so servers stop there).
    - The template strips assistant content (`content.strip()`) and drops any
      inline `<think>...</think>` reasoning in no-thinking mode; this renderer
      does the same. User and system content is rendered verbatim.
    - GLM's tool-use format (`<|observation|>`, `<tool_call>` blocks) is not
      implemented — conversations with tool turns raise rather than silently
      training on wrong tokens.
    """

    supports_streaming = True

    @functools.cached_property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("[gMASK]<sop>", add_special_tokens=False)

    @property
    def has_extension_property(self) -> bool:
        # No-thinking mode renders historical and final turns identically, so
        # the conversation is a pure concatenation and prefixes extend cleanly.
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        if role == "tool" or message.get("tool_calls"):
            raise RendererError(
                "GLM52NoThinkRenderer does not implement GLM's tool-use format "
                "(<|observation|>/<tool_call> blocks). Extend render_message "
                "following the zai-org/GLM-5.2 chat template if your data has "
                "tool turns."
            )
        content = message["content"]
        if role == "assistant" and isinstance(content, list):
            # This renderer trains the no-thinking shape: reasoning is dropped,
            # every assistant turn keeps the empty <think></think> prefix.
            content = remove_thinking(content)
        text = ensure_text(content)
        if role == "assistant":
            # Mirror the template: inline reasoning is split off, and assistant
            # content is stripped. User/system content is rendered verbatim.
            if "</think>" in text:
                text = text.split("</think>")[-1]
            text = text.strip()

        if role == "system":
            header_str = "<|system|>"
        elif role == "assistant":
            header_str = "<|assistant|><think></think>"
        else:
            header_str = "<|user|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(text, add_special_tokens=False)
            )
        ]
        stop_overlap = (
            tinker.types.EncodedTextChunk(tokens=[self._stop_token])
            if role == "assistant"
            else None
        )
        return RenderedMessage(header=header, output=output, stop_overlap=stop_overlap)

    @functools.cached_property
    def _stop_token(self) -> int:
        tokens = self.tokenizer.encode("<|user|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|user|>, got {tokens}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._stop_token]

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        return parse_response_for_stop_token(response, self.tokenizer, self._stop_token)


def _glm_tokenizer_factory() -> Tokenizer:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


def _glm_renderer_factory(tokenizer: Tokenizer, image_processor=None) -> Renderer:
    return GLM52NoThinkRenderer(tokenizer)


# Registered at import time so both the dataset builder and the training loop
# resolve GLM through the custom registries.
register_tokenizer(MODEL_NAME, _glm_tokenizer_factory)
register_renderer(RENDERER_NAME, _glm_renderer_factory)


def verify_template_parity(tokenizer: Tokenizer) -> None:
    """Assert the renderer reproduces GLM's chat template token-for-token.

    Runs in milliseconds before any GPU is provisioned. If zai-org ships a
    template change that breaks parity, this fails the run instead of
    silently training on a different format than the one served.
    """
    conversations: list[list[Message]] = [
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Name a prime number."},
            {"role": "assistant", "content": "7"},
            {"role": "user", "content": "Another?"},
            {"role": "assistant", "content": "13"},
        ],
        [
            # Trailing whitespace: the template strips assistant content.
            {"role": "user", "content": "Write a haiku.\n"},
            {"role": "assistant", "content": "Old pond, quiet splash.\n\n"},
        ],
    ]
    renderer = GLM52NoThinkRenderer(tokenizer)
    for messages in conversations:
        model_input, weights = renderer.build_supervised_example(messages)
        ours = [t for chunk in model_input.chunks for t in chunk.tokens]
        reference = tokenizer.apply_chat_template(
            messages, tokenize=True, enable_thinking=False
        )
        if hasattr(reference, "get"):  # transformers 5.x returns BatchEncoding
            reference = reference["input_ids"]
            if reference and isinstance(reference[0], list):
                reference = reference[0]
        # The supervised example ends with the <|user|> terminator, which the
        # bare template render does not include.
        assert ours[:-1] == list(reference) and ours[-1] == renderer._stop_token, (
            f"renderer/template divergence:\n ours={ours}\n  ref={list(reference)}"
        )
        # Generation-prompt parity: everything up to the final assistant reply.
        gen = renderer.build_generation_prompt(messages[:-1])
        gen_tokens = [t for chunk in gen.chunks for t in chunk.tokens]
        gen_reference = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if hasattr(gen_reference, "get"):
            gen_reference = gen_reference["input_ids"]
            if gen_reference and isinstance(gen_reference[0], list):
                gen_reference = gen_reference[0]
        assert gen_tokens == list(gen_reference), (
            f"generation-prompt divergence:\n ours={gen_tokens}\n  ref={list(gen_reference)}"
        )
        # Loss must land only on assistant text and the final terminator.
        assert weights.sum() > 0 and len(weights) == len(ours)


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        max_length=16384,
        batch_size=128,
    )
    dataset = chat_datasets.NoRobotsBuilder(common_config=common_config)
    if 0:  # To swap in your own dataset:
        dataset = FromConversationFileBuilder(
            common_config=common_config, file_path="/path/to/your/dataset.jsonl"
        )
        # ^^^ Create a dataset from a JSONL file in the same format as
        # tinker_cookbook/example_data/conversations.jsonl
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": "/tmp/loops-cookbook/sft-glm",
            "model_name": MODEL_NAME,
            "renderer_name": RENDERER_NAME,
            "dataset_builder": dataset,
            # Rank 64 with lr 1e-4 is a proven LoRA recipe for GLM-5.2 SFT.
            "lora_rank": 64,
            "learning_rate": 1e-4,
            "lr_schedule": "linear",
            "num_epochs": 1,
            "eval_every": 8,
        }
    )


def main(config: train.Config):
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    verify_template_parity(get_tokenizer(MODEL_NAME))
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
