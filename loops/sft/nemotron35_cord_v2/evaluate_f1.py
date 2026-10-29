import argparse
import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import wandb
from baseten.loops import (
    EncodedTextChunk,
    ImageChunk,
    ModelInput,
    SamplingParams,
    ServiceClient,
)
from huggingface_hub import HfApi, hf_hub_download
from pyarrow import parquet
from transformers import AutoTokenizer

from common import ROOT, api_request, dataset_cache, load_env
from data import load_prepared, sha256
from prepare_assets import verify_assets
from receipt_f1 import score
from vision import _prepare_image, load_processor_config

LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def validate_comparison(reference, evaluation, audit):
    manifest = json.loads((reference / "evaluation_manifest.json").read_text())
    reference_audit = json.loads((reference / "test_audit.json").read_text())
    result = json.loads((reference / "completion_audit.json").read_text())
    rows = [
        json.loads(line)
        for line in (reference / "rollouts.jsonl").read_text().splitlines()
    ]
    if (
        not result.get("complete")
        or result.get("checkpoint_step") != 400
        or len(rows) != 100
        or {row["index"] for row in rows} != set(range(100))
        or result["rollouts_sha256"] != sha256(reference / "rollouts.jsonl")
    ):
        raise ValueError("Comparison requires an audited, complete step400 evaluation.")
    if any(result[key] != value for key, value in score(rows).items()):
        raise ValueError("Saved reference scores disagree with its rollouts.")
    for key in (
        "base_model",
        "dataset",
        "split",
        "rows",
        "prompt",
        "temperature",
        "top_p",
        "seed",
        "thinking",
        "medium_effort",
        "reasoning_effort_override",
        "inference_context",
        "output_budget",
        "image_pixels",
        "managed_decoding",
        "input_truncation",
    ):
        if manifest[key] != evaluation[key]:
            raise ValueError(f"Comparison protocol mismatch: {key}.")
    for key in (
        "dataset",
        "revision",
        "split",
        "rows",
        "snapshot_sha256",
        "tokenizer_sha256",
        "processor_sha256",
    ):
        if reference_audit[key] != audit[key]:
            raise ValueError(f"Comparison test-data mismatch: {key}.")
    if (
        sha256(reference / "test_snapshot.json") != audit["snapshot_sha256"]
        or result["snapshot_sha256"] != audit["snapshot_sha256"]
    ):
        raise ValueError(
            "Reference snapshot no longer matches the frozen test receipts."
        )
    return result


def sampler_source(base_model, checkpoint):
    return (
        {"base_model": base_model} if checkpoint is None else {"model_path": checkpoint}
    )


class SamplerRecorder(logging.Handler):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.sampler_id = None

    def emit(self, record):
        match = re.search(
            r"Provisioned Loops sampler (\S+) on session (\S+)\.", record.getMessage()
        )
        if match:
            self.sampler_id = match[1]
            write_json(
                self.output / "sampler.json",
                {
                    "sampler_id": match[1],
                    "session_id": match[2],
                    "owned_by_this_evaluation": True,
                },
            )


def image_prompt(tokenizer, prepared, image_format, expected_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        medium_effort=False,
    )
    if rendered.count("<image>") != 1:
        raise ValueError("Expected exactly one receipt image placeholder.")
    prefix, suffix = rendered.split("<image>")
    prefix_ids = tokenizer.encode(prefix + "<img>", add_special_tokens=False)
    suffix_ids = tokenizer.encode("</img>" + suffix, add_special_tokens=False)
    image_ids = tokenizer.encode("<image>", add_special_tokens=False)
    if len(image_ids) != 1:
        raise ValueError("Image placeholder must be a single token.")
    expected_ids = prefix_ids + image_ids * expected_tokens + suffix_ids
    prompt = ModelInput(
        chunks=[
            EncodedTextChunk(tokens=prefix_ids),
            ImageChunk(
                data=prepared, format=image_format, expected_tokens=expected_tokens
            ),
            EncodedTextChunk(tokens=suffix_ids),
        ]
    )
    if prompt.length != len(expected_ids):
        raise ValueError("Expanded multimodal prompt length mismatch.")
    return prompt, expected_ids


def prepare_test(output, config, tokenizer, context, maximum_output):
    import torch

    torch.set_num_threads(2)
    known, train_audit = load_prepared(config)
    used_hashes = {
        row["original_image_sha256"] for split in known.values() for row in split
    }
    info = HfApi().dataset_info(config["dataset"], revision=config["dataset_revision"])
    if info.sha != config["dataset_revision"]:
        raise ValueError("Test dataset revision changed.")
    filenames = sorted(
        entry.rfilename
        for entry in info.siblings
        if entry.rfilename.startswith("data/test-")
        and entry.rfilename.endswith(".parquet")
    )
    if not filenames:
        raise ValueError("Pinned dataset has no test parquet files.")
    sources = []
    examples = []
    images = output / "images"
    images.mkdir()
    processor = load_processor_config()
    for filename in filenames:
        print("Downloading frozen test split:", filename, flush=True)
        path = hf_hub_download(
            config["dataset"],
            filename,
            repo_type="dataset",
            revision=info.sha,
            cache_dir=dataset_cache(),
        )
        sources.append({"file": filename, "sha256": sha256(path)})
        for batch in parquet.ParquetFile(path).iter_batches(
            batch_size=1, columns=["image", "ground_truth"]
        ):
            for row in batch.to_pylist():
                raw = row["image"]["bytes"]
                prepared, image_format, geometry, audit = _prepare_image(
                    raw,
                    "png" if raw.startswith(b"\x89PNG") else "jpeg",
                    processor,
                    allow_quantized_resize=True,
                )
                if audit["original_image_sha256"] in used_hashes:
                    raise ValueError("Test receipt image overlaps train/validation.")
                ground_truth = json.loads(row["ground_truth"])
                candidates = ground_truth.get(
                    "gt_parses", [ground_truth.get("gt_parse")]
                )
                if len(candidates) != 1 or not isinstance(candidates[0], dict):
                    raise ValueError("Expected one authoritative receipt parse.")
                prompt, expected_ids = image_prompt(
                    tokenizer, prepared, image_format, geometry.expected_tokens
                )
                if prompt.length + maximum_output > context:
                    raise ValueError(
                        "Complete prompt and generation budget exceed inference context; truncation forbidden."
                    )
                index = len(examples)
                image = images / f"test-{index:04d}.{image_format}"
                image.write_bytes(prepared)
                examples.append(
                    {
                        "index": index,
                        "prompt": prompt,
                        "expected_prompt_ids": expected_ids,
                        "ground_truth": candidates[0],
                        "prompt_tokens": prompt.length,
                        "image": str(image.relative_to(output)),
                        **audit,
                    }
                )
    if len(examples) != 100:
        raise ValueError("Expected all 100 official test receipts.")
    snapshot = [
        {
            key: value
            for key, value in example.items()
            if key not in {"prompt", "expected_prompt_ids"}
        }
        for example in examples
    ]
    write_json(output / "test_snapshot.json", snapshot)
    audit = {
        "dataset": config["dataset"],
        "revision": info.sha,
        "split": "test",
        "rows": 100,
        "source_files": sources,
        "snapshot_sha256": sha256(output / "test_snapshot.json"),
        "overlap_with_train_or_validation": 0,
        "input_truncations": 0,
        "maximum_prompt_tokens": max(example["prompt_tokens"] for example in examples),
        "training_manifest_sha256": sha256(
            ROOT / "data" / "prepared" / "manifest.json"
        ),
        "tokenizer_sha256": train_audit["tokenizer_sha256"],
        "processor_sha256": train_audit["processor_sha256"],
        "image_pixels": config["managed_settings"]["image_pixels"],
    }
    write_json(output / "test_audit.json", audit)
    return examples, audit


async def generate(sampler, examples, tokenizer, output, timeout, maximum_output):
    semaphore = asyncio.Semaphore(2)

    async def one(example):
        async with semaphore:
            response = await asyncio.wait_for(
                sampler.sample_async(
                    example["prompt"],
                    num_samples=1,
                    sampling_params=SamplingParams(
                        max_tokens=maximum_output,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        seed=42,
                        stop=["<|im_end|>", "<|endoftext|>"],
                    ),
                    affinity_key=f"cord-test-{example['index']}",
                    include_routed_experts=False,
                    include_prompt_logprobs=True,
                ),
                timeout=timeout,
            )
            if len(response.sequences) != 1:
                raise ValueError(
                    "Sampler must return exactly one completion per receipt."
                )
            if response.prompt_token_ids != example["expected_prompt_ids"]:
                raise ValueError(
                    "Sampler did not verify the complete, untruncated multimodal prompt."
                )
            sequence = response.sequences[0]
            if not sequence.tokens or sequence.stop_reason not in {"stop", "length"}:
                raise ValueError("Empty completion or unknown stop reason.")
            return {
                "index": example["index"],
                "ground_truth": example["ground_truth"],
                "text": tokenizer.decode(sequence.tokens, skip_special_tokens=False),
                "finish_reason": sequence.stop_reason,
                "output_tokens": len(sequence.tokens),
                "prompt_tokens": example["prompt_tokens"],
                "original_image_sha256": example["original_image_sha256"],
                "response": response.model_dump(mode="json"),
            }

    tasks = [asyncio.create_task(one(example)) for example in examples]
    rows = []
    try:
        with (output / "rollouts.jsonl").open("x") as handle:
            for task in asyncio.as_completed(tasks):
                row = await task
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                handle.flush()
                rows.append(row)
                if len(rows) == 1 or len(rows) % 10 == 0:
                    print(f"Complete receipt rollouts: {len(rows)}/100", flush=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if len(rows) != 100 or {row["index"] for row in rows} != set(range(100)):
        raise ValueError(
            "Refusing to report F1 without all 100 distinct test receipts."
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Score CORD-v2 on the base model or a frozen sampler checkpoint; never train."
    )
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--compare-output", type=Path)
    parser.add_argument("--env-file")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--name")
    arguments = parser.parse_args()
    if arguments.baseline and arguments.compare_output is None:
        parser.error(
            "--baseline requires --compare-output pointing to the audited step400 evaluation."
        )
    if arguments.compare_output is not None and not arguments.baseline:
        parser.error("--compare-output is used only with --baseline.")
    load_env(arguments.env_file)
    if (
        not os.environ.get("WANDB_API_KEY")
        and os.environ.get("WANDB_MODE") != "offline"
    ):
        raise ValueError("W&B credential is required.")
    if any(
        os.environ.get(name)
        for name in ("LOOPS_REUSE_FROM_RUN_ID", "LOOPS_REUSE_FROM_SESSION_ID")
    ):
        raise ValueError(
            "Unset reuse variables; this evaluator must not attach to unrelated samplers."
        )
    config = json.loads((arguments.training_output / "recipe.json").read_text())
    config["team"] = os.environ.get("BASETEN_TEAM") or config.get("team")
    config["wandb_entity"] = os.environ.get("WANDB_ENTITY") or config.get(
        "wandb_entity"
    )
    config["wandb_project"] = os.environ.get("WANDB_PROJECT") or config["wandb_project"]
    completion = json.loads((arguments.training_output / "completion.json").read_text())
    checkpoints = json.loads(
        (arguments.training_output / "checkpoints.json").read_text()
    )
    if not completion["complete"] or completion["updates"] != 400:
        raise ValueError(
            "This evaluator requires verified completion of the 400-update run."
        )
    checkpoint = (
        None
        if arguments.baseline
        else next(entry for entry in checkpoints if entry["step"] == 400)[
            "sampler_weights"
        ]
    )
    checkpoint_step = 0 if arguments.baseline else 400
    variant = "baseline" if arguments.baseline else "step400"
    context, maximum_output = 65536, 32768
    name = (
        arguments.name
        or f"baseten_cord_v2_{variant}_test_f1_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    )
    if not name.startswith("baseten_") or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        parser.error(
            "--name must start baseten_ and use only letters, digits, dots, underscores or hyphens."
        )
    output = ROOT / "outputs" / name
    verify_assets()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "assets" / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    examples, audit = prepare_test(output, config, tokenizer, context, maximum_output)
    evaluation = {
        "variant": variant,
        "checkpoint": checkpoint,
        "checkpoint_step": checkpoint_step,
        "training_wandb_url": completion["wandb_url"],
        "base_model": config["base_model"],
        "dataset": config["dataset"],
        "split": "test",
        "rows": 100,
        "prompt": "Describe this image.",
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "thinking": True,
        "medium_effort": False,
        "reasoning_effort_override": "Not exposed by raw-token Loops SamplingParams; template default retained.",
        "inference_context": context,
        "output_budget": maximum_output,
        "training_context": config["max_seq_len"],
        "image_pixels": config["managed_settings"]["image_pixels"],
        "managed_decoding": "MTP/speculative decoding and sampler precision are not exposed or asserted.",
        "input_truncation": "Forbidden; returned prompt tokens must exactly match every expanded image/text token.",
    }
    reference_result = (
        validate_comparison(arguments.compare_output, evaluation, audit)
        if arguments.baseline
        else None
    )
    write_json(output / "evaluation_manifest.json", evaluation)
    logging.basicConfig(level=logging.INFO)
    recorder = SamplerRecorder(output)
    logger = logging.getLogger("baseten.loops.service_client")
    logger.addHandler(recorder)
    sampler = run = None
    completed = False
    try:
        run = wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=name,
            group="baseten_cord_v2",
            job_type="receipt_f1",
            config=evaluation,
            dir=str(output),
        )
        print("Evaluation W&B:", run.url, flush=True)
        service = ServiceClient(team=config["team"])
        if checkpoint is not None:
            registered = service.list_checkpoints(
                checkpoint.split(":", 2)[2].split("/")[0]
            )
            if not any(
                entry.checkpoint_id == "step-400"
                and entry.target == "sampler"
                and entry.base_model == config["base_model"]
                for entry in registered
            ):
                raise ValueError(
                    "Requested sampler checkpoint is missing or has the wrong base model."
                )
        sampler = service.create_sampling_client(
            **sampler_source(config["base_model"], checkpoint),
            max_seq_len=context,
            num_replicas=1,
            timeout=arguments.timeout,
            ready_timeout=arguments.timeout,
        )
        if not recorder.sampler_id:
            raise ValueError("Cannot track ownership of the newly created sampler.")
        write_json(
            output / "sampler_metadata.json",
            api_request(f"/v1/loops/samplers/{recorder.sampler_id}"),
        )
        rows = asyncio.run(
            generate(
                sampler, examples, tokenizer, output, arguments.timeout, maximum_output
            )
        )
        result = {
            **score(rows),
            "complete": True,
            "variant": variant,
            "checkpoint_step": checkpoint_step,
            "checkpoint": checkpoint,
            "split": "test",
            "snapshot_sha256": audit["snapshot_sha256"],
            "wandb_url": run.url,
            "rollouts_sha256": sha256(output / "rollouts.jsonl"),
            "input_truncations": 0,
        }
        write_json(output / "completion_audit.json", result)
        metrics = {
            f"baseten_test_{key}": result[key]
            for key in (
                "f1",
                "precision",
                "recall",
                "rows",
                "invalid_outputs",
                "output_truncations",
                "input_truncations",
            )
        }
        comparison = None
        if reference_result is not None:
            comparison = {
                "complete": True,
                "baseline": result,
                "step400": reference_result,
                "f1_delta_percentage_points": 100
                * (reference_result["f1"] - result["f1"]),
                "snapshot_sha256": audit["snapshot_sha256"],
                "protocol_identical": True,
                "interpretation": "Same Describe this image prompt; strict structured-field F1 includes output-format compliance, not OCR ability alone.",
            }
            write_json(output / "comparison.json", comparison)
            metrics["baseten_test_f1_improvement_pp"] = comparison[
                "f1_delta_percentage_points"
            ]
        run.log(metrics)
        run.summary.update(metrics)
        artifact = wandb.Artifact(name + "_rollouts", type="receipt-evaluation")
        for filename in (
            "evaluation_manifest.json",
            "test_audit.json",
            "test_snapshot.json",
            "rollouts.jsonl",
            "completion_audit.json",
        ):
            artifact.add_file(str(output / filename))
        if comparison is not None:
            artifact.add_file(str(output / "comparison.json"))
        run.log_artifact(artifact)
        table = wandb.Table(
            columns=[
                "index",
                "ground_truth",
                "prediction",
                "output_tokens",
                "finish_reason",
            ]
        )
        for row in sorted(rows, key=lambda entry: entry["index"]):
            table.add_data(
                row["index"],
                json.dumps(row["ground_truth"], ensure_ascii=False),
                row["text"],
                row["output_tokens"],
                row["finish_reason"],
            )
        run.log({"baseten_test_receipts": table})
        try:
            parent = wandb.Api().run(
                "/".join(completion["wandb_url"].rstrip("/").split("/")[-4:]).replace(
                    "/runs/", "/"
                )
            )
            if arguments.baseline:
                parent.summary.update(
                    {
                        **{
                            key.replace(
                                "baseten_test_", "baseten_baseline_test_", 1
                            ): value
                            for key, value in metrics.items()
                            if key != "baseten_test_f1_improvement_pp"
                        },
                        "baseten_baseline_test_f1_evaluation_url": run.url,
                        "baseten_test_f1_improvement_pp": comparison[
                            "f1_delta_percentage_points"
                        ],
                    }
                )
            else:
                parent.summary.update(
                    {**metrics, "baseten_test_f1_evaluation_url": run.url}
                )
        except Exception as error:
            LOGGER.exception(
                "Training-run summary update failed: %s",
                type(error).__name__,
                exc_info=False,
            )
        completed = True
        print("Complete audited test F1:", json.dumps(result), flush=True)
        if comparison is not None:
            print(
                "F1 improvement over baseline (percentage points):",
                comparison["f1_delta_percentage_points"],
                flush=True,
            )
    finally:
        if sampler is not None:
            try:
                sampler.close()
            except Exception as error:
                LOGGER.exception(
                    "Sampler client close failed: %s",
                    type(error).__name__,
                    exc_info=False,
                )
        if recorder.sampler_id:
            try:
                api_request(
                    f"/v1/loops/samplers/{recorder.sampler_id}/deactivate",
                    method="POST",
                )
                write_json(
                    output / "deactivation.json",
                    {"sampler_id": recorder.sampler_id, "requested": True},
                )
                print(
                    "Deactivated only this evaluator's sampler:",
                    recorder.sampler_id,
                    flush=True,
                )
            except Exception as error:
                LOGGER.exception(
                    "Sampler cleanup needs attention: %s %s",
                    recorder.sampler_id,
                    type(error).__name__,
                    exc_info=False,
                )
        logger.removeHandler(recorder)
        if run is not None:
            run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
