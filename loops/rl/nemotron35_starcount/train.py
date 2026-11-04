"""LoRA reinforcement learning for visual star counting with Baseten Loops."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import numbers
import os
from pathlib import Path
import re
import time

from baseten.loops import AdamParams, Datum, EncodedTextChunk, ServiceClient, SamplingParams, TensorData
import wandb

from common import API_BASE_URL, ROOT, api_request, load_env, prepare, reward, settings, validate_catalog
from core import context_limit_reached, leave_one_out_advantages
from loss import PPOLossConfig, build_ppo_record, make_ppo_loss, valid_sequence_count
from lifecycle import cleanup_owned_run
from vision import validate_resolved_prompt


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(config, audit):
    return hashlib.sha256(json.dumps({"recipe": config, "audit": audit}, sort_keys=True).encode()).hexdigest()


class RunRecorder(logging.Handler):
    """Record only a run created by this client's provisioning log, even on failure."""
    def __init__(self, output):
        super().__init__()
        self.output, self.run_id = output, None

    def emit(self, record):
        import re
        match = re.search(r"Provisioned Loops run (\S+) on session", record.getMessage())
        if not match:
            # This hint is emitted immediately after successful creation, even
            # when a later SDK readiness check raises before its provision log.
            match = re.search(r"export LOOPS_REUSE_FROM_RUN_ID=([A-Za-z0-9_-]+)", record.getMessage())
        if match:
            self.run_id = match[1]
            write_json(self.output / "loops_run.json", {"run_id": self.run_id, "status": "provisioning"})


def check_metrics(metrics):
    for key, value in (metrics or {}).items():
        if isinstance(value, numbers.Real):
            if not math.isfinite(value):
                raise RuntimeError(f"Nonfinite server metric: {key}")
            if value > 0 and any(term in key.lower() for term in ("skipped", "nan_iterations")):
                raise RuntimeError(f"Server reported skipped update: {key}")


def numeric(metrics, prefix):
    check_metrics(metrics)
    return {prefix + k: float(v) for k, v in (metrics or {}).items() if isinstance(v, numbers.Real)}


async def rollouts(sampler, examples, config, *, update, training, expected_version, timeout, tokenizer):
    semaphore = asyncio.Semaphore(config["rollout_concurrency"])
    count = config["responses_per_prompt"] if training else 1

    async def one(index, example):
        async with semaphore:
            result = await asyncio.wait_for(sampler.sample_async(
                example["prompt"], num_samples=count,
                sampling_params=SamplingParams(max_tokens=config["max_new_tokens"],
                    temperature=config["temperature"] if training else config["validation_temperature"],
                    top_p=config["top_p"], top_k=-1,
                    seed=config["seed"] + update * 10000 + index),
                include_prompt_logprobs=True,
                affinity_key=f"starcount-{update}-{'train' if training else 'eval'}-{index}",
                include_routed_experts=False,
            ), timeout=timeout)
            validate_resolved_prompt(result, example["expected_prompt_ids"])
            # The fresh base sampler may omit its optional version-zero header.
            # Every post-update sampler must confirm the synchronized version.
            if (result.policy_version is None and expected_version > 0) or (
                    result.policy_version is not None and result.policy_version < expected_version):
                raise RuntimeError("Sampler did not confirm the freshly synchronized policy version.")
            if len(result.sequences) != count:
                raise RuntimeError("Sampler returned the wrong number of responses.")
            group = []
            for sequence in result.sequences:
                if not sequence.tokens or sequence.stop_reason not in ("stop", "length"):
                    raise RuntimeError("Empty completion or unknown stop reason.")
                if len(sequence.tokens) > config["max_new_tokens"]:
                    raise RuntimeError("Sampler exceeded generation budget.")
                context_truncated = context_limit_reached(
                    prompt_length=example["prompt"].length,
                    response_length=len(sequence.tokens), max_seq_len=config["max_seq_len"])
                if sequence.logprobs is None or len(sequence.tokens) != len(sequence.logprobs):
                    raise RuntimeError("Completion tokens and generation logprobs are not aligned.")
                if not all(math.isfinite(v) for v in sequence.logprobs):
                    raise RuntimeError("Nonfinite generation logprob.")
                # Keep reasoning separators; the reward scores the final answer only.
                text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
                group.append({"sequence": sequence, "example": example, "text": text,
                    "reward": reward(text, example["row"]), "policy_version": result.policy_version,
                    "context_truncated": context_truncated,
                    "valid": not (config["overlong_filtering"] and context_truncated)})
            return group

    return await asyncio.gather(*(one(i, example) for i, example in enumerate(examples)))


def record_generations(output, groups, *, update, split, run):
    table = wandb.Table(columns=["update", "split", "prompt_index", "sample", "expected", "reward", "tokens", "stop_reason", "valid_for_training", "completion", "prompt_tokens", "context_tokens", "context_truncated"])
    with (output / "generations.jsonl").open("a") as file:
        for index, group in enumerate(groups):
            for sample, item in enumerate(group):
                row = [update, split, index, sample, item["example"]["ground_truth"], item["reward"],
                    len(item["sequence"].tokens), item["sequence"].stop_reason, item["valid"], item["text"],
                    item["example"]["prompt"].length,
                    item["example"]["prompt"].length + len(item["sequence"].tokens),
                    item["context_truncated"]]
                table.add_data(*row)
                file.write(json.dumps(dict(zip(table.columns, row, strict=True))) + "\n")
    run.log({f"{split}/generations": table}, step=update, commit=False)


def make_datums(groups):
    datums, items = [], []
    for group in groups:
        advantages = leave_one_out_advantages([item["reward"] for item in group])
        for item, advantage in zip(group, advantages, strict=True):
            sequence, prompt = item["sequence"], item["example"]["prompt"]
            targets = [-100] * (prompt.length - 1) + list(sequence.tokens)
            datum = Datum(model_input=prompt.append(EncodedTextChunk(tokens=sequence.tokens[:-1])),
                loss_fn_inputs={"target_tokens": TensorData(data=targets, dtype="int64", shape=[len(targets)])},
                routed_experts_ref=sequence.routed_experts)
            if datum.model_input.length != len(targets):
                raise RuntimeError("Causal targets do not align with the multimodal input.")
            item["advantage"] = advantage
            datums.append(datum)
            items.append(item)
    return datums, items


def previous_logprobs(client, datums, items, config, timeout):
    output = []
    for start in range(0, len(datums), config["forward_batch_size"]):
        batch = datums[start:start + config["forward_batch_size"]]
        result = client.forward(batch, loss_fn="cross_entropy", loss_fn_config={"skip_token_normalization": True}).result(timeout=timeout)
        if result.metrics.get("skip_token_normalization") != 1.0:
            raise RuntimeError("Trainer does not support custom-loss normalization.")
        if len(result.loss_fn_outputs) != len(batch):
            raise RuntimeError("Trainer logprob batch count mismatch.")
        for datum, values, item in zip(batch, result.loss_fn_outputs, items[start:start + len(batch)], strict=True):
            lp = values["logprobs"].data
            length = len(datum.loss_fn_inputs["target_tokens"].data)
            offset = item["example"]["prompt"].length - 1
            if len(lp) != length or not all(math.isfinite(v) for v in lp[offset:]):
                raise RuntimeError("Trainer response logprobs are missing or misaligned.")
            output.append(list(lp[offset:]))
    return output


def learning_rate(config, update):
    # Use the zero-based update index, so update 1 starts at warmup_start_lr.
    fraction = min(1.0, (update - 1) / config["warmup_updates"])
    return config["warmup_start_lr"] + fraction * (config["learning_rate"] - config["warmup_start_lr"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--steps", type=int, help="Number of optimizer updates; defaults to the recipe target.")
    parser.add_argument("--smoke-then-full", action="store_true",
        help="Gate a fresh 50-update run on full validation and a registered checkpoint after update1.")
    parser.add_argument("--name")
    parser.add_argument("--resume", type=Path, help="Our saved resume.json; restore weights AND optimizer into a new owned run.")
    parser.add_argument("--keep-active", action="store_true", help="Leave this run billable after exit.")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    load_env(args.env_file)
    config = settings()
    if config.get("overlong_filtering_basis") != "total_context_equals_max_seq_len":
        raise ValueError("Recipe must explicitly use the total-context truncation mask.")
    target = args.steps if args.steps is not None else config["steps"]
    if not 1 <= target <= config["steps"]:
        parser.error("--steps must be between1 and50.")
    if args.smoke_then_full and (target != 50 or args.resume is not None):
        parser.error("--smoke-then-full requires a fresh 50-update run without --resume.")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number.")
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("W&B key required; no resources have been provisioned.")
    if any(os.environ.get(k) for k in ("LOOPS_REUSE_FROM_RUN_ID", "LOOPS_REUSE_FROM_SESSION_ID")):
        raise ValueError("Unset reuse variables; never attach to an unrelated run.")
    if os.environ.get("INTERNAL_LOOPS_BACKEND_VERSION"):
        raise ValueError("Unset INTERNAL_LOOPS_BACKEND_VERSION; an unrecorded backend substitution is forbidden.")
    capability = validate_catalog(config)
    examples, audit = prepare(config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "tokenizer", local_files_only=True, trust_remote_code=False)
    digest = fingerprint(config, audit)
    restored = json.loads(args.resume.read_text()) if args.resume else None
    start = restored["completed_updates"] if restored else 0
    if type(start) is not int or not 0 <= start < target:
        raise ValueError("Resume update must be a nonnegative integer below the target.")
    if restored and (restored["fingerprint"] != digest or
            restored.get("next_prompt_offset") != start * config["num_prompts"] or
            not isinstance(restored.get("trainer_checkpoint"), str) or not restored["trainer_checkpoint"]):
        raise ValueError("Resume recipe/data mismatch or checkpoint already reached target.")
    name = args.name or f"baseten_starcount_loops_r64_{target}updates_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    if not name.startswith(config["wandb_prefix"]) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
        raise ValueError("Run name must begin baseten_starcount_ and use only letters, digits, underscores and hyphens.")
    output = ROOT / "outputs" / name
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("recipe.json", config), ("input_audit.json", audit), ("capability.json", capability)):
        write_json(output / filename, value)
    logging.basicConfig(level=logging.INFO)
    recorder = RunRecorder(output)
    logger = logging.getLogger("baseten.loops.service_client")
    logger.addHandler(recorder)
    service = client = run = sampler = None
    sampling_loop = asyncio.Runner()
    complete = False
    completed_updates = start
    checkpoints = []
    try:
        run = wandb.init(entity=config["wandb_entity"], project=config["wandb_project"], name=name,
            group="baseten-starcount-loops", config=config, dir=str(output))
        run.define_metric("update")
        run.define_metric("train/*", step_metric="update")
        run.define_metric("validation/*", step_metric="update")
        run.summary["recipe_version"] = config["recipe_version"]
        run.summary["training_api"] = "baseten-loops"
        run.summary["smoke_then_full"] = args.smoke_then_full
        print("W&B:", run.url, flush=True)
        service = ServiceClient(api_key=os.environ["BASETEN_API_KEY"], base_url=API_BASE_URL,
            team=config["team"], availability_model=config["availability_model"])
        client = service.create_lora_training_client(base_model=config["base_model"], rank=config["rank"],
            replicas=1, seed=config["seed"], max_seq_len=config["max_seq_len"], with_sampler=True,
            sampler_num_replicas=1, name=name, timeout=args.timeout, ready_timeout=args.timeout)
        if not client.run_id:
            raise RuntimeError("Trainer has no run ID; refusing untracked training.")
        recorder.run_id = client.run_id
        write_json(output / "loops_run.json", {"run_id": client.run_id, "wandb_url": run.url, "status": "ready"})
        metadata = api_request(f"/v1/loops/runs/{client.run_id}")
        metadata = metadata.get("run", metadata)
        # Avoid persisting any future secret-bearing fields added to the API.
        write_json(output / "run_metadata.json", {key: metadata[key] for key in (
            "id", "run_id", "name", "base_model", "created_at", "lora_config_version", "lora_rank",
            "max_seq_len", "trainer_status", "sampler_status") if key in metadata})
        print("Owned Loops run:", client.run_id, flush=True)
        if restored:
            loaded = client.load_state_with_optimizer(restored["trainer_checkpoint"]).result(timeout=args.timeout)
            if loaded.step != start:
                raise RuntimeError("Restored optimizer step differs from resume manifest.")
        sampler = client.save_weights_and_get_sampling_client(f"initial-{start}")

        def evaluate(update):
            version = client.policy_version
            groups = sampling_loop.run(rollouts(sampler, examples["validation"], config, update=update,
                training=False, expected_version=version, timeout=args.timeout, tokenizer=tokenizer))
            accuracy = sum(g[0]["reward"] for g in groups) / len(groups)
            record_generations(output, groups, update=update, split="validation", run=run)
            metrics = {"update": update, "validation/accuracy": accuracy, "validation/examples": len(groups)}
            print(json.dumps(metrics), flush=True)
            run.log(metrics, step=update, commit=False)
            return metrics

        baseline = evaluate(start)
        run.summary["baseline_accuracy"] = baseline["validation/accuracy"]
        run.log(baseline, step=start, commit=True)
        with (output / "metrics.jsonl").open("x") as history:
            history.write(json.dumps(baseline) + "\n")
            history.flush()
            for update in range(start + 1, target + 1):
                smoke_gate = args.smoke_then_full and update == 1
                started = time.monotonic()
                offset = (update - 1) * config["num_prompts"]
                batch = examples["train"][offset:offset + config["num_prompts"]]
                if len(batch) != config["num_prompts"]:
                    raise RuntimeError("Training exhausted the one-pass fixed dataset.")
                version = client.policy_version
                groups = sampling_loop.run(rollouts(sampler, batch, config, update=update, training=True,
                    expected_version=version, timeout=args.timeout, tokenizer=tokenizer))
                rollout_seconds = time.monotonic() - started
                record_generations(output, groups, update=update, split="train", run=run)
                datums, items = make_datums(groups)
                valid_count = sum(item["valid"] for item in items)
                if valid_count == 0:
                    raise RuntimeError("Every completion is overlong; no policy update can be applied.")
                previous = previous_logprobs(client, datums, items, config, args.timeout)
                records = [build_ppo_record(prompt_length=item["example"]["prompt"].length,
                    response_token_ids=item["sequence"].tokens,
                    previous_trainer_logprobs=lp, generation_sampler_logprobs=item["sequence"].logprobs,
                    advantage=item["advantage"], sample_mask=item["valid"])
                    for d, item, lp in zip(datums, items, previous, strict=True)]
                if valid_sequence_count(records) != valid_count:
                    raise RuntimeError("Valid-sequence loss denominator is inconsistent.")
                # The previous-logprob forward scores every response. Custom backprop
                # uses record labels, masking overlong samples all the way on server.
                datums = [datum.model_copy(update={"loss_fn_inputs": {"target_tokens":
                    TensorData(data=list(record.target_tokens), dtype="int64", shape=[record.input_length])}})
                    for datum, record in zip(datums, records, strict=True)]
                aggregate = {}
                for first in range(0, len(datums), config["forward_batch_size"]):
                    last = first + config["forward_batch_size"]
                    callback = make_ppo_loss(records[first:last], global_valid_sequence_count=valid_count,
                        config=PPOLossConfig(ratio_min=config["ratio_clip_low"], ratio_max=config["ratio_clip_high"],
                            tis_min=config["tis_low"], tis_max=config["tis_high"]))
                    result = client.forward_backward_custom(datums[first:last], callback).result(timeout=args.timeout)
                    for key, value in numeric(result.metrics, "train/loss_backend/").items():
                        aggregate[key] = aggregate.get(key, 0.0) + value
                lr = learning_rate(config, update)
                optimizer = AdamParams(learning_rate=lr, beta1=config["beta1"], beta2=config["beta2"],
                    eps=config["eps"], weight_decay=config["weight_decay"], grad_clip_norm=config["grad_clip_norm"])
                optimized = client.optim_step(optimizer).result(timeout=args.timeout)
                check_metrics(optimized.metrics)
                after = client.policy_version
                if after != version + 1:
                    raise RuntimeError("Trainer did not advance exactly one optimizer update.")
                completed_updates = update
                old_sampler = sampler
                sampler = client.save_weights_and_get_sampling_client(f"update-{update}")
                sampling_loop.run(old_sampler.aclose())
                metrics = {"update": update, "train/reward": sum(i["reward"] for i in items) / len(items),
                    "train/lr": lr, "train/prompts": len(batch), "train/responses": len(items),
                    "train/valid_responses": valid_count, "train/overlong_responses": len(items) - valid_count,
                    "train/generation_length_stops": sum(i["sequence"].stop_reason == "length" for i in items),
                    "train/context_limit_responses": sum(i["context_truncated"] for i in items),
                    "train/negative_advantage_retained": sum(i["valid"] and i["advantage"] < 0 for i in items),
                    "train/negative_advantage_masked": sum(not i["valid"] and i["advantage"] < 0 for i in items),
                    "train/response_tokens": sum(len(i["sequence"].tokens) for i in items),
                    "train/rollout_seconds": rollout_seconds, "train/update_seconds": time.monotonic() - started,
                    "train/policy_version": after, **aggregate, **numeric(optimized.metrics, "train/optimizer/")}
                if smoke_gate or update % config["validation_interval"] == 0 or update == target:
                    metrics.update(evaluate(update))
                if smoke_gate and metrics["validation/examples"] != 256:
                    raise RuntimeError("Smoke validation must cover all 256 examples.")
                if smoke_gate or update % config["checkpoint_interval"] == 0 or update == target:
                    saved = client.save_state(f"step-{update}").result(timeout=args.timeout)
                    if not saved.path:
                        raise RuntimeError("Checkpoint save returned no path.")
                    if smoke_gate:
                        registered_smoke = client.list_checkpoints()
                        write_json(output / "smoke_registered_checkpoints.json",
                            [checkpoint.model_dump(mode="json") for checkpoint in registered_smoke])
                        if "step-1" not in {checkpoint.checkpoint_id for checkpoint in registered_smoke}:
                            raise RuntimeError("Smoke checkpoint step-1 is absent from the checkpoint registry.")
                    checkpoint = {"update": update, "trainer_checkpoint": saved.path}
                    checkpoints.append(checkpoint)
                    write_json(output / "checkpoints.json", checkpoints)
                    write_json(output / "resume.json", {"fingerprint": digest, "completed_updates": update,
                        "next_prompt_offset": update * config["num_prompts"], "trainer_checkpoint": saved.path,
                        "source_run_id": client.run_id, "wandb_url": run.url})
                history.write(json.dumps(metrics, allow_nan=False) + "\n")
                history.flush()
                run.log(metrics, step=update, commit=True)
                print(json.dumps(metrics, allow_nan=False), flush=True)
                if smoke_gate:
                    write_json(output / "smoke_success.json", {"passed": True, "completed_updates": 1,
                        "validation_examples": metrics["validation/examples"],
                        "validation_accuracy": metrics["validation/accuracy"],
                        "trainer_checkpoint": saved.path, "checkpoint_id": "step-1",
                        "checkpoint_registered": True, "run_id": client.run_id, "wandb_url": run.url})
                    run.summary["smoke_passed"] = True
        registered = client.list_checkpoints()
        write_json(output / "registered_checkpoints.json", [c.model_dump(mode="json") for c in registered])
        if not {f"step-{c['update']}" for c in checkpoints}.issubset({c.checkpoint_id for c in registered}):
            raise RuntimeError("Saved states are absent from the checkpoint registry.")
        complete = True
        run.summary["completed_updates"] = completed_updates
        run.summary["final_accuracy"] = metrics["validation/accuracy"]
        write_json(output / "completion.json", {"complete": True, "updates": completed_updates,
            "recipe_version": config["recipe_version"],
            "final_accuracy": metrics["validation/accuracy"], "wandb_url": run.url})
    except BaseException as error:
        write_json(output / "failure.json", {"exception_type": type(error).__name__,
            "completed_updates": completed_updates, "complete": False})
        raise
    finally:
        if sampler is not None:
            try:
                sampling_loop.run(sampler.aclose())
            except Exception:
                print("Sampler HTTP cleanup failed; checking owned run deactivation.", flush=True)
        sampling_loop.close()
        for connection in (client, service):
            if connection is not None and hasattr(connection, "close"):
                try:
                    connection.close()
                except Exception:
                    print("Client connection cleanup failed; checking owned run deactivation.", flush=True)
        if recorder.run_id and not args.keep_active:
            try:
                cleanup = cleanup_owned_run(recorder.run_id, output)
                if cleanup["verified"]:
                    print("Verified owned run and paired sampler stopped:", recorder.run_id, flush=True)
                else:
                    print("CLEANUP NOT VERIFIED: inspect", output / "deactivation.json", flush=True)
            except Exception:
                print("CLEANUP REQUIRED: use cleanup.py --output with this run's output directory and --yes.", flush=True)
        elif recorder.run_id:
            print("Run remains active/billable; use cleanup.py --output with this run's output directory and --yes.", flush=True)
        logger.removeHandler(recorder)
        if run is not None:
            run.finish(exit_code=0 if complete else 1)


if __name__ == "__main__":
    main()
