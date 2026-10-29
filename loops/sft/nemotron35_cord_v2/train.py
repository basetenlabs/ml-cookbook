import argparse
import json
import logging
import math
import numbers
import os
import random
import re
import time
from datetime import UTC, datetime

import wandb
from baseten.loops import AdamParams, ServiceClient

from common import ROOT, api_request, load_env, prepare, settings, validate_catalog
from data import datum

LOGGER = logging.getLogger(__name__)


class RunRecorder(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.run_id = None

    def emit(self, record):
        match = re.search(
            r"Provisioned Loops run (\S+) on session", record.getMessage()
        )
        if match:
            self.run_id = match[1]
            self.path.write_text(
                json.dumps({"run_id": self.run_id, "status": "provisioning"}, indent=2)
            )


def numeric_metrics(metrics, prefix):
    return {
        prefix + key.replace(" ", "_"): float(value)
        for key, value in (metrics or {}).items()
        if isinstance(value, numbers.Real) and math.isfinite(value)
    }


def check_optimizer_metrics(metrics):
    for key, value in (metrics or {}).items():
        if isinstance(value, numbers.Real) and not math.isfinite(value):
            raise RuntimeError(f"Optimizer returned a nonfinite metric: {key}")
        if (
            isinstance(value, numbers.Real)
            and value > 0
            and any(term in key.lower() for term in ["skipped", "nan_iterations"])
        ):
            raise RuntimeError(f"Optimizer reported an unsuccessful update: {key}")


def validation(client, examples, batch_size, timeout):
    weighted_loss = 0.0
    token_count = 0
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset : offset + batch_size]
        result = client.forward(
            data=[datum(example) for example in batch],
            loss_fn="cross_entropy",
            loss_fn_config={"skip_token_normalization": False},
        ).result(timeout=timeout)
        if not math.isfinite(result.loss):
            raise RuntimeError("Validation produced a nonfinite loss.")
        count = sum(example["active_tokens"] for example in batch)
        weighted_loss += result.loss * count
        token_count += count
    return weighted_loss / token_count


def batches(examples, size, seed):
    generator = random.Random(seed)
    while True:
        indices = list(range(len(examples)))
        generator.shuffle(indices)
        for offset in range(0, len(indices) - size + 1, size):
            yield [examples[index] for index in indices[offset : offset + size]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--name")
    parser.add_argument("--keep-active", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    arguments = parser.parse_args()
    load_env(arguments.env_file)
    config = settings()
    if arguments.max_seq_len is not None:
        if arguments.max_seq_len < 1:
            parser.error("--max-seq-len must be positive.")
        config["max_seq_len"] = arguments.max_seq_len
    if arguments.steps is not None:
        if arguments.steps < 1 or arguments.steps > config["steps"]:
            parser.error("--steps must be between 1 and 400.")
        config["steps"] = arguments.steps
    if (
        not os.environ.get("WANDB_API_KEY")
        and os.environ.get("WANDB_MODE") != "offline"
    ):
        raise ValueError("W&B credential is required; nothing has been provisioned.")
    if any(
        os.environ.get(name)
        for name in ["LOOPS_REUSE_FROM_RUN_ID", "LOOPS_REUSE_FROM_SESSION_ID"]
    ):
        raise ValueError(
            "Unset Loops reuse variables; this script must not attach to unrelated warm training runs."
        )
    logging.basicConfig(level=logging.INFO)
    capability = validate_catalog(config)
    examples, token_audit = prepare(config)
    name = (
        arguments.name
        or f"baseten_cord_v2_loops_r64_gb8_lr2e4_steps{config['steps']}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    )
    if not name.startswith(config["wandb_prefix"]) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", name
    ):
        raise ValueError(
            "Run names must begin baseten_ and contain only letters, digits, dots, underscores or hyphens."
        )
    output = ROOT / "outputs" / name
    output.mkdir(parents=True, exist_ok=False)
    (output / "recipe.json").write_text(json.dumps(config, indent=2))
    (output / "token_audit.json").write_text(json.dumps(token_audit, indent=2))
    (output / "model_capability.json").write_text(json.dumps(capability, indent=2))
    recorder = RunRecorder(output / "loops_run.json")
    logger = logging.getLogger("baseten.loops.service_client")
    logger.addHandler(recorder)
    client = None
    run = None
    completed = False
    checkpoints = []
    try:
        run = wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=name,
            group="baseten_cord_v2",
            config=config,
            dir=str(output),
        )
        run.define_metric("baseten_update")
        run.define_metric("baseten_*", step_metric="baseten_update")
        run.summary["baseten_exact_reproduction"] = False
        print("W&B:", run.url, flush=True)
        service = ServiceClient(team=config["team"])
        client = service.create_lora_training_client(
            base_model=config["base_model"],
            rank=config["rank"],
            replicas=config["replicas"],
            seed=config["seed"],
            max_seq_len=config["max_seq_len"],
            with_sampler=False,
            name=name,
            timeout=arguments.timeout,
            ready_timeout=arguments.timeout,
        )
        if not client.run_id:
            raise RuntimeError(
                "Created trainer has no run ID; refusing to train without lifecycle tracking."
            )
        recorder.run_id = client.run_id
        (output / "loops_run.json").write_text(
            json.dumps(
                {"run_id": client.run_id, "wandb_url": run.url, "status": "ready"},
                indent=2,
            )
        )
        print("Loops run:", client.run_id, flush=True)
        longest = max(
            examples["train"],
            key=lambda row: row.get("sequence_tokens", row["active_tokens"]),
        )
        probe = client.forward(
            data=[datum(longest)],
            loss_fn="cross_entropy",
            loss_fn_config={"skip_token_normalization": False},
        ).result(timeout=arguments.timeout)
        if not math.isfinite(probe.loss):
            raise RuntimeError(
                "Longest-receipt forward probe produced a nonfinite loss."
            )
        probe_audit = {
            "baseten_update": 0,
            "baseten_alignment_probe_loss": probe.loss,
            "baseten_alignment_probe_tokens": longest.get("sequence_tokens"),
            "baseten_alignment_probe_row": longest.get("row_index"),
        }
        (output / "alignment_probe.json").write_text(json.dumps(probe_audit, indent=2))
        run.log(probe_audit, step=0)
        print("Longest-receipt forward probe:", json.dumps(probe_audit), flush=True)
        optimizer = AdamParams(
            learning_rate=config["learning_rate"],
            beta1=config["beta1"],
            beta2=config["beta2"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
            grad_clip_norm=config["grad_clip_norm"],
        )
        stream = batches(examples["train"], config["batch_size"], config["seed"])
        with (output / "metrics.jsonl").open("x") as history:
            for step in range(1, config["steps"] + 1):
                started = time.monotonic()
                batch = next(stream)
                result = client.forward_backward(
                    data=[datum(example) for example in batch],
                    loss_fn="cross_entropy",
                    loss_fn_config={"skip_token_normalization": False},
                ).result(timeout=arguments.timeout)
                if not math.isfinite(result.loss):
                    raise RuntimeError(
                        "Nonfinite train loss; optimizer update is not applied."
                    )
                optimizer_result = client.optim_step(optimizer).result(
                    timeout=arguments.timeout
                )
                check_optimizer_metrics(optimizer_result.metrics)
                metrics = {
                    "baseten_update": step,
                    "baseten_train_loss": result.loss,
                    "baseten_learning_rate": config["learning_rate"],
                    "baseten_batch_size": len(batch),
                    "baseten_active_tokens": sum(
                        example["active_tokens"] for example in batch
                    ),
                    "baseten_step_seconds": time.monotonic() - started,
                }
                metrics.update(
                    numeric_metrics(result.metrics, "baseten_forward_backward/")
                )
                metrics.update(
                    numeric_metrics(optimizer_result.metrics, "baseten_optimizer/")
                )
                if step % config["validation_interval"] == 0 or step == config["steps"]:
                    metrics["baseten_validation_loss"] = validation(
                        client,
                        examples["validation"],
                        config["batch_size"],
                        arguments.timeout,
                    )
                    metrics["baseten_validation_rows"] = len(examples["validation"])
                if step % config["checkpoint_interval"] == 0 or step == config["steps"]:
                    trainer = client.save_state(name=f"step-{step}").result(
                        timeout=arguments.timeout
                    )
                    sampler = client.save_weights_for_sampler(
                        name=f"step-{step}"
                    ).result(timeout=arguments.timeout)
                    if not trainer.path or not sampler.path:
                        raise RuntimeError(
                            "Save operation did not return both checkpoint paths."
                        )
                    checkpoints.append(
                        {
                            "step": step,
                            "trainer_state": trainer.path,
                            "sampler_weights": sampler.path,
                        }
                    )
                    (output / "checkpoints.json").write_text(
                        json.dumps(checkpoints, indent=2)
                    )
                history.write(json.dumps(metrics) + "\n")
                history.flush()
                run.log(metrics, step=step)
                print(json.dumps(metrics), flush=True)
        registered = client.list_checkpoints()
        (output / "registered_checkpoints.json").write_text(
            json.dumps(
                [checkpoint.model_dump(mode="json") for checkpoint in registered],
                indent=2,
            )
        )
        expected_names = {f"step-{checkpoint['step']}" for checkpoint in checkpoints}
        if not expected_names.issubset(
            {checkpoint.checkpoint_id for checkpoint in registered}
        ):
            raise RuntimeError(
                "Saved checkpoint names are missing from the Loops checkpoint registry."
            )
        run.summary["baseten_completed_updates"] = config["steps"]
        run.summary["baseten_final_train_loss"] = metrics["baseten_train_loss"]
        run.summary["baseten_final_validation_loss"] = metrics[
            "baseten_validation_loss"
        ]
        from plot_loss import plot

        image = plot(output / "metrics.jsonl")
        run.log({"baseten_loss_curves": wandb.Image(str(image))}, step=config["steps"])
        completed = True
        (output / "completion.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "updates": config["steps"],
                    "checkpoint_steps": [
                        checkpoint["step"] for checkpoint in checkpoints
                    ],
                    "wandb_url": run.url,
                },
                indent=2,
            )
        )
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as error:
                LOGGER.exception(
                    "Client close failed: %s", type(error).__name__, exc_info=False
                )
        if recorder.run_id and not arguments.keep_active:
            try:
                api_request(
                    f"/v1/loops/runs/{recorder.run_id}/deactivate", method="POST"
                )
                (output / "deactivation.json").write_text(
                    json.dumps({"run_id": recorder.run_id, "requested": True}, indent=2)
                )
                print(
                    "Deactivated only this recipe's Loops run:",
                    recorder.run_id,
                    flush=True,
                )
            except Exception as error:
                LOGGER.exception(
                    "Cleanup needs attention: baseten loops run deactivate --run-id %s --yes. %s",
                    recorder.run_id,
                    type(error).__name__,
                    exc_info=False,
                )
        elif recorder.run_id:
            print(
                f"Run remains active and billable: baseten loops run deactivate --run-id {recorder.run_id} --yes",
                flush=True,
            )
        logger.removeHandler(recorder)
        if run is not None:
            run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
