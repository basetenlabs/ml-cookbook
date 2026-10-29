import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import requests
from baseten.loops import AdamParams, Datum, ModelInput, ServiceClient, TensorData


def to_datum(tokenizer, row, max_seq_len=1024):
    def render(messages, generation=False):
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation,
            enable_thinking=False,
        )
        return ids["input_ids"] if hasattr(ids, "keys") else ids

    prefix, full = render(row["messages"][:-1], True), render(row["messages"])
    if not prefix or full[: len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Tokenizer assistant prefix mismatch")
    if len(full) > max_seq_len or tokenizer.eos_token_id not in full[len(prefix) :]:
        raise ValueError(f"Example exceeds {max_seq_len} tokens or lacks native EOS")
    targets = [-100] * (len(prefix) - 1) + full[len(prefix) :]
    return Datum(
        model_input=ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=targets, dtype="int64", shape=[len(targets)]
            )
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "runs"
        / f"lora-{datetime.now():%Y%m%d-%H%M%S-%f}",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args(argv)

    if (
        min(args.batch_size, args.epochs, args.rank, args.max_seq_len) <= 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        raise ValueError("Sizes, epochs and finite learning rate must be positive")

    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    if not rows:
        raise ValueError("Training data must be nonempty")
    for row in rows:
        messages = row.get("messages") if isinstance(row, dict) else None
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or m.get("role") not in ("system", "user", "assistant")
                or not isinstance(m.get("content"), str)
                or not m["content"].strip()
                for m in messages
            )
            or messages[-1]["role"] != "assistant"
        ):
            raise ValueError("Expected nonempty text messages ending in assistant")

    key = os.environ["BASETEN_API_KEY"]
    if any(
        os.environ.get(k)
        for k in (
            "LOOPS_REUSE_FROM_RUN_ID",
            "LOOPS_REUSE_FROM_SESSION_ID",
            "LOOPS_BASE_URL",
            "LOOPS_TEAM",
        )
    ):
        raise ValueError(
            "Unset Loops reuse/routing overrides: this script owns a fresh run"
        )

    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2))

    def api(method, path):
        response = requests.request(
            method,
            "https://api.baseten.co/v1/loops" + path,
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    service = trainer = run = None

    def finish_tracking(exc_type, exc, traceback):
        if run is not None:
            try:
                run.finish(exit_code=0 if exc_type is None else 1)
            except Exception:
                print(
                    "W&B finalization failed; remote sync unconfirmed.", file=sys.stderr
                )

    def shutdown():
        try:
            if trainer is not None:
                trainer.close()
        finally:
            if service is not None:
                owned = [
                    r["id"]
                    for r in api("GET", "/runs")["runs"]
                    if r["session_id"] == service.session_id
                ]
                for run_id in owned:
                    print(f"Deactivating Baseten run {run_id}...", flush=True)
                    api("POST", f"/runs/{run_id}/deactivate")
                    for _ in range(30):
                        result = api("GET", f"/runs/{run_id}")
                        if result.get("run", result)["status"]["name"] == "INACTIVE":
                            break
                        time.sleep(2)
                    else:
                        raise RuntimeError(
                            f"Shutdown unverified: check Baseten run {run_id}"
                        )
                save("cleanup.json", {"verified": True, "deactivated_run_ids": owned})
                print("Baseten shutdown verified.", flush=True)

    with ExitStack() as cleanup:
        cleanup.push(finish_tracking)
        if args.wandb_project:
            import wandb

            if wandb.run is not None or not wandb.login(
                verify=True, timeout=30, prompt=False
            ):
                raise RuntimeError("W&B requires a fresh authenticated online run")
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.output.name,
                mode="online",
                resume="never",
                dir=str(args.output),
                settings=wandb.Settings(
                    init_timeout=30, finish_timeout=30, finish_timeout_raises=True
                ),
            )
            if run is None or run.settings.mode != "online" or not run.url:
                raise RuntimeError("W&B did not create an online run")
            print(f"W&B URL: {run.url}", flush=True)

        cleanup.callback(shutdown)
        service = ServiceClient(availability_model="dedicated")
        trainer = service.create_lora_training_client(
            base_model=args.model,
            rank=args.rank,
            seed=1111,
            max_seq_len=args.max_seq_len,
            with_sampler=False,
            name=args.output.name,
        )
        print(f"run_id={trainer.run_id}", flush=True)
        save("run.json", {"run_id": trainer.run_id, "session_id": service.session_id})

        tokenizer = trainer.get_tokenizer()
        data = [to_datum(tokenizer, row, args.max_seq_len) for row in rows]
        rng, step = random.Random(1111), 0
        total = math.ceil(len(data) / args.batch_size) * args.epochs

        with (args.output / "training.jsonl").open("w", buffering=1) as log:
            for _ in range(args.epochs):
                rng.shuffle(data)
                for start in range(0, len(data), args.batch_size):
                    batch = data[start : start + args.batch_size]
                    step += 1
                    loss = trainer.forward_backward(batch).result(timeout=600).loss
                    if not math.isfinite(loss):
                        raise ValueError(
                            "Nonfinite loss; stopping before optimizer update"
                        )
                    trainer.optim_step(
                        AdamParams(learning_rate=args.learning_rate)
                    ).result(timeout=600)
                    record = {
                        "step": step,
                        "loss": loss,
                        "learning_rate": args.learning_rate,
                        "examples": len(batch),
                    }
                    log.write(json.dumps(record) + "\n")
                    print(f"Training {step}/{total}: loss={loss:.6f}", flush=True)
                    if run is not None:
                        run.log(record, step=step)

        print("Saving resumable state...", flush=True)
        state = trainer.save_state(name="final").result(timeout=600)
        print(f"Saved state: {state.path}", flush=True)
        print("Saving sampler weights...", flush=True)
        weights = trainer.save_weights_for_sampler(name="final").result(timeout=600)
        print(f"Saved sampler weights: {weights.path}", flush=True)
        save(
            "checkpoints.json",
            {"state": state.path, "weights": weights.path, "run_id": trainer.run_id},
        )
        print(f"Checkpoint receipts: {args.output / 'checkpoints.json'}", flush=True)


if __name__ == "__main__":
    main()
