import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--base-only", action="store_true", help="Evaluate the base model before training"
    )
    mode.add_argument(
        "--checkpoint",
        help="checkpoints.json or bt://loops:RUN/sampler_weights/NAME",
    )
    result.add_argument("--model", default=MODEL)
    result.add_argument("--data", type=Path, default=ROOT / "data/banking77-validation.jsonl")
    result.add_argument(
        "--output", type=Path, default=ROOT / "runs" / f"eval-{datetime.now():%Y%m%d-%H%M%S-%f}"
    )
    result.add_argument("--max-seq-len", type=int, default=1024)
    return result


def parse_prediction(content, labels):
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(content, object_pairs_hook=unique_pairs)
        if isinstance(value, dict) and set(value) == {"intent"}:
            if isinstance(value["intent"], str) and value["intent"] in labels:
                return value["intent"]
    except (ValueError, TypeError):
        pass
    return None


def load_inputs(args):
    checkpoint = None
    if args.checkpoint:
        raw = str(args.checkpoint)
        checkpoint = (
            {"weights": raw} if raw.startswith("bt://") else json.loads(Path(raw).read_text())
        )
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("weights"), str):
            raise ValueError("Checkpoint must contain a weights URI")
        match = re.fullmatch(
            r"bt://loops:([A-Za-z0-9_-]+)/sampler_weights/([A-Za-z0-9_-]+)",
            checkpoint["weights"],
        )
        if not match or checkpoint.get("run_id", match[1]) != match[1]:
            raise ValueError("Expected sampler_weights URI with matching run_id")
        if not raw.startswith("bt://") and "run_id" not in checkpoint:
            raise ValueError("Checkpoint file must contain weights and run_id")
        checkpoint = {"weights": checkpoint["weights"], "run_id": match[1]}
    if args.max_seq_len <= 256:
        raise ValueError("--max-seq-len must leave room for prompt plus 256 output tokens")
    labels = json.loads((ROOT / "data/banking77-labels.json").read_text())
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    if not rows or len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("Expected nonempty validation data and 77 unique labels")
    seen = set()
    for row in rows:
        identifier, messages = row.get("id"), row.get("messages")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Expected unique nonempty example IDs")
        seen.add(identifier)
        if (
            not isinstance(messages, list)
            or len(messages) < 2
            or any(
                not isinstance(m, dict)
                or m.get("role") not in ("system", "user", "assistant")
                or not isinstance(m.get("content"), str)
                or not m["content"].strip()
                for m in messages
            )
            or messages[-1]["role"] != "assistant"
        ):
            raise ValueError("Expected text messages ending in assistant reference")
        row["label"] = parse_prediction(messages[-1]["content"], labels)
        if row["label"] is None:
            raise ValueError("Reference must be strict allowed-label intent JSON")
    return rows, labels, checkpoint


def encode_prompt(tokenizer, row, max_seq_len):
    tokens = tokenizer.apply_chat_template(
        row["messages"][:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokens = tokens["input_ids"] if hasattr(tokens, "keys") else tokens
    if not tokens or len(tokens) + 256 > max_seq_len:
        raise ValueError("Prompt plus 256 output tokens exceeds --max-seq-len")
    return tokens


def complete_score(rows, labels, base, adapter=None):
    import random
    from collections import Counter

    results, correct = {}, {}
    gold = {row["id"]: row["label"] for row in rows}
    if not gold or len(gold) != len(rows):
        raise ValueError("Expected unique, nonempty gold IDs")
    arms = {"base": base}
    if adapter is not None:
        arms["adapter"] = adapter
    for arm, records in arms.items():
        lookup = {record["id"]: record for record in records}
        if (
            len(lookup) != len(records)
            or set(lookup) != set(gold)
            or any(record.get("status") != "ok" for record in records)
        ):
            raise ValueError(
                "Incomplete evaluation: missing, failed or truncated rows; no complete score"
            )
        pairs = [
            (label, parse_prediction(lookup[identifier]["content"], labels))
            for identifier, label in gold.items()
        ]
        correct[arm] = [int(expected == predicted) for expected, predicted in pairs]
        true_positive = Counter(expected for expected, predicted in pairs if expected == predicted)
        support = Counter(gold.values())
        predicted_count = Counter(predicted for _, predicted in pairs)
        f1 = [
            2 * true_positive[label] / (support[label] + predicted_count[label])
            if support[label] + predicted_count[label]
            else 0
            for label in labels
        ]
        valid = sum(predicted is not None for _, predicted in pairs)
        results[arm] = dict(
            total=len(rows),
            valid=valid,
            invalid=len(rows) - valid,
            correct=sum(correct[arm]),
            accuracy=sum(correct[arm]) / len(rows),
            macro_f1=sum(f1) / len(labels),
        )
    if adapter is None:
        return results
    delta = [b - a for a, b in zip(correct["base"], correct["adapter"])]
    rng = random.Random(1111)
    boots = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(10000))
    results["paired"] = dict(
        adapter_minus_base_accuracy=sum(delta) / len(delta),
        adapter_only_correct=delta.count(1),
        base_only_correct=delta.count(-1),
        bootstrap_ci95=[boots[250], boots[9749]],
        method="10000 paired percentile bootstrap seed1111",
    )
    return results


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def shutdown(api, receipt, timeout=180):
    import time

    path = f"/samplers/{receipt['id']}"
    deadline = time.monotonic() + timeout
    target = api("GET", path)["sampler"]
    if any(target.get(key) != receipt[key] for key in ("id", "model_id", "deployment_id")):
        raise ValueError("Sampler identity mismatch; refusing deactivation")
    api("POST", path + "/deactivate")
    while True:
        target = api("GET", path)["sampler"]
        if any(target.get(key) != receipt[key] for key in ("id", "model_id", "deployment_id")):
            raise ValueError("Sampler identity changed during shutdown")
        if target["status"]["name"] == "INACTIVE":
            return {"sampler_id": receipt["id"], "verified": True, "status": "INACTIVE"}
        if time.monotonic() >= deadline:
            raise RuntimeError("Sampler shutdown timed out; inspect Baseten dashboard")
        time.sleep(2)


def evaluate_arm(api, key, session, args, rows, labels, arm, checkpoint):
    from baseten.loops import ModelInput, SamplingParams
    from baseten.loops.sampling_client import BasetenDeployment, SamplingClient

    receipt = client = None
    try:
        source = (
            {"base_model": args.model} if arm == "base" else {"model_path": checkpoint["weights"]}
        )
        receipt = api(
            "POST",
            "/samplers",
            dict(
                session_id=session,
                availability_model="dedicated",
                max_seq_length=args.max_seq_len,
                num_replicas=1,
                **source,
            ),
        )["sampler"]
        # Ownership exists in memory before any disk write, tokenizer or readiness wait.
        write_json(
            args.output / f"{arm}-sampler.json",
            {
                "session_id": session,
                **{k: receipt.get(k) for k in ("id", "model_id", "deployment_id", "base_model")},
            },
        )
        if receipt["base_model"] != args.model:
            raise ValueError("Sampler/checkpoint base model differs from --model")
        client = SamplingClient(
            base_model=receipt["base_model"],
            sampler_id=receipt["id"],
            session_id=session,
            checkpoint_path=source.get("model_path"),
            api_key=key,
            timeout=600,
            ready_timeout=1800,
            deployment=BasetenDeployment(
                base_url=receipt["base_url"],
                model_id=receipt["model_id"],
                deployment_id=receipt["deployment_id"],
                api_base_url="https://api.baseten.co",
            ),
        )
        tokenizer = client.get_tokenizer()
        if not isinstance(tokenizer.eos_token_id, int):
            raise ValueError("Tokenizer must supply native EOS token ID")
        params = SamplingParams(
            temperature=0, top_p=1, seed=1111, max_tokens=256, stop=[tokenizer.eos_token_id]
        )
        prompts = [encode_prompt(tokenizer, row, args.max_seq_len) for row in rows]
        records = []
        with (args.output / f"{arm}.jsonl").open("x") as log:
            for index, (row, prompt) in enumerate(zip(rows, prompts), 1):
                record = dict(id=row["id"], status="error", prompt_token_ids=prompt)
                try:
                    response = client.sample(
                        ModelInput.from_ints(prompt), num_samples=1, sampling_params=params
                    )
                    record["response"] = response.model_dump(mode="json")
                    if len(response.sequences) != 1:
                        raise ValueError("Expected exactly one completion")
                    sequence = response.sequences[0]
                    tokens = sequence.tokens
                    if not tokens and sequence.text is None:
                        raise ValueError("Completion contains neither tokens nor text")
                    content = (
                        tokenizer.decode(
                            tokens[:-1] if tokens[-1:] == [tokenizer.eos_token_id] else tokens,
                            skip_special_tokens=False,
                        )
                        if tokens
                        else sequence.text
                    )
                    prediction = parse_prediction(content, labels)
                    truncated = sequence.stop_reason == "length" or (
                        len(tokens) >= 256 and tokens[-1] != tokenizer.eos_token_id
                    )
                    record.update(
                        content=content,
                        prediction=prediction,
                        valid=prediction is not None,
                        raw_token_ids=tokens,
                        returned_text=sequence.text,
                        stop_token_ids=[tokenizer.eos_token_id],
                        finish_reason=sequence.stop_reason,
                        status="truncated" if truncated else "ok",
                    )
                    if truncated:
                        raise ValueError("Truncated completion; no complete score")
                except BaseException as exc:
                    record["error_type"] = type(exc).__name__
                    raise
                finally:
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                records.append(record)
                if index % 50 == 0 or index == len(rows):
                    print(f"{arm}: {index}/{len(rows)}", flush=True)
        return records
    finally:
        try:
            if client is not None:
                client.close()  # Closes connections, NOT billable GPUs.
        finally:
            # Never include exception text or HTTP headers in artifacts/CLI errors.
            report = dict(
                verified=False, provisioning_ownership_unknown=receipt is None, session_id=session
            )
            try:
                if receipt is None:
                    raise RuntimeError(
                        "Create response lost: inspect Baseten dashboard/support; never auto-sweep unrelated samplers"
                    )
                report.update(shutdown(api, receipt))
            except BaseException as exc:
                report.update(
                    error_type=type(exc).__name__,
                    message="Shutdown unverified: inspect Baseten dashboard/support before retrying",
                )
                raise RuntimeError(report["message"]) from exc
            finally:
                write_json(args.output / f"{arm}-cleanup.json", report)


def main(argv=None):
    args = parser().parse_args(argv)  # --help never reads credentials or creates resources.
    import os
    import time

    import requests

    rows, labels, checkpoint = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    key = os.environ["BASETEN_API_KEY"]

    def api(method, path, body=None):
        response = requests.request(
            method,
            "https://api.baseten.co/v1/loops" + path,
            json=body,
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    started = time.monotonic()
    metadata = dict(
        model=args.model,
        checkpoint=checkpoint,
        data=str(args.data),
        rows=len(rows),
        max_seq_len=args.max_seq_len,
        sdk="baseten-loops==0.28.0",
        sampling=dict(
            temperature=0,
            top_p=1,
            seed=1111,
            max_tokens=256,
            enable_thinking=False,
            num_replicas=1,
            structured_outputs=None,
        ),
    )
    write_json(args.output / "metadata.json", metadata)
    try:
        session = api("POST", "/sessions", {})["session"]["id"]
        write_json(args.output / "session.json", {"session_id": session})
        base = evaluate_arm(api, key, session, args, rows, labels, "base", checkpoint)
        adapter = (
            None
            if args.base_only
            else evaluate_arm(api, key, session, args, rows, labels, "adapter", checkpoint)
        )
        scores = complete_score(rows, labels, base, adapter)
        scores.update(complete=True, metadata=metadata, elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "scores.json", scores)
        for arm in ("base",) if args.base_only else ("base", "adapter"):
            result = scores[arm]
            print(
                f"{arm}: {result['correct']}/{result['total']} correct; accuracy {result['accuracy']:.2%}; "
                f"macro-F1 {result['macro_f1']:.4f}; valid JSON {result['valid']}/{result['total']}"
            )
        if not args.base_only:
            print(f"Adapter minus base: {scores['paired']['adapter_minus_base_accuracy']:+.2%}")
        print(f"Artifacts: {args.output}")
    except BaseException as exc:
        write_json(
            args.output / "failure.json", {"complete": False, "error_type": type(exc).__name__}
        )
        raise SystemExit(
            f"Evaluation failed ({type(exc).__name__}); inspect artifacts in {args.output}"
        ) from None


if __name__ == "__main__":
    main()
