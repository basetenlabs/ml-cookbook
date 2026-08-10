#!/usr/bin/env python3
"""Generate 5 synthetic training runs into ../runs/ — a stand-in for sync.sh,
so you can demo the dashboard without S3 or AWS credentials.

Each run dir gets exactly what training code would upload to
s3://<bucket>/<prefix>/<run_id>/ (see README "What your training runs must write"):
  metrics.jsonl     one record per step with the usual W&B-style scalars:
                    train_loss, learning_rate (warmup + cosine decay),
                    grad_norm (noisy; spikes on the divergent run), epoch,
                    tokens_per_sec (ramp to a plateau + noise), and eval_loss
                    every EVAL_EVERY steps — plus step and time
  hyperparams.json  lr, batch_size, model, max_steps, warmup_steps
  meta.json         source (ec2|baseten), started_at, git_sha

Reproducible: fixed RNG seed, no per-process randomness.
"""
import json
import math
import random
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "runs"
EVAL_EVERY = 25
DATASET_EXAMPLES = 12_000  # for the epoch counter: epoch = step * batch / this

# Deterministic stand-in for hash(): str hashes are salted per process.
stable_hash = lambda s: zlib.crc32(s.encode())

# run_id, lr, batch_size, model, max_steps, warmup, source, started_at, git_sha, behavior
RUNS = [
    ("run-2026-07-25-a1b2", 1e-5, 16, "qwen2.5-7b", 800, 40, "ec2",
     "2026-07-25T09:14:02Z", "8f3ac1d", "slow"),       # slow convergence, high floor
    ("run-2026-07-26-b2c3", 3e-5, 32, "qwen2.5-7b", 600, 30, "baseten",
     "2026-07-26T11:41:55Z", "8f3ac1d", "good"),       # solid
    ("run-2026-07-27-c3d4", 1e-4, 32, "qwen2.5-7b", 500, 25, "ec2",
     "2026-07-27T16:03:11Z", "b71e920", "best"),       # fastest, lowest floor
    ("run-2026-07-28-d4e5", 3e-4, 16, "qwen2.5-7b", 400, 20, "baseten",
     "2026-07-28T08:22:47Z", "b71e920", "diverge"),    # lr too hot: dips then blows up
    ("run-2026-07-29-e5f6", 1e-4, 16, "qwen2.5-7b", 650, 30, "ec2",
     "2026-07-29T10:55:30Z", "c904f7e", "decent"),     # good but noisier (small batch)
]

BEHAVIOR = {
    # (decay_rate, floor, noise)
    "slow":    (0.0035, 1.55, 0.020),
    "good":    (0.0090, 1.22, 0.018),
    "best":    (0.0160, 1.08, 0.016),
    "decent":  (0.0130, 1.18, 0.030),
    "diverge": (0.0200, 1.15, 0.028),
}

START_LOSS = 2.85


def loss_at(step, behavior, rng, max_steps):
    decay, floor, noise = BEHAVIOR[behavior]
    base = floor + (START_LOSS - floor) * math.exp(-decay * step)
    if behavior == "diverge":
        # trains OK for the first ~35% of the run, then the loss walks up and
        # gets spiky — classic too-hot learning rate.
        turn = int(max_steps * 0.35)
        if step > turn:
            t = step - turn
            base += 0.004 * t + 0.35 * math.sin(t / 9.0) ** 2 * (t / max_steps)
            noise *= 3.0
    return max(0.05, base + rng.gauss(0, noise))


def lr_at(step, lr, warmup, max_steps):
    """Linear warmup to lr, then cosine decay to 10% of lr."""
    if step <= warmup:
        return lr * step / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def grad_norm_at(step, behavior, rng, max_steps):
    """High early, settles as the loss flattens; lognormal jitter. The
    divergent run's gradients grow and spike after the loss turns."""
    decay, _, _ = BEHAVIOR[behavior]
    base = 0.6 + 2.4 * math.exp(-1.5 * decay * step)
    val = base * math.exp(rng.gauss(0, 0.18))
    if behavior == "diverge":
        turn = int(max_steps * 0.35)
        if step > turn:
            val *= 1.0 + 2.5 * (step - turn) / max_steps
            if rng.random() < 0.08:            # occasional loss-spike gradients
                val *= rng.uniform(3.0, 9.0)
    return val


def tokens_per_sec_at(step, bs, sec_per_step, rng):
    """Throughput: short ramp (compile/caching) to a plateau, small jitter."""
    plateau = bs * 1024 / sec_per_step
    ramp = min(1.0, 0.55 + 0.045 * step)
    return plateau * ramp * (1 + rng.gauss(0, 0.02))


def main():
    rng = random.Random(20260729)
    for run_id, lr, bs, model, max_steps, warmup, source, started_at, sha, beh in RUNS:
        d = ROOT / run_id
        d.mkdir(parents=True, exist_ok=True)

        # wall-clock: a fake but stable epoch base per run
        t0 = 1_753_000_000 + stable_hash(run_id) % 100_000
        sec_per_step = 2.2 if bs == 16 else 3.6

        with (d / "metrics.jsonl").open("w") as f:
            for step in range(1, max_steps + 1):
                rec = {
                    "step": step,
                    "time": round(t0 + step * sec_per_step, 2),
                    "train_loss": round(loss_at(step, beh, rng, max_steps), 4),
                    "learning_rate": round(lr_at(step, lr, warmup, max_steps), 10),
                    "grad_norm": round(grad_norm_at(step, beh, rng, max_steps), 4),
                    "epoch": round(step * bs / DATASET_EXAMPLES, 4),
                    "tokens_per_sec": round(tokens_per_sec_at(step, bs, sec_per_step, rng), 1),
                }
                if step % EVAL_EVERY == 0 or step == max_steps:
                    # eval loss tracks train loss with a small generalization gap
                    ev = loss_at(step, beh, rng, max_steps) + 0.06 + rng.gauss(0, 0.01)
                    rec["eval_loss"] = round(max(0.05, ev), 4)
                f.write(json.dumps(rec) + "\n")

        (d / "hyperparams.json").write_text(json.dumps({
            "lr": lr,
            "batch_size": bs,
            "model": model,
            "max_steps": max_steps,
            "warmup_steps": warmup,
        }, indent=2) + "\n")

        (d / "meta.json").write_text(json.dumps({
            "source": source,
            "started_at": started_at,
            "git_sha": sha,
            "host": f"{source}-trainer-{stable_hash(run_id) % 90 + 10}",
        }, indent=2) + "\n")

        print(f"wrote {run_id}: {max_steps} steps, lr={lr}, batch={bs}, {beh}")

    print(f"\n{len(RUNS)} runs in {ROOT}")
    print("next: python3 build_manifests.py && python3 dashboard.py")


if __name__ == "__main__":
    main()
