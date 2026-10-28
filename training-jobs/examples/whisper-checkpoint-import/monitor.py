"""Follow an import through checkpoint sync, then follow a UI-created deployment."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def submitted_job(output):
    match = re.search(r"--job-id\s+([a-zA-Z0-9]+)", output)
    if not match:
        raise ValueError(
            "Could not read the submitted job ID. Find it in the output above, then "
            "resume with --monitor-job <id>; do not resubmit the import."
        )
    return match[1]


class Monitor:
    def __init__(self, profile=None, timeout_sec=3600, poll_sec=10):
        self.profile_args = ["--profile", profile] if profile else []
        self.timeout_sec = timeout_sec
        self.deadline = None
        self.poll_sec = poll_sec

    def start_stage(self):
        self.deadline = time.monotonic() + self.timeout_sec

    def read(self, *args):
        result = subprocess.run(
            ["baseten", *args, *self.profile_args, "--output", "json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return json.loads(result.stdout)

    def pause(self):
        if self.deadline is None:
            raise RuntimeError("Monitoring stage has not started")
        if time.monotonic() >= self.deadline:
            raise TimeoutError(
                "Timed out monitoring; the cloud job/deployment was not stopped."
            )
        time.sleep(self.poll_sec)

    def watch(self, log_args, read_state, ready, failed):
        process = subprocess.Popen(["baseten", *log_args, *self.profile_args, "--tail"])
        previous = None
        stream_rechecked = False
        try:
            while True:
                status = read_state()
                if status != previous:
                    print(f"Status: {status}", flush=True)
                    previous = status
                if failed(status):
                    raise RuntimeError(f"Stopped with status {status}; see logs above.")
                if ready(status):
                    print(
                        "Stage completed; detaching its local log stream.", flush=True
                    )
                    return
                if process.poll() is not None:
                    # The stream can observe completion before the status API does.
                    if not stream_rechecked:
                        stream_rechecked = True
                        self.pause()
                        continue
                    raise RuntimeError(
                        f"Log stream exited before readiness (exit {process.returncode}). "
                        "Resume with --monitor-job; cloud resources were not stopped."
                    )
                self.pause()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def training(self, job_id, checkpoint_name):
        self.start_stage()

        def job_status():
            return self.read("train", "job", "describe", "--job-id", job_id)[
                "training_job"
            ]["current_status"]

        print(
            f"Following Training job {job_id} (Ctrl-C detaches without stopping it).",
            flush=True,
        )
        self.watch(
            ["train", "job", "logs", "--job-id", job_id],
            job_status,
            lambda status: status == "TRAINING_JOB_COMPLETED",
            lambda status: (
                status
                in {
                    "TRAINING_JOB_FAILED",
                    "TRAINING_JOB_STOPPED",
                    "TRAINING_JOB_CANCELLED",
                    "TRAINING_JOB_CANCELED",
                    "TRAINING_JOB_STOPPING",
                }
            ),
        )
        print(
            "Import completed. Waiting for checkpoint upload to finish...", flush=True
        )
        previous = None
        while True:
            result = self.read("train", "checkpoint", "list", "--job-id", job_id)
            checkpoint = next(
                (
                    c
                    for c in result["checkpoints"]
                    if c["checkpoint_id"] == checkpoint_name
                ),
                None,
            )
            status = (
                checkpoint.get("sync_status")
                if checkpoint
                else result["training_job"].get("checkpoint_sync_status")
            )
            if status != previous:
                print(f"Checkpoint sync: {status or 'waiting'}", flush=True)
                previous = status
            if checkpoint and status == "COMPLETE":
                print(
                    f"{checkpoint_name} is COMPLETE and ready for template selection.",
                    flush=True,
                )
                return
            if status in {"FAILED", "ERROR"}:
                raise RuntimeError(f"Checkpoint sync failed: {status}")
            self.pause()

    def deployment(self, model_id, deployment_id):
        self.start_stage()
        print(
            f"Following deployment {model_id}/{deployment_id} until ready.", flush=True
        )
        self.watch(
            [
                "model",
                "deployment",
                "logs",
                "--model-id",
                model_id,
                "--deployment-id",
                deployment_id,
            ],
            lambda: self.read(
                "model",
                "deployment",
                "describe",
                "--model-id",
                model_id,
                "--deployment-id",
                deployment_id,
            )["status"],
            lambda status: status in {"ACTIVE", "SCALED_TO_ZERO"},
            lambda status: (
                "FAILED" in status
                or status in {"INACTIVE", "DEACTIVATING", "BUILD_STOPPED"}
            ),
        )
        print(
            "Deployment is ready. Send an audio request to verify transcription.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-id")
    group.add_argument("--submission-output", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--checkpoint-name", default="checkpoint-0")
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--deployment-id")
    args = parser.parse_args()
    if bool(args.model_id) != bool(args.deployment_id):
        parser.error("--model-id and --deployment-id must be supplied together")
    job_id = args.job_id or submitted_job(args.submission_output.read_text())
    monitor = Monitor(args.profile)
    monitor.training(job_id, args.checkpoint_name)
    if args.import_only:
        return
    model_id, deployment_id = args.model_id, args.deployment_id
    if not model_id:
        print(
            "Deploy the checkpoint in the Training UI, then attach its logs below.",
            flush=True,
        )
        if not sys.stdin.isatty():
            print(
                f"Resume with --monitor-job {job_id} --model-id <id> --deployment-id <id> "
                "(and the same --profile / --checkpoint-name options).",
                flush=True,
            )
            return
        model_id = input("Model ID (Enter to finish without deployment): ").strip()
        if not model_id:
            return
        deployment_id = input("Deployment ID: ").strip()
        if not deployment_id:
            raise ValueError("A deployment ID is required to attach logs.")
    monitor.deployment(model_id, deployment_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nDetached. Cloud resources continue running; resume with --monitor-job.",
            file=sys.stderr,
        )
        sys.exit(130)
    except (
        ValueError,
        RuntimeError,
        TimeoutError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
