#!/usr/bin/env python3
"""Publish Training Jobs v1 metrics to the S3 experiment dashboard contract."""

import atexit
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse


FILES = ("metrics.jsonl", "hyperparams.json", "meta.json")


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("EXPERIMENT_S3_URI must look like s3://bucket/prefix")
    return parsed.netloc, parsed.path.strip("/")


def _is_primary_process() -> bool:
    """Return true once per distributed job, including torchrun jobs."""
    global_rank = os.environ.get("RANK")
    if global_rank:  # empty string means unset; fall through to the next check
        return global_rank == "0"
    return (
        os.environ.get("BT_NODE_RANK", "0") == "0"
        and os.environ.get("LOCAL_RANK", "0") == "0"
    )


class S3ExperimentLogger:
    """Write dashboard files locally and periodically upload them to S3.

    Training Jobs v1 injects the BT_TRAINING_* values used below. Only the
    primary process writes, so the helper is safe with multinode and torchrun
    launches when every process constructs it.
    """

    def __init__(
        self,
        hyperparameters: Mapping[str, Any],
        s3_uri: Optional[str] = None,
        local_root: Optional[str] = None,
        sync_every_seconds: float = 60,
        strict_uploads: bool = False,
        s3_client: Optional[Any] = None,
        extra_meta: Optional[Mapping[str, Any]] = None,
    ):
        if sync_every_seconds <= 0:
            raise ValueError("sync_every_seconds must be greater than 0")

        self.enabled = _is_primary_process()
        self.closed = False
        self.strict_uploads = strict_uploads
        self.sync_every_seconds = sync_every_seconds
        self._last_sync = 0.0
        if not self.enabled:
            print(
                "[s3-experiment-dashboard] logging disabled: non-primary process"
                f" (RANK={os.environ.get('RANK')!r},"
                f" BT_NODE_RANK={os.environ.get('BT_NODE_RANK')!r},"
                f" LOCAL_RANK={os.environ.get('LOCAL_RANK')!r})",
                file=sys.stderr,
            )
            return

        self.run_id = os.environ.get("BT_TRAINING_JOB_ID")
        if not self.run_id:
            raise RuntimeError(
                "BT_TRAINING_JOB_ID is missing; run this inside Training Jobs v1"
            )

        resolved_s3_uri = s3_uri or os.environ.get("EXPERIMENT_S3_URI")
        if not resolved_s3_uri:
            raise RuntimeError("Set EXPERIMENT_S3_URI to s3://bucket/prefix")
        self.bucket, self.prefix = _parse_s3_uri(resolved_s3_uri)

        root = Path(local_root or "/tmp/s3-experiment-dashboard")
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.metrics_path.touch(exist_ok=True)

        meta: Dict[str, Any] = {
            "source": "baseten-training-jobs-v1",
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "training_job_id": self.run_id,
            "training_job_name": os.environ.get("BT_TRAINING_JOB_NAME"),
            "training_project_id": os.environ.get("BT_TRAINING_PROJECT_ID"),
            "training_project_name": os.environ.get("BT_TRAINING_PROJECT_NAME"),
            "git_sha": os.environ.get("GIT_SHA") or os.environ.get("GITHUB_SHA"),
        }
        if extra_meta:
            meta.update(extra_meta)
        meta = {key: value for key, value in meta.items() if value is not None}

        self._write_json(self.run_dir / "hyperparams.json", dict(hyperparameters))
        self._write_json(self.run_dir / "meta.json", meta)
        self.s3_client = s3_client or self._new_s3_client()
        self.sync()
        atexit.register(self.close)
        self._install_sigterm_handler()
        print(
            f"[s3-experiment-dashboard] logging enabled: run_id={self.run_id}"
            f" uploading to s3://{self.bucket}/{self._object_key('')}/",
            file=sys.stderr,
        )

    def _install_sigterm_handler(self) -> None:
        """Flush on SIGTERM (scheduler preemption), then defer to any prior handler."""
        previous = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm(signum, frame):
            self.close()
            if callable(previous):
                previous(signum, frame)
            else:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        try:
            signal.signal(signal.SIGTERM, _handle_sigterm)
        except ValueError:
            pass  # not the main thread; atexit still covers normal exit

    @staticmethod
    def _new_s3_client():
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required; install training_jobs_v1/requirements.txt"
            ) from exc
        return boto3.client("s3")

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(value, indent=2, default=str) + "\n")

    def _object_key(self, filename: str) -> str:
        parts = [part for part in (self.prefix, self.run_id, filename) if part]
        return "/".join(parts)

    def log(self, step: int, **metrics: Any) -> None:
        """Append numeric training metrics and upload when the interval elapses."""
        if not self.enabled or self.closed:
            return
        record = {"step": step, "time": time.time(), **metrics}
        with self.metrics_path.open("a") as metrics_file:
            metrics_file.write(json.dumps(record, default=str) + "\n")

        if time.monotonic() - self._last_sync >= self.sync_every_seconds:
            self.sync(files=("metrics.jsonl",))

    def sync(self, files: Tuple[str, ...] = FILES) -> bool:
        """Upload the current run snapshot. Return false after a non-strict failure.

        hyperparams.json and meta.json never change after __init__, so the
        periodic sync from log() passes files=("metrics.jsonl",); __init__ and
        close() do a full pass.
        """
        if not self.enabled or self.closed:
            return True

        success = True
        for filename in files:
            path = self.run_dir / filename
            try:
                self.s3_client.upload_file(
                    str(path), self.bucket, self._object_key(filename)
                )
            except Exception as exc:
                success = False
                message = (
                    f"[s3-experiment-dashboard] upload failed for {filename}: {exc}"
                )
                if self.strict_uploads:
                    raise RuntimeError(message) from exc
                print(message, file=sys.stderr)
        self._last_sync = time.monotonic()
        return success

    def close(self) -> None:
        if not self.enabled or self.closed:
            return
        self.sync()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
