#!/usr/bin/env -S python3 -u
"""
Smoke test runner for the Baseten Training platform.

Submits training jobs using ml-cookbook examples and validates:
- Job lifecycle (submit → run → complete)
- Checkpointing (saved, retrievable via API, presigned URLs work)
- Cache (populated after job, visible via CLI)
- Checkpoint deployment (deploy → active → serves inference)

Validation steps use the truss CLI (same interface customers use) to catch
CLI-specific regressions. The CLI wraps the REST API, so both layers are tested.

Also handles teardown (per-project cleanup) and sweep (stale resource cleanup).
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

API_BASE = "https://api.baseten.co/v1"
POLL_INTERVAL = 15  # seconds between status checks
TERMINAL_STATUSES = {
    "TRAINING_JOB_COMPLETED",
    "TRAINING_JOB_DEPLOY_FAILED",
    "TRAINING_JOB_FAILED",
    "TRAINING_JOB_STOPPED",
}


def get_api_key() -> str:
    key = os.environ.get("BASETEN_API_KEY")
    if not key:
        print("ERROR: BASETEN_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)
    return key


def api_headers(api_key: str) -> dict:
    return {"Authorization": f"Api-Key {api_key}"}


def redact_url(url: str) -> str:
    """Redact query parameters from presigned URLs to avoid leaking tokens."""
    if "?" in url:
        return url.split("?")[0] + "?<REDACTED>"
    return url


# ---------------------------------------------------------------------------
# CLI Helper
# ---------------------------------------------------------------------------


def run_truss_cli(
    args: list[str], check: bool = True, timeout: int = 300
) -> subprocess.CompletedProcess:
    """Run a truss CLI command via uv and return the result.

    Uses --non-interactive to prevent prompts in CI.
    """
    cmd = ["uv", "run", "truss"] + args + ["--non-interactive"]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    if result.stdout:
        for line in result.stdout.rstrip().split("\n"):
            print(f"  {line}")
    if result.stderr:
        for line in result.stderr.rstrip().split("\n"):
            print(f"  [stderr] {line}", file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"CLI command failed (exit {result.returncode}): {' '.join(cmd)}"
        )
    return result


# ---------------------------------------------------------------------------
# Resource Tracker
# ---------------------------------------------------------------------------


@dataclass
class ResourceTracker:
    """Tracks all resources created during a smoke test for teardown."""

    api_key: str
    remote: str = "baseten"
    project_ids: list[str] = field(default_factory=list)
    jobs: list[tuple[str, str]] = field(default_factory=list)  # (project_id, job_id)
    model_ids: list[str] = field(default_factory=list)
    deployments: list[tuple[str, str]] = field(
        default_factory=list
    )  # (model_id, deployment_id)

    def track_project(self, project_id: str):
        self.project_ids.append(project_id)

    def track_job(self, project_id: str, job_id: str):
        self.jobs.append((project_id, job_id))

    def track_model(self, model_id: str):
        self.model_ids.append(model_id)

    def track_deployment(self, model_id: str, deployment_id: str):
        self.deployments.append((model_id, deployment_id))

    def teardown_all(self):
        """Tear down all tracked resources in reverse creation order."""
        headers = api_headers(self.api_key)

        for model_id, deployment_id in reversed(self.deployments):
            try:
                print(f"  Deactivating deployment {deployment_id}...")
                requests.post(
                    f"{API_BASE}/models/{model_id}/deployments/{deployment_id}/deactivate",
                    headers=headers,
                )
            except Exception as e:
                print(f"  WARNING: Failed to deactivate deployment: {e}")

        for model_id in reversed(self.model_ids):
            try:
                print(f"  Deleting model {model_id}...")
                requests.delete(f"{API_BASE}/models/{model_id}", headers=headers)
            except Exception as e:
                print(f"  WARNING: Failed to delete model: {e}")

        for project_id, job_id in reversed(self.jobs):
            try:
                run_truss_cli(
                    ["train", "stop", "--job-id", job_id, "--remote", self.remote],
                    check=False,
                )
            except Exception as e:
                print(f"  WARNING: Failed to stop job {job_id}: {e}")


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


def load_training_project(config_path: Path):
    """Import a config.py and find the TrainingProject instance."""
    from truss_train import definitions

    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, definitions.TrainingProject):
            return attr

    print(f"ERROR: No TrainingProject found in {config_path}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Job Submission (SDK — need structured IDs back)
# ---------------------------------------------------------------------------


def submit_job(config_path: Path, project_name: str, remote: str = "baseten") -> dict:
    """Submit a training job with a custom project name."""
    from truss_train import definitions
    from truss_train.public_api import push

    project = load_training_project(config_path)

    # Override the project name with our smoke-test-prefixed name
    custom_project = definitions.TrainingProject(
        name=project_name, job=project.job
    )

    source_dir = config_path.parent
    print(f"Submitting job for project '{project_name}' from {source_dir} (remote={remote})...")
    result = push(config=custom_project, source_dir=source_dir, remote=remote)

    # push() returns a flat dict with job fields at the top level
    # and training_project nested inside.
    job_id = result["id"]
    project_id = result["training_project"]["id"]
    print(f"  Job submitted: job_id={job_id}, project_id={project_id}")

    # Normalize into the shape the rest of the script expects
    return {
        "training_job": result,
        "training_project": result["training_project"],
    }


# ---------------------------------------------------------------------------
# Status Polling (REST — need structured status for loop control)
# ---------------------------------------------------------------------------


def poll_job_status(
    project_id: str, job_id: str, api_key: str, remote: str, timeout: int
) -> str:
    """Poll job status until terminal or timeout. Returns final status."""
    headers = api_headers(api_key)
    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            print(f"\n  TIMEOUT after {int(elapsed)}s — job still not in terminal state.")
            # Dump logs via CLI for debugging
            print("\n  Fetching logs via CLI...")
            run_truss_cli(
                ["train", "logs", "--job-id", job_id, "--remote", remote],
                check=False,
            )
            # Stop the timed-out job via CLI
            run_truss_cli(
                ["train", "stop", "--job-id", job_id, "--remote", remote],
                check=False,
            )
            return "TIMEOUT"

        resp = requests.get(
            f"{API_BASE}/training_projects/{project_id}/jobs/{job_id}",
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data["training_job"]["current_status"]
        print(f"  [{int(elapsed)}s] Status: {status}")

        if status in TERMINAL_STATUSES:
            if status != "TRAINING_JOB_COMPLETED":
                error_msg = data["training_job"].get("error_message", "")
                if error_msg:
                    print(f"  Error: {error_msg}")
                # Dump logs via CLI for debugging
                print("\n  Fetching logs via CLI...")
                run_truss_cli(
                    ["train", "logs", "--job-id", job_id, "--remote", remote],
                    check=False,
                )
            return status

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# CLI Validation: View
# ---------------------------------------------------------------------------


def check_view(job_id: str, remote: str):
    """Validate that 'truss train view --job-id' works."""
    print("\nValidating 'truss train view'...")
    run_truss_cli(["train", "view", "--job-id", job_id, "--remote", remote])
    print("  View validation PASSED")


# ---------------------------------------------------------------------------
# CLI Validation: Checkpoints
# ---------------------------------------------------------------------------


def check_checkpoints(project_id: str, job_id: str, api_key: str, remote: str):
    """Validate checkpoints exist and are accessible.

    Uses REST API for checkpoint metadata validation (sizes, presigned URLs)
    since the CLI doesn't expose this data in a machine-readable format.
    Also runs 'truss train view --job-id' which displays checkpoints.
    """
    headers = api_headers(api_key)
    print("\nValidating checkpoints...")

    # List checkpoints via API
    resp = requests.get(
        f"{API_BASE}/training_projects/{project_id}/jobs/{job_id}/checkpoints",
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()
    checkpoints = data.get("checkpoints", [])

    assert len(checkpoints) > 0, "No checkpoints found"
    print(f"  Found {len(checkpoints)} checkpoint(s)")

    for cp in checkpoints:
        cp_id = cp.get("checkpoint_id", "unknown")
        size = cp.get("size_bytes", 0)
        print(f"    checkpoint_id={cp_id}  size_bytes={size}")
        assert size > 0, f"Checkpoint {cp_id} has size_bytes=0"

    # Validate checkpoint files are accessible via presigned URLs
    resp = requests.get(
        f"{API_BASE}/training_projects/{project_id}/jobs/{job_id}/checkpoint_files",
        headers=headers,
    )
    resp.raise_for_status()
    files_data = resp.json()
    presigned_urls = files_data.get("presigned_urls", [])

    assert len(presigned_urls) > 0, "No checkpoint files returned"
    print(f"  Found {files_data.get('total_count', len(presigned_urls))} checkpoint file(s)")

    # Spot-check first 3 URLs are accessible (use GET with Range header
    # since some S3 presigned URLs don't support HEAD)
    for file_info in presigned_urls[:3]:
        url = file_info["url"]
        filename = file_info.get("relative_file_name", "unknown")
        size = file_info.get("size_bytes", "?")
        check_resp = requests.get(
            url, headers={"Range": "bytes=0-0"}, timeout=10
        )
        assert check_resp.status_code in (200, 206), (
            f"Checkpoint file {filename} not accessible (HTTP {check_resp.status_code})"
        )
        print(f"    OK: {filename} ({size} bytes)")

    # Also verify 'truss train view' shows checkpoints
    print("\n  Verifying checkpoints visible in CLI view...")
    run_truss_cli(["train", "view", "--job-id", job_id, "--remote", remote])

    print("  Checkpoint validation PASSED")


# ---------------------------------------------------------------------------
# CLI Validation: Cache
# ---------------------------------------------------------------------------


def check_cache(project_name: str, remote: str):
    """Validate cache is populated using the truss CLI.

    Runs 'truss train cache summarize' with JSON output — the same command
    customers use to inspect their project cache.
    """
    print("\nValidating cache via CLI...")

    result = run_truss_cli([
        "train", "cache", "summarize", project_name,
        "--remote", remote,
        "-o", "json",
    ])

    # Parse JSON output to validate content.
    # CLI JSON uses "files"; REST API uses "file_summaries".
    data = json.loads(result.stdout)
    files = data.get("files", [])
    total_files = data.get("total_files", len(files))
    total_bytes = data.get("total_size_bytes", sum(f.get("size_bytes", 0) for f in files))

    assert total_files > 0, "Cache summary returned but has no files"

    print(f"  Cache populated: {total_files} file(s), {total_bytes} bytes total")
    print("  Cache validation PASSED")


# ---------------------------------------------------------------------------
# Deploy and Infer
# ---------------------------------------------------------------------------


def deploy_and_infer(
    project_id: str,
    job_id: str,
    project_name: str,
    api_key: str,
    tracker: ResourceTracker,
    timeout: int = 1200,
    remote: str = "baseten",
):
    """Deploy a checkpoint and validate inference works."""
    from truss.cli.train.deploy_checkpoints import (
        create_model_version_from_inference_template,
    )
    from truss.remote.remote_factory import RemoteFactory

    headers = api_headers(api_key)
    print("\nDeploying checkpoint...")

    remote_provider = RemoteFactory.create(remote=remote)

    # Fetch checkpoints and pick the latest numbered one
    resp = requests.get(
        f"{API_BASE}/training_projects/{project_id}/jobs/{job_id}/checkpoints",
        headers=headers,
    )
    resp.raise_for_status()
    checkpoints = resp.json().get("checkpoints", [])
    if not checkpoints:
        raise RuntimeError("No checkpoints found for deployment")

    # Prefer a numbered checkpoint (e.g. checkpoint-20) over "."
    numbered = [c for c in checkpoints if c["checkpoint_id"].startswith("checkpoint-")]
    target_cp = numbered[-1] if numbered else checkpoints[-1]
    checkpoint_id = target_cp["checkpoint_id"]
    checkpoint_type = target_cp.get("checkpoint_type", "lora")
    base_model_id = target_cp.get("base_model")
    print(f"  Deploying checkpoint: {checkpoint_id} (type={checkpoint_type})")

    # Build a DeployCheckpointsConfig programmatically, providing all fields
    # so that _hydrate_deploy_config doesn't trigger interactive prompts.
    from truss.base import truss_config as tc
    from truss.cli.train.deploy_checkpoints.deploy_checkpoints import hydrate_checkpoint
    from truss_train.definitions import (
        CheckpointList,
        Compute,
        DeployCheckpointsConfig,
        DeployCheckpointsRuntime,
        SecretReference,
    )

    cp_obj = hydrate_checkpoint(job_id, checkpoint_id, target_cp, checkpoint_type)
    cp_list = CheckpointList(
        checkpoints=[cp_obj],
        base_model_id=base_model_id,
    )

    model_name = f"smoke-test-{job_id}"
    deploy_config = DeployCheckpointsConfig(
        checkpoint_details=cp_list,
        model_name=model_name,
        compute=Compute(
            node_count=1,
            accelerator=tc.AcceleratorSpec(
                accelerator=tc.Accelerator.H100,
                count=1,
            ),
        ),
        runtime=DeployCheckpointsRuntime(
            environment_variables={
                "HF_TOKEN": SecretReference(name="hf_access_token"),
            }
        ),
    )

    result = create_model_version_from_inference_template(
        remote_provider,
        deploy_config,
        project_id=project_id,
        job_id=job_id,
        dry_run=False,
    )

    model_version = result.model_version
    if not model_version:
        raise RuntimeError("deploy_checkpoints returned no model version")
    deployment_id = model_version.id
    print(f"  Model version created: name={model_version.name}, id={deployment_id}")

    # Find the model by listing all models and matching the name
    print("  Looking up deployed model...")
    resp = requests.get(f"{API_BASE}/models", headers=headers)
    resp.raise_for_status()
    models = resp.json().get("models", [])

    target_model = None
    for model in models:
        if model.get("name", "") == model_name:
            target_model = model
            break

    if not target_model:
        raise RuntimeError("Could not find deployed model after deploy_checkpoints")

    model_id = target_model["id"]
    tracker.track_model(model_id)
    tracker.track_deployment(model_id, deployment_id)
    print(f"  Model found: id={model_id}, name={target_model.get('name', '?')}")

    # Wait for the deployment to become ACTIVE
    print("  Waiting for deployment to become ACTIVE...")
    deploy_start = time.monotonic()

    while True:
        elapsed = time.monotonic() - deploy_start
        if elapsed > timeout:
            raise RuntimeError(f"Deployment did not become ACTIVE within {int(timeout)}s")

        resp = requests.get(
            f"{API_BASE}/models/{model_id}/deployments/{deployment_id}",
            headers=headers,
        )
        resp.raise_for_status()
        dep_data = resp.json().get("deployment", resp.json())
        dep_status = dep_data.get("status", "UNKNOWN")
        print(f"  [{int(elapsed)}s] Deployment status: {dep_status}")

        if dep_status == "ACTIVE":
            # Run inference using OpenAI-compatible chat format
            print("  Sending inference request...")
            predict_url = (
                f"https://model-{model_id}.api.baseten.co/production/predict"
            )
            infer_resp = requests.post(
                predict_url,
                headers=headers,
                json={
                    "model": checkpoint_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
                timeout=60,
            )

            assert infer_resp.status_code == 200, (
                f"Inference failed: HTTP {infer_resp.status_code} — "
                f"{infer_resp.text[:200]}"
            )
            print(f"  Inference response (truncated): {infer_resp.text[:200]}")
            print("  Deploy and infer validation PASSED")
            return

        if dep_status in ("FAILED", "BUILD_FAILED"):
            raise RuntimeError(f"Deployment failed with status: {dep_status}")

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def teardown_project(project_name: str, api_key: str, remote: str):
    """Stop all running jobs in a project via the CLI."""
    print(f"Tearing down project '{project_name}'...")
    run_truss_cli(
        ["train", "stop", "--project", project_name, "--all", "--remote", remote],
        check=False,
    )
    print(f"  Teardown complete for '{project_name}'")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep_stale_projects(prefix: str, older_than_hours: float, api_key: str):
    """Find and clean up stale smoke test projects."""
    headers = api_headers(api_key)
    print(f"Sweeping projects matching '{prefix}*' older than {older_than_hours}h...")

    resp = requests.get(f"{API_BASE}/training_projects", headers=headers)
    resp.raise_for_status()
    projects = resp.json().get("training_projects", [])

    now = datetime.now(timezone.utc)
    cleaned = 0

    for p in projects:
        if not p["name"].startswith(prefix):
            continue

        created_at = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
        age_hours = (now - created_at).total_seconds() / 3600

        if age_hours < older_than_hours:
            print(f"  Skipping '{p['name']}' (age: {age_hours:.1f}h — too recent)")
            continue

        project_id = p["id"]

        try:
            jobs_resp = requests.get(
                f"{API_BASE}/training_projects/{project_id}/jobs",
                headers=headers,
            )
            jobs_resp.raise_for_status()
            jobs = jobs_resp.json().get("training_jobs", [])
            active_jobs = [
                j for j in jobs if j.get("current_status") not in TERMINAL_STATUSES
            ]

            if not active_jobs:
                print(f"  Skipping '{p['name']}' (age: {age_hours:.1f}h — all jobs already terminal)")
                continue

            print(f"  Cleaning up '{p['name']}' (age: {age_hours:.1f}h, {len(active_jobs)} active job(s))...")
            for job in active_jobs:
                requests.post(
                    f"{API_BASE}/training_projects/{project_id}/jobs/{job['id']}/stop",
                    headers=headers,
                    json={},
                )
                print(f"    Stopped job {job['id']}")
        except Exception as e:
            print(f"    WARNING: Failed to clean up jobs for '{p['name']}': {e}")
            continue

        cleaned += 1

    print(f"Sweep complete: stopped jobs in {cleaned} project(s)")
    print("  Note: training projects cannot be deleted via API (dashboard only)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Baseten Training Platform Smoke Test Runner"
    )

    # Positional: config path (required for smoke tests, not for teardown/sweep)
    parser.add_argument("config", nargs="?", help="Path to training config.py")

    # Project naming
    parser.add_argument("--project", help="Project name (e.g. smoke-test-lifecycle-123)")

    # Smoke test options
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Per-job timeout in seconds (default: 1200 = 20 min)",
    )
    parser.add_argument(
        "--assert-status",
        default="TRAINING_JOB_COMPLETED",
        help="Expected terminal status (default: TRAINING_JOB_COMPLETED)",
    )
    parser.add_argument(
        "--check-checkpoints",
        action="store_true",
        help="Validate checkpoints after job completes",
    )
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="Validate cache after job completes",
    )
    parser.add_argument(
        "--deploy-and-infer",
        action="store_true",
        help="Deploy checkpoint and run inference",
    )

    # Teardown / sweep modes
    parser.add_argument(
        "--teardown",
        action="store_true",
        help="Teardown mode: stop all jobs in --project",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep mode: clean up stale smoke-test projects",
    )
    parser.add_argument(
        "--prefix",
        default="smoke-test-",
        help="Project name prefix for sweep (default: smoke-test-)",
    )
    parser.add_argument(
        "--older-than",
        default="2h",
        help="Age threshold for sweep (default: 2h)",
    )
    parser.add_argument(
        "--remote",
        default="baseten",
        help="Truss remote name from .trussrc (default: baseten)",
    )

    args = parser.parse_args()
    api_key = get_api_key()

    # --- Sweep mode ---
    if args.sweep:
        hours = float(args.older_than.rstrip("h"))
        sweep_stale_projects(args.prefix, hours, api_key)
        return

    # --- Teardown mode ---
    if args.teardown:
        if not args.project:
            print("ERROR: --project is required with --teardown", file=sys.stderr)
            sys.exit(1)
        teardown_project(args.project, api_key, args.remote)
        return

    # --- Smoke test mode ---
    if not args.config:
        parser.print_help()
        print("\nERROR: config path is required for smoke test mode", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    project_name = args.project or f"smoke-test-{int(time.time())}"
    tracker = ResourceTracker(api_key=api_key, remote=args.remote)

    try:
        # 1. Submit job (SDK — need structured IDs)
        result = submit_job(config_path, project_name, remote=args.remote)
        project_id = result["training_project"]["id"]
        job_id = result["training_job"]["id"]
        tracker.track_project(project_id)
        tracker.track_job(project_id, job_id)

        # 2. Poll until completion or timeout (REST — need structured status)
        print(f"\nPolling job {job_id} (timeout: {args.timeout}s)...")
        final_status = poll_job_status(
            project_id, job_id, api_key, args.remote, args.timeout
        )

        if final_status == "TIMEOUT":
            print(f"\nFAIL: Job did not complete within {args.timeout}s")
            sys.exit(1)

        if final_status != args.assert_status:
            print(f"\nFAIL: Expected {args.assert_status}, got {final_status}")
            sys.exit(1)

        print(f"\nJob completed: {final_status}")

        # 3. View validation (CLI)
        check_view(job_id, args.remote)

        # 4. Checkpoint validation (optional)
        if args.check_checkpoints:
            check_checkpoints(project_id, job_id, api_key, args.remote)

        # 5. Cache validation (optional, CLI)
        if args.check_cache:
            check_cache(project_name, args.remote)

        # 6. Deploy and infer (optional)
        if args.deploy_and_infer:
            deploy_and_infer(
                project_id, job_id, project_name, api_key, tracker, args.timeout,
                remote=args.remote,
            )

        print("\n=== PASS: All smoke test checks passed ===")

    except Exception as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    finally:
        print("\nCleaning up tracked resources...")
        tracker.teardown_all()
        print("Done.")


if __name__ == "__main__":
    main()
