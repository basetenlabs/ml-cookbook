"""Bounded, fail-closed cleanup of this controller's exact Loops resources."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time

from common import API_BASE_URL, api_request


IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
TRAINER_STOPPED = frozenset({"STOPPED", "SCALED_TO_ZERO", "PREEMPTED", "FAILED"})


class UnsafeCleanup(RuntimeError):
    """Association or provider visibility cannot establish exclusive ownership."""


def _identifier(value):
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise UnsafeCleanup("invalid_resource_identifier")
    return value


def _object(response, key):
    value = response.get(key, response) if isinstance(response, dict) else None
    if not isinstance(value, dict):
        raise UnsafeCleanup("invalid_provider_metadata")
    return value


def _status(value):
    value = value.get("name") if isinstance(value, dict) else value
    if not isinstance(value, str) or re.fullmatch(r"[A-Z_]{1,40}", value) is None:
        raise UnsafeCleanup("unknown_resource_status")
    return value


def _owner(value):
    user = value.get("user")
    return user.get("email") if isinstance(user, dict) else None


def _sampler_identity(sampler):
    if sampler is None:
        return None
    if not isinstance(sampler, dict):
        raise UnsafeCleanup("invalid_sampler_association")
    return tuple(_identifier(sampler.get(key)) for key in ("id", "model_id", "deployment_id"))


def _write_report(output, report):
    path = Path(output) / "deactivation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_deployment_request(model_id, deployment_id, *, method="GET"):
    """Use the pinned SDK's exact deployment route, never a model-wide stop."""
    from baseten.client import ManagementClient

    _identifier(model_id)
    _identifier(deployment_id)
    key = os.environ.get("BASETEN_API_KEY")
    if not key or "\n" in key or "\r" in key:
        raise ValueError("Set a valid BASETEN_API_KEY for cleanup.")
    with ManagementClient(api_key=key, base_url_override=API_BASE_URL) as client:
        if method == "GET":
            result = client.api.get_models_deployments_deployment_id(
                model_id=model_id, deployment_id=deployment_id)
        elif method == "POST":
            result = client.api.post_models_deployments_deactivate(
                model_id=model_id, deployment_id=deployment_id)
        else:
            raise ValueError("Cleanup supports only GET and POST.")
        return result.model_dump(mode="json")


def _exclusive_sampler(run_id, sampler_identity, owner):
    """An org-complete read is required only for the exceptional direct stop.

    A denied read or future paginated response is deliberately not treated as
    proof that nobody else uses this deployment. No provider body is persisted.
    """
    if not isinstance(owner, str) or not owner:
        raise UnsafeCleanup("sampler_owner_not_verifiable")
    for resource in ("runs", "samplers"):
        response = api_request(f"/v1/loops/{resource}?scope=org")
        rows = response.get(resource)
        if not isinstance(rows, list) or any(response.get(key) for key in (
                "next_page_token", "next_page", "next_cursor", "has_more")):
            raise UnsafeCleanup("complete_shared_usage_visibility_required")
        own_sampler_seen = False
        for row in rows:
            if not isinstance(row, dict):
                raise UnsafeCleanup("invalid_shared_usage_metadata")
            if resource == "runs":
                if row.get("id") == run_id or _status(row.get("status")) == "INACTIVE":
                    continue
                candidate = _sampler_identity(row.get("sampler"))
            else:
                candidate = _sampler_identity(row)
                if candidate == sampler_identity:
                    own_sampler_seen = True
                    if _owner(row) != owner:
                        raise UnsafeCleanup("sampler_owner_changed")
                if _status(row.get("status")) == "INACTIVE":
                    continue
            if candidate is not None and (candidate[0] == sampler_identity[0]
                    or candidate[1:] == sampler_identity[1:]):
                if resource == "runs" or candidate != sampler_identity:
                    raise UnsafeCleanup("sampler_shared_with_another_resource")
        if resource == "samplers" and not own_sampler_seen:
            raise UnsafeCleanup("complete_shared_usage_visibility_required")


def cleanup_owned_run(run_id, output, *, timeout=120.0, poll_interval=2.0):
    """Request owned shutdown, verify it, and safely handle an orphan sampler.

    ``run_id`` must come from this controller's creation record, not a reused
    session. Return a non-secret report; ``requested`` is never a synonym for
    ``verified``. Ambiguous associations stop cleanup without broadening scope.
    """
    _identifier(run_id)
    if (not math.isfinite(timeout) or timeout < 0 or not math.isfinite(poll_interval)
            or poll_interval <= 0):
        raise ValueError("Cleanup timeout must be nonnegative and polling interval positive.")
    report = {"run_id": run_id, "requested": False, "verified": False,
              "sampler_deactivation_requested": False}
    deadline = time.monotonic() + timeout
    identity = deployment_id = owner = None
    try:
        initial = _object(api_request(f"/v1/loops/runs/{run_id}"), "run")
        if initial.get("id") != run_id:
            raise UnsafeCleanup("run_identity_mismatch")
        initial_status = _status(initial.get("status"))
        if initial_status not in {"ACTIVE", "INACTIVE"}:
            raise UnsafeCleanup("unknown_run_status")
        deployment_id = initial.get("deployment_id")
        if deployment_id is not None:
            _identifier(deployment_id)
        identity = _sampler_identity(initial.get("sampler"))
        owner = _owner(initial)

        def state():
            run = _object(api_request(f"/v1/loops/runs/{run_id}"), "run")
            if run.get("id") != run_id or run.get("deployment_id") != deployment_id:
                raise UnsafeCleanup("run_association_changed")
            if _sampler_identity(run.get("sampler")) != identity or _owner(run) != owner:
                raise UnsafeCleanup("run_association_changed")
            run_status = _status(run.get("status"))
            if run_status not in {"ACTIVE", "INACTIVE"}:
                raise UnsafeCleanup("unknown_run_status")
            trainer_status = None
            if deployment_id is not None:
                trainer = _object(api_request(f"/v1/loops/deployments/{deployment_id}"), "deployment")
                if trainer.get("id") != deployment_id:
                    raise UnsafeCleanup("trainer_identity_mismatch")
                if ("active_run_id" not in trainer or trainer["active_run_id"] not in (None, run_id)
                        or trainer.get("latest_run_id") not in (None, run_id)):
                    raise UnsafeCleanup("trainer_reused_by_another_run")
                if run_status == "ACTIVE" and trainer["active_run_id"] != run_id:
                    raise UnsafeCleanup("trainer_active_run_not_verifiable")
                trainer_status = _status(trainer.get("status"))
            report.update(run_status=run_status, trainer_status=trainer_status,
                trainer_terminal=deployment_id is None or trainer_status in TRAINER_STOPPED)
            sampler_status = model_status = replica_count = None
            if identity is not None:
                sampler_id, model_id, model_deployment_id = identity
                sampler = _object(api_request(f"/v1/loops/samplers/{sampler_id}"), "sampler")
                if _sampler_identity(sampler) != identity or _owner(sampler) != owner or not owner:
                    raise UnsafeCleanup("sampler_association_or_owner_changed")
                model = _object(model_deployment_request(model_id, model_deployment_id), "deployment")
                if model.get("id") != model_deployment_id or model.get("model_id") != model_id:
                    raise UnsafeCleanup("model_deployment_identity_mismatch")
                replica_count = model.get("active_replica_count")
                if type(replica_count) is not int or replica_count < 0:
                    raise UnsafeCleanup("sampler_replica_count_not_verifiable")
                sampler_status, model_status = _status(sampler.get("status")), _status(model.get("status"))
                report["sampler"] = {"id": sampler_id, "model_id": model_id,
                    "deployment_id": model_deployment_id, "status": sampler_status,
                    "deployment_status": model_status, "active_replica_count": replica_count}
            return run_status, trainer_status, sampler_status, model_status, replica_count

        current = state()  # Re-check associations before the first mutation.
        if current[0] == "ACTIVE":
            report["requested"] = True
            _write_report(output, report)
            api_request(f"/v1/loops/runs/{run_id}/deactivate", method="POST")
        while True:
            current = state()
            run_status, trainer_status, sampler_status, model_status, replica_count = current
            trainer_stopped = deployment_id is None or trainer_status in TRAINER_STOPPED
            sampler_stopped = identity is None or (
                sampler_status == model_status == "INACTIVE" and replica_count == 0)
            report["sampler_zero_replicas_verified"] = identity is not None and sampler_stopped
            if run_status == "INACTIVE" and trainer_stopped and sampler_stopped:
                report["verified"] = True
                break
            if (run_status == "INACTIVE" and identity is not None
                    and not report["sampler_deactivation_requested"]
                    and sampler_status != "INACTIVE" and model_status not in {"INACTIVE", "DEACTIVATING"}):
                _exclusive_sampler(run_id, identity, owner)
                fresh = state()  # Never stop a stale or reassigned deployment.
                if fresh != current:
                    raise UnsafeCleanup("resource_status_changed_before_sampler_stop")
                report["sampler_deactivation_requested"] = True
                _write_report(output, report)
                response = model_deployment_request(identity[1], identity[2], method="POST")
                if response.get("success") is not True:
                    raise UnsafeCleanup("sampler_deactivation_not_acknowledged")
            if time.monotonic() >= deadline:
                report["cleanup_error"] = "shutdown_not_verified_before_timeout"
                break
            _write_report(output, report)
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    except UnsafeCleanup as error:
        report["cleanup_error"] = str(error)  # Static codes only, never provider text.
    except Exception as error:
        report["cleanup_error"] = "provider_cleanup_request_failed"
        report["exception_type"] = type(error).__name__
    if not report["verified"]:
        report["manual_cleanup_commands"] = [f"baseten loops run deactivate --run-id {run_id} --yes"]
        if identity is not None:
            report["manual_cleanup_commands"].append(
                "baseten model deployment deactivate --model-id " + identity[1]
                + " --deployment-id " + identity[2] + " --yes")
        report["manual_cleanup_warning"] = "Re-verify exclusive ownership and the current association before manual cleanup."
    _write_report(output, report)
    return report
