"""Owned cleanup and recovery tests; every provider operation is mocked."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import cleanup
import lifecycle


RUN = "owned-run"
TRAINER = "owned-trainer"
SAMPLER = "owned-sampler"
MODEL = "owned-model"
DEPLOYMENT = "owned-deployment"
OWNER = {"email": "owner@example.test"}


class Provider:
    def __init__(self, *, paired=True, orphan=False, active=True):
        self.calls = []
        self.orphan = orphan
        self.sampler = {"id": SAMPLER, "model_id": MODEL, "deployment_id": DEPLOYMENT,
                        "user": OWNER, "status": {"name": "ACTIVE"}}
        self.run = {"id": RUN, "deployment_id": TRAINER, "status": {"name": "ACTIVE"},
                    "user": OWNER, "sampler": deepcopy(self.sampler) if paired else None}
        self.trainer = {"id": TRAINER, "active_run_id": RUN, "latest_run_id": RUN,
                        "status": {"name": "RUNNING"}}
        self.model = {"id": DEPLOYMENT, "model_id": MODEL, "status": "ACTIVE",
                      "active_replica_count": 1}
        self.extra_runs, self.extra_samplers = [], []
        self.paginated = self.org_forbidden = False
        self.change_on_org_read = None
        if not active:
            self.run["status"]["name"] = "INACTIVE"
            self.trainer.update(active_run_id=None, status={"name": "STOPPED"})
            if not orphan:
                self.stop_sampler()

    def stop_sampler(self):
        self.sampler["status"]["name"] = "INACTIVE"
        self.model.update(status="INACTIVE", active_replica_count=0)

    def request(self, path, *, method="GET"):
        self.calls.append((method, path))
        if path == f"/v1/loops/runs/{RUN}/deactivate" and method == "POST":
            self.run["status"]["name"] = "INACTIVE"
            self.trainer.update(active_run_id=None, status={"name": "STOPPED"})
            if not self.orphan:
                self.stop_sampler()
            return {"id": RUN}
        if method != "GET":
            raise AssertionError("Unexpected mutation: " + path)
        if path == f"/v1/loops/runs/{RUN}":
            return {"run": deepcopy(self.run)}
        if path == f"/v1/loops/deployments/{TRAINER}":
            return {"deployment": deepcopy(self.trainer)}
        if path == f"/v1/loops/samplers/{SAMPLER}":
            return {"sampler": deepcopy(self.sampler)}
        for resource, rows in (("runs", [self.run, *self.extra_runs]),
                               ("samplers", [self.sampler, *self.extra_samplers])):
            if path == f"/v1/loops/{resource}?scope=org":
                if self.org_forbidden:
                    raise RuntimeError("secret-provider-body-must-not-be-recorded")
                result = {resource: deepcopy(rows)}
                if self.paginated:
                    result["next_page_token"] = "more-results"
                if self.change_on_org_read is not None:
                    self.change_on_org_read()
                return result
        raise AssertionError("Unexpected read: " + path)

    def model_request(self, model_id, deployment_id, *, method="GET"):
        self.calls.append((method, f"/v1/models/{model_id}/deployments/{deployment_id}"
                           + ("/deactivate" if method == "POST" else "")))
        if (model_id, deployment_id) != (MODEL, DEPLOYMENT):
            raise AssertionError("Stopped a different model deployment")
        if method == "POST":
            self.stop_sampler()
            return {"success": True, "no_op": False}
        return deepcopy(self.model)


class CleanupTests(unittest.TestCase):
    def test_pinned_management_client_uses_only_exact_deployment_routes(self):
        from baseten.client import ManagementClient

        self.assertTrue(callable(ManagementClient))
        manager = MagicMock()
        client = manager.__enter__.return_value
        client.api.get_models_deployments_deployment_id.return_value.model_dump.return_value = {"id": DEPLOYMENT}
        client.api.post_models_deployments_deactivate.return_value.model_dump.return_value = {"success": True}
        with patch("baseten.client.ManagementClient", return_value=manager), \
                patch.dict("os.environ", {"BASETEN_API_KEY": "unit-test-placeholder"}, clear=True):
            self.assertEqual(lifecycle.model_deployment_request(MODEL, DEPLOYMENT), {"id": DEPLOYMENT})
            self.assertEqual(lifecycle.model_deployment_request(MODEL, DEPLOYMENT, method="POST"), {"success": True})
        client.api.get_models_deployments_deployment_id.assert_called_once_with(
            model_id=MODEL, deployment_id=DEPLOYMENT)
        client.api.post_models_deployments_deactivate.assert_called_once_with(
            model_id=MODEL, deployment_id=DEPLOYMENT)

    def exercise(self, provider, **kwargs):
        with TemporaryDirectory(prefix="starcount-cleanup-test-") as directory, \
                patch.object(lifecycle, "api_request", side_effect=provider.request), \
                patch.object(lifecycle, "model_deployment_request", side_effect=provider.model_request):
            report = lifecycle.cleanup_owned_run(RUN, Path(directory), timeout=0.1,
                poll_interval=0.001, **kwargs)
            self.assertEqual(json.loads((Path(directory) / "deactivation.json").read_text()), report)
            return report

    @staticmethod
    def mutations(provider):
        return [path for method, path in provider.calls if method == "POST"]

    def test_normal_shutdown_verifies_inactive_and_zero_sampler_replicas(self):
        provider = Provider()
        result = self.exercise(provider)
        self.assertTrue(result["requested"])
        self.assertTrue(result["verified"])
        self.assertFalse(result["sampler_deactivation_requested"])
        self.assertEqual(result["sampler"]["active_replica_count"], 0)
        self.assertEqual(self.mutations(provider), [f"/v1/loops/runs/{RUN}/deactivate"])
        self.assertFalse(any("scope=org" in path for _, path in provider.calls))

    def test_readiness_failure_orphan_stops_only_verified_exact_sampler_deployment(self):
        provider = Provider(orphan=True)
        result = self.exercise(provider)
        self.assertTrue(result["verified"])
        self.assertTrue(result["sampler_deactivation_requested"])
        self.assertEqual(result["sampler"]["status"], "INACTIVE")
        self.assertEqual(result["sampler"]["active_replica_count"], 0)
        self.assertEqual(self.mutations(provider), [f"/v1/loops/runs/{RUN}/deactivate",
            f"/v1/models/{MODEL}/deployments/{DEPLOYMENT}/deactivate"])
        fallback = next(i for i, call in enumerate(provider.calls)
                        if call == ("POST", self.mutations(provider)[-1]))
        self.assertIn(("GET", f"/v1/loops/runs/{RUN}"), provider.calls[:fallback])
        self.assertIn(("GET", "/v1/loops/runs?scope=org"), provider.calls[:fallback])
        self.assertIn(("GET", "/v1/loops/samplers?scope=org"), provider.calls[:fallback])

    def test_already_inactive_run_with_leftover_sampler_does_not_restop_trainer(self):
        provider = Provider(orphan=True, active=False)
        result = self.exercise(provider)
        self.assertTrue(result["verified"])
        self.assertFalse(result["requested"])
        self.assertEqual(self.mutations(provider), [f"/v1/models/{MODEL}/deployments/{DEPLOYMENT}/deactivate"])

    def test_already_inactive_zero_replica_run_is_idempotent(self):
        provider = Provider(active=False)
        self.assertTrue(self.exercise(provider)["verified"])
        self.assertEqual(self.mutations(provider), [])

    def test_failed_trainer_is_terminal_but_leftover_sampler_still_requires_zero_replicas(self):
        provider = Provider(orphan=True, active=False)
        provider.trainer["status"]["name"] = "FAILED"
        result = self.exercise(provider)
        self.assertTrue(result["verified"])
        self.assertTrue(result["trainer_terminal"])
        self.assertEqual(result["trainer_status"], "FAILED")
        self.assertTrue(result["sampler_zero_replicas_verified"])
        self.assertEqual(self.mutations(provider), [f"/v1/models/{MODEL}/deployments/{DEPLOYMENT}/deactivate"])

    def test_trainer_only_or_unallocated_run_does_not_touch_models(self):
        for allocated in (True, False):
            provider = Provider(paired=False)
            if not allocated:
                provider.run["deployment_id"] = None
            with self.subTest(allocated=allocated):
                self.assertTrue(self.exercise(provider)["verified"])
                self.assertFalse(any("/models/" in path for _, path in provider.calls))

    def test_malformed_ids_are_rejected_before_any_provider_request(self):
        for value in (None, "", "..", "other/run", "x?scope=org", "x\nsecret", "https://outside"):
            with self.subTest(value=value), TemporaryDirectory() as directory, \
                    patch.object(lifecycle, "api_request") as request:
                with self.assertRaisesRegex(lifecycle.UnsafeCleanup, "invalid_resource_identifier"):
                    lifecycle.cleanup_owned_run(value, directory)
                request.assert_not_called()

    def test_trainer_reuse_refuses_all_mutations(self):
        for key in ("active_run_id", "latest_run_id"):
            provider = Provider()
            provider.trainer[key] = "unrelated-new-run"
            with self.subTest(key=key):
                report = self.exercise(provider)
                self.assertFalse(report["verified"])
                self.assertEqual(report["cleanup_error"], "trainer_reused_by_another_run")
                self.assertEqual(self.mutations(provider), [])

    def test_missing_active_run_metadata_refuses_all_mutations(self):
        provider = Provider()
        del provider.trainer["active_run_id"]
        report = self.exercise(provider)
        self.assertFalse(report["verified"])
        self.assertEqual(self.mutations(provider), [])

    def test_sampler_owner_mismatch_refuses_all_mutations(self):
        provider = Provider()
        provider.sampler["user"] = {"email": "different-owner@example.test"}
        report = self.exercise(provider)
        self.assertFalse(report["verified"])
        self.assertEqual(self.mutations(provider), [])

    def test_sampler_association_changes_before_direct_stop_fail_closed(self):
        for changed in ("run", "sampler", "model"):
            provider = Provider(orphan=True)

            def change():
                if changed == "run":
                    provider.run["sampler"]["deployment_id"] = "reassigned-deployment"
                elif changed == "sampler":
                    provider.sampler["deployment_id"] = "reassigned-deployment"
                else:
                    provider.model["model_id"] = "unrelated-model"

            provider.change_on_org_read = change
            with self.subTest(changed=changed):
                report = self.exercise(provider)
                self.assertFalse(report["verified"])
                self.assertFalse(report["sampler_deactivation_requested"])
                self.assertEqual(self.mutations(provider), [f"/v1/loops/runs/{RUN}/deactivate"])

    def test_sampler_shared_by_other_active_run_or_sampler_is_not_stopped(self):
        for resource in ("run", "sampler"):
            provider = Provider(orphan=True)
            if resource == "run":
                provider.extra_runs = [{"id": "another-run", "status": {"name": "ACTIVE"},
                                        "sampler": deepcopy(provider.sampler)}]
            else:
                provider.extra_samplers = [dict(deepcopy(provider.sampler), id="other-sampler")]
            with self.subTest(resource=resource):
                report = self.exercise(provider)
                self.assertEqual(report["cleanup_error"], "sampler_shared_with_another_resource")
                self.assertFalse(report["verified"])
                self.assertEqual(self.mutations(provider), [f"/v1/loops/runs/{RUN}/deactivate"])

    def test_incomplete_or_forbidden_visibility_cannot_authorize_sampler_stop(self):
        for problem in ("paginated", "org_forbidden"):
            provider = Provider(orphan=True)
            setattr(provider, problem, True)
            with self.subTest(problem=problem):
                report = self.exercise(provider)
                self.assertFalse(report["verified"])
                self.assertFalse(report["sampler_deactivation_requested"])
                self.assertNotIn("secret-provider-body", json.dumps(report))
                self.assertIn(f"--deployment-id {DEPLOYMENT}", report["manual_cleanup_commands"][-1])

    def test_inactive_status_with_nonzero_replicas_is_not_gpu_release(self):
        provider = Provider(active=False)
        provider.model["active_replica_count"] = 1
        with TemporaryDirectory() as directory, \
                patch.object(lifecycle, "api_request", side_effect=provider.request), \
                patch.object(lifecycle, "model_deployment_request", side_effect=provider.model_request):
            report = lifecycle.cleanup_owned_run(RUN, directory, timeout=0)
        self.assertFalse(report["verified"])
        self.assertEqual(report["cleanup_error"], "shutdown_not_verified_before_timeout")
        self.assertEqual(self.mutations(provider), [])

    def test_ambiguous_noninteger_replica_count_is_not_gpu_release(self):
        provider = Provider(active=False)
        provider.model["active_replica_count"] = False
        report = self.exercise(provider)
        self.assertFalse(report["verified"])
        self.assertEqual(report["cleanup_error"], "sampler_replica_count_not_verifiable")

    def test_cli_requires_confirmation_before_credentials_or_cleanup(self):
        with patch.object(sys, "argv", ["cleanup.py", "--output", "never-read"]), \
                patch.object(cleanup, "load_env") as load_env, \
                patch.object(cleanup, "cleanup_owned_run") as helper, \
                patch("sys.stderr", new_callable=StringIO), self.assertRaises(SystemExit):
            cleanup.main()
        load_env.assert_not_called()
        helper.assert_not_called()

    def test_cli_uses_only_selected_run_and_returns_failure_if_unverified(self):
        for verified in (True, False):
            with self.subTest(verified=verified), TemporaryDirectory() as directory:
                output = Path(directory)
                (output / "loops_run.json").write_text(json.dumps({"run_id": RUN}))
                with patch.object(sys, "argv", ["cleanup.py", "--output", directory, "--yes"]), \
                        patch.object(cleanup, "load_env"), \
                        patch.object(cleanup, "cleanup_owned_run", return_value={"verified": verified}) as helper, \
                        redirect_stdout(StringIO()):
                    if verified:
                        cleanup.main()
                    else:
                        with self.assertRaises(SystemExit) as caught:
                            cleanup.main()
                        self.assertEqual(caught.exception.code, 1)
                helper.assert_called_once_with(RUN, output)


if __name__ == "__main__":
    unittest.main()
