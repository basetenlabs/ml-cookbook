"""Offline checks for credential handling, pinned downloads, and checkpoint publication."""

import importlib.util
import json
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "import_checkpoint", ROOT / "import_checkpoint.py"
)
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)
MONITOR_SPEC = importlib.util.spec_from_file_location("monitor", ROOT / "monitor.py")
monitoring = importlib.util.module_from_spec(MONITOR_SPEC)
MONITOR_SPEC.loader.exec_module(monitoring)
COMMIT = "a" * 40


def model_config(family="large-v3"):
    return {**importer.COMMON_SHAPE, **importer.MODEL_SHAPES[family]}


class ImportTest(unittest.TestCase):
    def test_names_default_to_model_and_commit_and_allow_overrides(self):
        with patch.dict(
            os.environ,
            {"SOURCE_REPO": "example/private-whisper", "SOURCE_REVISION": COMMIT},
            clear=True,
        ):
            project = runpy.run_path(str(ROOT / "config.py"))["training_project"]
        self.assertEqual(project.name, "example/private-whisper")
        self.assertEqual(project.job.name, "example-private-whisper-aaaaaa")
        with (
            patch.dict(
                os.environ,
                {"SOURCE_REPO": "example/private-whisper", "SOURCE_REVISION": "main"},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "full HF commit SHA"),
        ):
            runpy.run_path(str(ROOT / "config.py"))
        with patch.dict(
            os.environ,
            {
                "SOURCE_REPO": "example/private-whisper",
                "SOURCE_REVISION": "main",
                "TRAINING_PROJECT_NAME": "custom-project",
                "TRAINING_JOB_NAME": "custom-job",
            },
            clear=True,
        ):
            project = runpy.run_path(str(ROOT / "config.py"))["training_project"]
        self.assertEqual(project.name, "custom-project")
        self.assertEqual(project.job.name, "custom-job")

    def test_config_uses_workspace_secret_without_forwarding_local_token(self):
        with patch.dict(
            os.environ,
            {"HF_SECRET_NAME": "existing-secret", "HF_TOKEN": "local-only"},
            clear=True,
        ):
            job = runpy.run_path(str(ROOT / "config.py"))["training_project"].job
        self.assertEqual(
            job.runtime.environment_variables["HF_TOKEN"].name, "existing-secret"
        )
        self.assertNotIn("local-only", repr(job))
        self.assertIsNone(job.compute.accelerator)
        self.assertEqual(job.weights, [])
        with patch.dict(os.environ, {"HF_TOKEN": "local-only"}, clear=True):
            job = runpy.run_path(str(ROOT / "config.py"))["training_project"].job
        self.assertNotIn("HF_TOKEN", job.runtime.environment_variables)

    def download(self, source, files, token=None):
        def download_file(repo, filename, **kwargs):
            self.assertEqual(kwargs["revision"], COMMIT)
            self.assertEqual(kwargs["token"], token or False)
            (source / filename).write_bytes(files[filename])

        def download_snapshot(repo, **kwargs):
            self.assertEqual(kwargs["revision"], COMMIT)
            self.assertEqual(kwargs["token"], token or False)
            for filename in kwargs["allow_patterns"]:
                if filename in files:
                    (source / filename).write_bytes(files[filename])

        env = {"SOURCE_REPO": "example/private-whisper", "SOURCE_REVISION": "release"}
        if token:
            env["HF_TOKEN"] = token
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(importer, "HfApi") as api,
            patch.object(importer, "hf_hub_download", side_effect=download_file),
            patch.object(
                importer, "snapshot_download", side_effect=download_snapshot
            ) as snapshot,
        ):
            api.return_value.model_info.return_value = SimpleNamespace(
                sha=COMMIT, siblings=[SimpleNamespace(rfilename=f) for f in files]
            )
            commit = importer.download_checkpoint(source)
            api.return_value.model_info.assert_called_once_with(
                "example/private-whisper", revision="release", token=token or False
            )
            return commit, snapshot

    def test_private_sharded_download_pins_all_files_and_omits_other_weights(self):
        files = {
            "config.json": json.dumps(model_config()).encode(),
            "model.safetensors.index.json": json.dumps(
                {"weight_map": {"encoder.weight": "custom-shard.safetensors"}}
            ).encode(),
            "custom-shard.safetensors": b"fixture weights",
            "tokenizer.json": b"{}",
            "model.fp32.safetensors": b"unselected weights",
            "pytorch_model.bin": b"unselected weights",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            commit, _ = self.download(source, files, token="test-token")
            self.assertEqual(commit, COMMIT)
            self.assertEqual(
                (source / "custom-shard.safetensors").read_bytes(), b"fixture weights"
            )
            self.assertFalse((source / "model.fp32.safetensors").exists())
            self.assertFalse((source / "pytorch_model.bin").exists())

    def test_rejects_missing_or_traversing_shard_before_downloading_weights(self):
        for shard in ("missing.safetensors", "../outside.safetensors"):
            with self.subTest(shard=shard), tempfile.TemporaryDirectory() as directory:
                files = {
                    "config.json": json.dumps(model_config()).encode(),
                    "model.safetensors.index.json": json.dumps(
                        {"weight_map": {"weight": shard}}
                    ).encode(),
                }
                with self.assertRaisesRegex(ValueError, "missing or invalid shards"):
                    self.download(Path(directory), files)

    def test_supported_families_keep_weights_and_record_provenance(self):
        for family in importer.MODEL_SHAPES:
            with (
                self.subTest(family=family),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                original = {
                    **model_config(family),
                    "_name_or_path": "example/private-whisper",
                }
                (source / "config.json").write_text(json.dumps(original))
                (source / "model.safetensors").write_bytes(b"fixture weights")
                (source / "tokenizer.json").write_text("{}")
                with patch.dict(
                    os.environ,
                    {
                        "BT_CHECKPOINT_DIR": str(root / "checkpoints"),
                        "CHECKPOINT_NAME": "checkpoint-0",
                        "SOURCE_REPO": "example/private-whisper",
                    },
                ):
                    importer.save_checkpoint(source, COMMIT)
                    with self.assertRaises(FileExistsError):
                        importer.save_checkpoint(source, COMMIT)
                checkpoint = root / "checkpoints/checkpoint-0"
                self.assertEqual(
                    (checkpoint / "model.safetensors").read_bytes(), b"fixture weights"
                )
                self.assertEqual(
                    json.loads((checkpoint / "config.json").read_text())[
                        "_name_or_path"
                    ],
                    f"openai/whisper-{family}",
                )
                provenance = json.loads(
                    (checkpoint / "import_provenance.json").read_text()
                )
                self.assertEqual(provenance["source_revision"], COMMIT)
                self.assertEqual(provenance["optimizer_steps"], 0)
                self.assertEqual(
                    json.loads((source / "config.json").read_text()), original
                )

    def test_invalid_checkpoint_does_not_publish_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(model_config()))
            (root / "model.safetensors").write_bytes(b"fixture weights")
            with (
                patch.dict(
                    os.environ,
                    {
                        "BT_CHECKPOINT_DIR": str(root / "output"),
                        "CHECKPOINT_NAME": "checkpoint-0",
                    },
                ),
                self.assertRaisesRegex(ValueError, "tokenizer is incomplete"),
            ):
                importer.save_checkpoint(root, COMMIT)
            self.assertFalse((root / "output/checkpoint-0/config.json").exists())
        for changed in ({"decoder_layers": 2}, {"vocab_size": 60000}):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "Unsupported"),
            ):
                importer.checkpoint_model({**model_config(), **changed})

    def test_wrapper_passes_options_as_literal_arguments_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "baseten"
            stub.write_text(
                '#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\nprintf "%s" "$SOURCE_REPO" > "$CAPTURE_REPO"\nexit 23\n'
            )
            stub.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "CAPTURE_ARGS": str(root / "args"),
                "CAPTURE_REPO": str(root / "repo"),
            }
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "import_whisper_checkpoint.sh"),
                    "--repo",
                    "example/model",
                    "--team",
                    "team with spaces",
                    "--profile",
                    "customer",
                ],
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 23)
            self.assertNotIn("Wait for checkpoint", result.stdout)
            self.assertEqual(
                (root / "args").read_text().splitlines(),
                [
                    "train",
                    "push",
                    "--config",
                    str(ROOT / "config.py"),
                    "--team",
                    "team with spaces",
                    "--profile",
                    "customer",
                ],
            )
            self.assertEqual((root / "repo").read_text(), "example/model")
            for args in (["--repo"], ["--unknown", "x"], ["--repo", "--team"]):
                with self.subTest(args=args):
                    result = subprocess.run(
                        ["bash", str(ROOT / "import_whisper_checkpoint.sh"), *args],
                        env=env,
                        check=False,
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode, 1)

    def test_monitor_waits_for_complete_checkpoint_after_training_exits(self):
        monitor = monitoring.Monitor(poll_sec=0)
        results = [
            {"training_job": {"checkpoint_sync_status": "SYNCING"}, "checkpoints": []},
            {
                "training_job": {},
                "checkpoints": [
                    {"checkpoint_id": "checkpoint-0", "sync_status": "SYNCING"}
                ],
            },
            {
                "training_job": {},
                "checkpoints": [
                    {"checkpoint_id": "checkpoint-0", "sync_status": "COMPLETE"}
                ],
            },
        ]
        with (
            patch.object(monitor, "watch"),
            patch.object(monitor, "read", side_effect=results) as read,
        ):
            monitor.training("job123", "checkpoint-0")
        self.assertEqual(read.call_count, 3)

    def test_monitor_cleans_up_tail_and_reports_cloud_failure(self):
        process = Mock()
        process.poll.return_value = None
        monitor = monitoring.Monitor(poll_sec=0)
        with (
            patch.object(monitoring.subprocess, "Popen", return_value=process),
            self.assertRaisesRegex(RuntimeError, "FAILED"),
        ):
            monitor.watch(
                ["train", "job", "logs"],
                lambda: "FAILED",
                lambda s: s == "READY",
                lambda s: s == "FAILED",
            )
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)

    def test_submission_parser_and_profile_forwarding(self):
        self.assertEqual(
            monitoring.submitted_job(
                "View logs via 'truss train logs --job-id abc123 --tail'"
            ),
            "abc123",
        )
        with self.assertRaisesRegex(ValueError, "do not resubmit"):
            monitoring.submitted_job("unrecognized submission output")
        monitor = monitoring.Monitor(profile="customer")
        response = SimpleNamespace(stdout='{"training_job": {}}')
        with patch.object(monitoring.subprocess, "run", return_value=response) as run:
            monitor.read("train", "job", "describe", "--job-id", "abc123")
        self.assertEqual(
            run.call_args.args[0][-4:], ["--profile", "customer", "--output", "json"]
        )


if __name__ == "__main__":
    unittest.main()
