"""CPU checks for submission configuration and the shell-to-trainer contract."""

import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parent


class TrackingTest(unittest.TestCase):
    def test_gpu_request_and_effective_batch(self):
        for count in (1, 2, 4, 8):
            with (
                self.subTest(count=count),
                patch.dict(os.environ, {"GPU_COUNT": str(count)}, clear=True),
            ):
                config = runpy.run_path(str(ROOT / "config.py"))
                compute = config["training_compute"]
                env = config["training_runtime"].environment_variables
                self.assertEqual(compute.accelerator.count, count)
                self.assertEqual(env["EXPECTED_GPU_COUNT"], str(count))
                self.assertEqual(8 * int(env["GRAD_ACC"]) * count, 128)
                if count == 1:
                    self.assertEqual(compute.cpu_count, 8)
                    self.assertEqual(compute.memory, "64Gi")

    def test_unexpected_gpu_allocation_fails_before_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", str(ROOT / "run.sh")],
                cwd=directory,
                env={
                    "PATH": os.environ["PATH"],
                    "EXPECTED_GPU_COUNT": "1",
                    "BT_NUM_GPUS": "2",
                },
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Expected 1 GPUs, allocated 2", result.stderr)
            self.assertFalse((Path(directory) / ".venv").exists())

    def config(self, **env):
        with patch.dict(os.environ, env, clear=True):
            return runpy.run_path(str(ROOT / "config.py"))[
                "training_runtime"
            ].environment_variables

    def test_disabled_does_not_forward_wandb_credentials(self):
        env = self.config(
            WANDB_API_KEY="do-not-forward", WANDB_API_KEY_SECRET_NAME="unused"
        )
        self.assertEqual(env["REPORT_TO"], "none")
        self.assertFalse(any(name.startswith("WANDB") for name in env))

    def test_tensorboard_needs_no_secret(self):
        env = self.config(REPORT_TO="tensorboard", RUN_NAME="cpu-test")
        self.assertEqual(env["RUN_NAME"], "cpu-test")
        self.assertFalse(any(name.startswith("WANDB") for name in env))

    def test_wandb_uses_secret_reference(self):
        env = self.config(
            REPORT_TO="wandb",
            WANDB_API_KEY_SECRET_NAME="test-secret",
            WANDB_ENTITY="test-team",
            WANDB_PROJECT="test-project",
            WANDB_API_KEY="do-not-forward",
        )
        self.assertEqual(env["WANDB_API_KEY"].name, "test-secret")
        self.assertNotIn("do-not-forward", repr(env))
        self.assertEqual(env["WANDB_LOG_MODEL"], "false")
        self.assertEqual(env["WANDB_CONSOLE"], "off")

    def test_invalid_or_incomplete_configuration_fails(self):
        with self.assertRaisesRegex(ValueError, "REPORT_TO"):
            self.config(REPORT_TO="all")
        settings = dict(
            REPORT_TO="wandb",
            WANDB_API_KEY_SECRET_NAME="test-secret",
            WANDB_ENTITY="test-team",
            WANDB_PROJECT="test-project",
        )
        for missing in ("WANDB_API_KEY_SECRET_NAME", "WANDB_ENTITY", "WANDB_PROJECT"):
            with (
                self.subTest(missing=missing),
                self.assertRaisesRegex(ValueError, missing),
            ):
                self.config(**{k: v for k, v in settings.items() if k != missing})

    def test_shell_installs_only_requested_tracker_and_passes_arguments(self):
        for mode in ("none", "wandb", "tensorboard"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                venv = root / ".venv"
                (venv / "bin").mkdir(parents=True)
                (venv / "bin/activate").write_text(f'export PATH="{venv}/bin:$PATH"\n')
                fake_python = venv / "bin/python"
                fake_python.write_text(
                    '#!/bin/bash\nprintf "%s\\n" "$*" >> calls.txt\n'
                )
                fake_python.chmod(0o755)
                env = {
                    "PATH": os.environ["PATH"],
                    "REPORT_TO": mode,
                    "RUN_NAME": "run with spaces",
                    "MAX_STEPS": "2",
                    "NUM_GPUS": "1",
                    "WANDB_API_KEY": "sentinel-secret",
                }
                result = subprocess.run(
                    ["bash", "-x", str(ROOT / "run.sh")],
                    cwd=root,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                calls = (root / "calls.txt").read_text()
                self.assertIn(f"--report_to {mode}", calls)
                self.assertIn("--max_steps 2", calls)
                self.assertEqual("requirements.wandb.txt" in calls, mode == "wandb")
                self.assertEqual(
                    "requirements.tensorboard.txt" in calls, mode == "tensorboard"
                )
                self.assertNotIn(
                    "sentinel-secret", result.stdout + result.stderr + calls
                )
                if mode != "none":
                    self.assertIn("--run_name run with spaces", calls)


if __name__ == "__main__":
    unittest.main()
