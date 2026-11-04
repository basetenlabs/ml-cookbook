"""Portable account defaults and allowlisted explicit environment settings."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common


class PortabilityTests(unittest.TestCase):
    def test_recipe_uses_current_accounts_without_private_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = common.settings()
        self.assertEqual(config["recipe_version"], "cookbook-v2")
        self.assertIsNone(config["team"])
        self.assertIsNone(config["wandb_entity"])
        self.assertEqual(config["wandb_project"], "loops-starcount")

    def test_environment_overrides_only_account_fields_not_training_values(self):
        with patch.dict(os.environ, {}, clear=True):
            original = common.settings()
        environment = {"BASETEN_TEAM": "Example Team", "WANDB_ENTITY": "example-user",
                       "WANDB_PROJECT": "example-project", "HF_TOKEN": "never-in-config",
                       "BASETEN_API_KEY": "never-in-config", "learning_rate": "9000"}
        with patch.dict(os.environ, environment, clear=True):
            changed = common.settings()
        expected = dict(original, team="Example Team", wandb_entity="example-user", wandb_project="example-project")
        self.assertEqual(changed, expected)
        self.assertNotIn("never-in-config", json.dumps(changed))

    def test_blank_environment_values_preserve_defaults(self):
        with patch.dict(os.environ, {name: "  " for name in ("BASETEN_TEAM", "WANDB_ENTITY", "WANDB_PROJECT")}, clear=True):
            config = common.settings()
        self.assertIsNone(config["team"])
        self.assertIsNone(config["wandb_entity"])
        self.assertEqual(config["wandb_project"], "loops-starcount")

    def test_explicit_env_file_allows_only_six_settings_and_preserves_existing(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.env"
            path.write_text("BASETEN_API_KEY: fake-baseten\nWANDB_API_KEY=fake-wandb\nHF_TOKEN: fake-hf\n"
                            "BASETEN_TEAM: Example Team\nWANDB_ENTITY: example-user\nWANDB_PROJECT=example-project\n"
                            "LOOPS_REUSE_FROM_RUN_ID=never-load\nOTHER_TOKEN=never-load\n")
            with patch.dict(os.environ, {"WANDB_PROJECT": "existing-project"}, clear=True), \
                    redirect_stdout(StringIO()) as output, redirect_stderr(StringIO()) as errors:
                common.load_env(path)
                self.assertEqual(set(os.environ), common.ENV_KEYS)
                self.assertEqual(os.environ["WANDB_PROJECT"], "existing-project")
                config = common.settings()
                self.assertEqual(config["team"], "Example Team")
                self.assertEqual(config["wandb_entity"], "example-user")
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(errors.getvalue(), "")

    def test_missing_explicit_file_is_an_error_but_none_never_reads_files(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "explicitly supplied"):
                common.load_env(Path(directory) / "absent.env")
        with patch.object(Path, "is_file", side_effect=AssertionError("No implicit credential probe")):
            common.load_env()

    def test_account_override_rejects_newlines_without_echoing_value(self):
        with patch.dict(os.environ, {"BASETEN_TEAM": "team\nprivate-sentinel"}, clear=True):
            with self.assertRaises(ValueError) as caught:
                common.settings()
        self.assertNotIn("private-sentinel", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
