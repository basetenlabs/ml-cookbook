import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fixtures import PROCESSOR_CONFIG

import cleanup
import common
import prepare_assets


class PortabilityTests(unittest.TestCase):
    def test_no_implicit_parent_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "env.yaml").write_text("BASETEN_API_KEY: do-not-load\n")
            with (
                patch.object(common, "ROOT", root / "recipe"),
                patch.dict(os.environ, {}, clear=True),
            ):
                common.load_env()
                self.assertNotIn("BASETEN_API_KEY", os.environ)

    def test_explicit_environment_preserves_shell_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "keys"
            source.write_text(
                "BASETEN_API_KEY: file-value\nWANDB_PROJECT=test-project\nUNRELATED=value\n"
            )
            with patch.dict(os.environ, {"BASETEN_API_KEY": "shell-value"}, clear=True):
                common.load_env(source)
                self.assertEqual(os.environ["BASETEN_API_KEY"], "shell-value")
                self.assertEqual(os.environ["WANDB_PROJECT"], "test-project")
                self.assertNotIn("UNRELATED", os.environ)

    def test_generic_account_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            config = common.settings()
            self.assertIsNone(config["team"])
            self.assertIsNone(config["wandb_entity"])
            self.assertEqual(config["wandb_project"], "loops-cord-v2")
        with patch.dict(
            os.environ,
            {"BASETEN_TEAM": "custom-team", "WANDB_PROJECT": "custom-project"},
        ):
            config = common.settings()
            self.assertEqual(config["team"], "custom-team")
            self.assertEqual(config["wandb_project"], "custom-project")

    def test_missing_assets_fail_before_api_calls(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "prepare_assets.py first"),
        ):
            prepare_assets.verify_assets(Path(temporary))

    def test_all_assets_are_fingerprinted_including_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tokenizer").mkdir()
            for name in prepare_assets.TOKENIZER_FILES:
                (root / "tokenizer" / name).write_text("fixture")
            (root / "preprocessor_config.json").write_text(json.dumps(PROCESSOR_CONFIG))
            files = {
                str(path.relative_to(root)): prepare_assets.fingerprint(path)
                for path in root.rglob("*")
                if path.is_file()
            }
            (root / "manifest.json").write_text(json.dumps({"files": files}))
            self.assertEqual(prepare_assets.verify_assets(root)["files"], files)
            (root / "tokenizer" / "chat_template.jinja").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Asset fingerprint changed"):
                prepare_assets.verify_assets(root)

    def test_cleanup_targets_only_recorded_trainer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "loops_run.json").write_text(json.dumps({"run_id": "owned-run"}))
            with patch.object(
                cleanup, "api_request", side_effect=[{}, {"status": "INACTIVE"}]
            ) as request:
                result = cleanup.cleanup(root)
            self.assertEqual(result["status"], "INACTIVE")
            self.assertEqual(
                request.call_args_list[0].args, ("/v1/loops/runs/owned-run/deactivate",)
            )
            self.assertEqual(request.call_args_list[0].kwargs, {"method": "POST"})
            self.assertEqual(request.call_count, 2)

    def test_cleanup_requires_sampler_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sampler.json").write_text(json.dumps({"sampler_id": "not-owned"}))
            with patch.object(cleanup, "api_request") as request:
                with self.assertRaisesRegex(ValueError, "ownership"):
                    cleanup.cleanup(root, sampler=True)
                request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
