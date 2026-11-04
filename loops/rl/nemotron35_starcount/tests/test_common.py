import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

import common
import preflight


ROW = {"target_color": "blue", "circles": [{"color": "blue"}, {"color": "red"}, {"color": "blue"}]}


class RewardTests(unittest.TestCase):
    def test_reasoning_box_never_earns_reward(self):
        self.assertEqual(common.reward(r"I see \boxed{2}. </think>Answer: \boxed{3}", ROW), 0)
        self.assertEqual(common.reward(r"I see \boxed{3}. </think>Answer: \boxed{2}", ROW), 1)

    def test_separator_is_required_for_thinking_continuation(self):
        self.assertEqual(common.reward(r"\boxed{2}", ROW), 0)
        self.assertEqual(common.reward(r"<think>\boxed{2}", ROW), 0)
        self.assertEqual(common.reward(r"<think>reasoning</think>\boxed{2}", ROW), 1)

    def test_first_box_and_exact_regex(self):
        self.assertEqual(common.reward(r"</think>\boxed{3} and \boxed{2}", ROW), 0)
        for malformed in (r"\boxed{ 2}", r"\boxed{2 }", r"\boxed{+2}", r"\boxed{2.0}", r"boxed{2}"):
            with self.subTest(malformed=malformed):
                self.assertEqual(common.reward("</think>" + malformed, ROW), 0)
        self.assertEqual(common.reward(r"</think>\boxed{02}", ROW), 1)

    def test_zero_count_and_optional_non_thinking_mode(self):
        row = {"target_color": "green", "circles": ROW["circles"]}
        self.assertEqual(common.reward(r"</think>\boxed{0}", row), 1)
        self.assertEqual(common.reward(r"\boxed{2}", ROW, starts_in_thinking=False, require_separator=False), 1)

    def test_malformed_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            common.ground_truth({"target_color": "blue", "circles": [{}]})


class CredentialTests(unittest.TestCase):
    def test_no_implicit_parent_credentials_are_read(self):
        with mock.patch.object(Path, "is_file", side_effect=AssertionError("Implicit file probe")), \
                mock.patch.object(Path, "read_text", side_effect=AssertionError("Implicit file read")), \
                mock.patch.dict(os.environ, {}, clear=True):
            common.load_env()
            self.assertNotIn("BASETEN_API_KEY", os.environ)
            self.assertNotIn("WANDB_API_KEY", os.environ)

    def test_env_allowlist_and_existing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.yaml"
            path.write_text("BASETEN_API_KEY: 'file-secret'\nWANDB_API_KEY=wandb-secret\nOTHER_TOKEN=never-load\n")
            output = io.StringIO()
            with mock.patch.dict(os.environ, {"BASETEN_API_KEY": "existing-secret"}, clear=True):
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    common.load_env(path)
                self.assertEqual(os.environ["BASETEN_API_KEY"], "existing-secret")
                self.assertEqual(os.environ["WANDB_API_KEY"], "wandb-secret")
                self.assertNotIn("OTHER_TOKEN", os.environ)
            self.assertEqual(output.getvalue(), "")

    def test_http_error_does_not_echo_credentials_or_body(self):
        secret = "unique-credential-must-not-print"
        error = urllib.error.HTTPError("https://api.baseten.co/v1/loops/capabilities?token=" + secret,
                                       401, secret, {}, io.BytesIO(secret.encode()))
        with mock.patch.dict(os.environ, {"BASETEN_API_KEY": secret}, clear=True):
            with mock.patch("common.urllib.request.urlopen", side_effect=error):
                with self.assertRaises(RuntimeError) as caught:
                    common.api_request("/v1/loops/capabilities?secret=" + secret)
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("HTTP 401", str(caught.exception))

    def test_external_auth_target_is_rejected(self):
        with self.assertRaises(ValueError):
            common.api_request("https://outside.example/v1/loops/capabilities")

    def test_catalog_requires_enabled_vision_and_rl_query(self):
        config = {"base_model": "nvidia/model", "max_seq_len": 4096}
        catalog = {"supported_models": [{"model_name": "nvidia/model", "enabled": True, "supports_vision_language": True}]}
        with mock.patch("common.api_request", return_value=catalog) as request:
            self.assertEqual(common.validate_catalog(config)["model_name"], "nvidia/model")
            self.assertIn("use_case=rl", request.call_args.args[0])
            self.assertIn("max_seq_len=4096", request.call_args.args[0])
            self.assertIn("model=nvidia%2Fmodel", request.call_args.args[0])
        for field in ("enabled", "supports_vision_language"):
            bad = {"supported_models": [dict(catalog["supported_models"][0], **{field: False})]}
            with mock.patch("common.api_request", return_value=bad), self.assertRaises(ValueError):
                common.validate_catalog(config)


class FrozenInputTests(unittest.TestCase):
    def test_manifest_identity_and_actual_bytes_are_both_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            frozen = {}
            for split in ("train", "validation"):
                path = root / "data" / f"{split}.jsonl"
                contents = json.dumps(ROW) + "\n"
                path.write_text(contents)
                frozen[split] = {"path": f"data/{split}.jsonl", "rows": 1,
                                 "sha256": hashlib.sha256(contents.encode()).hexdigest()}
            manifest_path = root / "data_manifest.json"
            manifest_path.write_text(json.dumps({"splits": frozen}))
            with mock.patch.object(common, "ROOT", root), mock.patch.object(common, "FROZEN_SPLITS", frozen):
                self.assertEqual(len(common.read_data(manifest_path)["train"]), 1)
                (root / "data" / "train.jsonl").write_text(json.dumps(dict(ROW, target_color="red")) + "\n")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    common.read_data(manifest_path)
                altered = json.loads(manifest_path.read_text())
                altered["splits"]["train"]["rows"] = 2
                manifest_path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "manifest identity mismatch"):
                    common.read_data(manifest_path)

    def test_tokenizer_missing_or_changed_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "tokenizer"
            snapshot.mkdir()
            hashes = {}
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                path = snapshot / name
                path.write_text(name)
                hashes[name] = common.sha256_file(path)
            manifest = {"tokenizer": {"path": "tokenizer", "file_sha256": hashes}}
            with mock.patch.object(common, "ROOT", root):
                self.assertEqual(common.validate_tokenizer(manifest)[1], hashes)
                (snapshot / "tokenizer.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    common.validate_tokenizer(manifest)

    def test_dataset_relocation_uses_only_packaged_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            contents = json.dumps(ROW) + "\n"
            frozen = {split: {"path": f"data/{split}.jsonl", "rows": 1,
                              "sha256": hashlib.sha256(contents.encode()).hexdigest()}
                      for split in ("train", "validation")}
            for split in frozen:
                (root / "data" / f"{split}.jsonl").write_text(contents)
            manifest = root / "data_manifest.json"
            with mock.patch.object(common, "ROOT", root), mock.patch.object(common, "FROZEN_SPLITS", frozen):
                manifest.write_text(json.dumps({"splits": frozen}))
                self.assertEqual(common.read_data()["train"], [ROW])
                for unsafe in (str(root / "data/train.jsonl"), "../data/train.jsonl", "elsewhere/train.jsonl"):
                    changed = {name: dict(spec) for name, spec in frozen.items()}
                    changed["train"]["path"] = unsafe
                    manifest.write_text(json.dumps({"splits": changed}))
                    with self.subTest(path=unsafe), self.assertRaisesRegex(ValueError, "relative path"):
                        common.read_data()

    def test_tokenizer_rejects_external_and_traversing_locations(self):
        for location in ("/tmp/tokenizer", "../tokenizer", "other-tokenizer"):
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "relative path"):
                common.validate_tokenizer({"tokenizer": {"path": location}})


class PreflightTests(unittest.TestCase):
    def test_offline_does_not_load_keys_or_call_api(self):
        audit = {"compute_provisioned": False}
        with mock.patch("preflight.settings", return_value={}), mock.patch("preflight.prepare", return_value=({}, audit)):
            with mock.patch("preflight.load_env") as credentials, mock.patch("preflight.validate_catalog") as api:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(preflight.main(["--offline"]), 0)
                credentials.assert_not_called()
                api.assert_not_called()
                self.assertFalse(json.loads(output.getvalue())["compute_provisioned"])


if __name__ == "__main__":
    unittest.main()
