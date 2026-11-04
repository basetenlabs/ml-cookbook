"""Offline tokenizer setup tests with synthetic files and mocked Hub access."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_assets


class AssetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="starcount-assets-test-")
        self.addCleanup(temporary.cleanup)
        self.root, self.source = (Path(temporary.name) / name for name in ("package", "local"))
        self.root.mkdir()
        self.source.mkdir()
        self.payloads = {name: ("synthetic " + name).encode() for name in prepare_assets.TOKENIZER_FILES}
        self.hashes = {name: hashlib.sha256(content).hexdigest() for name, content in self.payloads.items()}
        for name, content in self.payloads.items():
            (self.source / name).write_bytes(content)
        self.manifest = {"tokenizer": {"path": "tokenizer", "file_sha256": self.hashes}}
        (self.root / "data_manifest.json").write_text(json.dumps(self.manifest))
        guard = patch("socket.socket.connect", side_effect=AssertionError("No network in asset tests"))
        guard.start()
        self.addCleanup(guard.stop)

    def prepare(self, **kwargs):
        return prepare_assets.prepare_tokenizer(self.root, **kwargs)

    def assert_published(self):
        target = self.root / "tokenizer"
        self.assertEqual({path.name for path in target.iterdir()}, prepare_assets.TOKENIZER_FILES)
        for name, expected in self.payloads.items():
            self.assertEqual((target / name).read_bytes(), expected)

    def test_local_import_copies_only_pinned_files_without_hub(self):
        (self.source / "model.safetensors").write_bytes(b"never copy model weights")
        (self.source / "remote_code.py").write_text("raise AssertionError()")
        with patch("prepare_assets._download", side_effect=AssertionError("No Hub initialization")):
            result = self.prepare(tokenizer_dir=self.source)
        self.assert_published()
        self.assertFalse(result["network_used"])
        self.assertFalse(result["compute_provisioned"])
        self.assertFalse((self.root / "tokenizer_source.json").exists())

    def test_matching_cache_is_reused_without_source_or_rewrite(self):
        self.prepare(tokenizer_dir=self.source)
        target = self.root / "tokenizer"
        before = {path.name: path.stat().st_mtime_ns for path in target.iterdir()}
        with patch("prepare_assets._download", side_effect=AssertionError("Cache must remain offline")):
            result = self.prepare()
        self.assertEqual(result["status"], "reused")
        self.assertEqual(before, {path.name: path.stat().st_mtime_ns for path in target.iterdir()})

    def test_partial_matching_cache_is_completed_without_overwrite(self):
        target = self.root / "tokenizer"
        target.mkdir()
        name = sorted(self.payloads)[0]
        (target / name).write_bytes(self.payloads[name])
        before = (target / name).stat().st_mtime_ns
        self.prepare(tokenizer_dir=self.source)
        self.assert_published()
        self.assertEqual((target / name).stat().st_mtime_ns, before)

    def test_mismatch_or_missing_source_publishes_nothing(self):
        name = sorted(self.payloads)[-1]
        for content in (b"mismatch", None):
            with self.subTest(content=content):
                path = self.source / name
                if content is None:
                    path.unlink()
                else:
                    path.write_bytes(content)
                with self.assertRaises(ValueError):
                    self.prepare(tokenizer_dir=self.source)
                self.assertFalse((self.root / "tokenizer").exists())

    def test_existing_mismatch_is_never_overwritten_or_downloaded_over(self):
        target = self.root / "tokenizer"
        target.mkdir()
        path = target / "tokenizer.json"
        path.write_bytes(b"user owns this file")
        with patch("prepare_assets._download", side_effect=AssertionError("Must fail before download")):
            with self.assertRaisesRegex(ValueError, "Existing tokenizer file"):
                self.prepare(model_repository="example/model", revision="a" * 40)
        self.assertEqual(path.read_bytes(), b"user owns this file")

    def test_directory_and_file_symlink_targets_fail_closed(self):
        target = self.root / "tokenizer"
        target.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepare(tokenizer_dir=self.source)
        target.unlink()
        target.mkdir()
        (target / "tokenizer.json").symlink_to(self.source / "tokenizer.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepare(tokenizer_dir=self.source)
        self.assertEqual((self.source / "tokenizer.json").read_bytes(), self.payloads["tokenizer.json"])

    def test_manifest_cannot_add_arbitrary_or_credential_filenames(self):
        self.manifest["tokenizer"]["file_sha256"][".env"] = "a" * 64
        (self.root / "data_manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "exactly the three"):
            self.prepare(tokenizer_dir=self.source)
        self.assertFalse((self.root / "tokenizer").exists())

    def test_only_three_files_from_immutable_revision_are_downloaded(self):
        def fake_download(repository, filename, **kwargs):
            destination = Path(kwargs["local_dir"]) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.payloads[filename])
            return str(destination)

        with patch("huggingface_hub.hf_hub_download", side_effect=fake_download) as download, \
                patch.dict(os.environ, {"HF_TOKEN": "synthetic-test-token"}, clear=True):
            result = self.prepare(model_repository="example/model", revision="a" * 40)
        self.assert_published()
        self.assertTrue(result["network_used"])
        self.assertEqual(download.call_count, 3)
        self.assertEqual({call.kwargs["filename"] for call in download.call_args_list}, prepare_assets.TOKENIZER_FILES)
        for call in download.call_args_list:
            self.assertEqual(call.args, ("example/model",))
            self.assertEqual(call.kwargs["revision"], "a" * 40)
            self.assertEqual(call.kwargs["repo_type"], "model")
            self.assertEqual(call.kwargs["endpoint"], "https://huggingface.co")
            self.assertEqual(call.kwargs["token"], "synthetic-test-token")

    def test_default_setup_downloads_public_manifest_snapshot_without_credentials(self):
        repository = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        revision = "f" * 40
        self.manifest["tokenizer"].update(model_repository=repository, revision=revision)
        (self.root / "data_manifest.json").write_text(json.dumps(self.manifest))

        def fake_download(selected_repository, filename, **kwargs):
            destination = Path(kwargs["local_dir"]) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.payloads[filename])
            return str(destination)

        with patch("huggingface_hub.hf_hub_download", side_effect=fake_download) as download, \
                patch.dict(os.environ, {}, clear=True):
            result = self.prepare(record_source=True)
        self.assert_published()
        self.assertTrue(result["network_used"])
        self.assertEqual(download.call_count, 3)
        for call in download.call_args_list:
            self.assertEqual(call.args, (repository,))
            self.assertEqual(call.kwargs["revision"], revision)
            self.assertIs(call.kwargs["token"], False)
        provenance = json.loads((self.root / "tokenizer_source.json").read_text())
        self.assertEqual(provenance["model_repository"], repository)
        self.assertEqual(provenance["revision"], revision)

    def test_default_setup_rejects_mutable_manifest_revision_before_download(self):
        self.manifest["tokenizer"].update(model_repository="example/model", revision="main")
        (self.root / "data_manifest.json").write_text(json.dumps(self.manifest))
        with patch("prepare_assets._download", side_effect=AssertionError("No mutable snapshot download")):
            with self.assertRaisesRegex(ValueError, "immutable"):
                self.prepare()
        self.assertFalse((self.root / "tokenizer").exists())

    def test_packaged_public_snapshot_matches_recipe_model(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "data_manifest.json").read_text())
        config = json.loads((root / "recipe.json").read_text())
        self.assertEqual(manifest["tokenizer"]["model_repository"], config["base_model"])
        self.assertRegex(manifest["tokenizer"]["revision"], r"^[a-f0-9]{40}$")
        self.assertEqual(set(manifest["tokenizer"]["file_sha256"]), prepare_assets.TOKENIZER_FILES)

    def test_missing_or_mutable_revision_is_rejected_before_download(self):
        with patch("prepare_assets._download", side_effect=AssertionError("No unpinned network request")):
            for revision in (None, "main", "v1", "abcd", "g" * 40):
                with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "immutable"):
                    self.prepare(model_repository="example/model", revision=revision)
        with self.assertRaisesRegex(ValueError, "not both"):
            self.prepare(tokenizer_dir=self.source, model_repository="example/model", revision="a" * 40)

    def test_hub_error_does_not_echo_token_response_or_signed_url(self):
        secret = "synthetic-token-do-not-echo"
        with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("https://signed.example/" + secret)):
            with self.assertRaises(ValueError) as caught:
                self.prepare(model_repository="example/model", revision="a" * 40)
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn("https://", str(caught.exception))
        self.assertFalse((self.root / "tokenizer").exists())

    def test_optional_provenance_does_not_record_local_paths(self):
        self.prepare(tokenizer_dir=self.source, record_source=True)
        content = (self.root / "tokenizer_source.json").read_text()
        self.assertNotIn(str(self.source), content)
        self.assertEqual(json.loads(content), {"source": "local-import", "file_sha256": self.hashes})

    def test_cached_source_record_is_optional_and_never_rewritten(self):
        self.prepare(tokenizer_dir=self.source)
        self.prepare(record_source=True)
        metadata = self.root / "tokenizer_source.json"
        self.assertEqual(json.loads(metadata.read_text())["source"], "verified-cache")
        before = metadata.read_bytes(), metadata.stat().st_mtime_ns
        self.prepare(record_source=True)
        self.assertEqual((metadata.read_bytes(), metadata.stat().st_mtime_ns), before)

    def test_cli_accepts_repository_revision_and_explicit_environment(self):
        environment = self.root / "settings.env"
        environment.write_text("HF_TOKEN=fake-cli-token\n")
        report = {"status": "prepared", "compute_provisioned": False}

        def prepared(**kwargs):
            self.assertEqual(os.environ["HF_TOKEN"], "fake-cli-token")
            self.assertEqual(kwargs["model_repository"], "example/model")
            self.assertEqual(kwargs["revision"], "a" * 40)
            return report

        with patch.dict(os.environ, {}, clear=True), patch("prepare_assets.prepare_tokenizer", side_effect=prepared), \
                redirect_stdout(StringIO()) as output:
            result = prepare_assets.main(["--model-repository", "example/model", "--revision", "a" * 40,
                                          "--env-file", str(environment)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_cli_missing_explicit_env_file_fails_without_preparing(self):
        with patch("prepare_assets.prepare_tokenizer") as prepare, redirect_stdout(StringIO()) as output, \
                redirect_stderr(StringIO()) as errors:
            result = prepare_assets.main(["--env-file", str(self.root / "absent.env")])
        self.assertEqual(result, 1)
        prepare.assert_not_called()
        self.assertEqual(output.getvalue(), "")
        self.assertIn("explicitly supplied", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
