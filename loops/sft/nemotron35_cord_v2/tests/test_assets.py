import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fixtures import PROCESSOR_CONFIG

import data
import prepare_assets


class AssetDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        self.revision = "a" * 40
        self.names = {*prepare_assets.TOKENIZER_FILES, "preprocessor_config.json"}
        self.download_directory = self.root / "downloads"
        self.download_directory.mkdir()
        for name in self.names:
            content = (
                json.dumps(PROCESSOR_CONFIG)
                if name == "preprocessor_config.json"
                else "fixture"
            )
            (self.download_directory / name).write_text(content)
        self.info = SimpleNamespace(
            sha=self.revision,
            siblings=[
                SimpleNamespace(rfilename=name)
                for name in sorted(self.names | {"model.safetensors"})
            ],
        )
        self.api = Mock()
        self.api.model_info.return_value = self.info
        self.tokenizer = SimpleNamespace(is_fast=True, chat_template="fixture")
        self.tokenizer_loader = Mock(return_value=self.tokenizer)
        self.probe = Mock()
        self.downloader = Mock(side_effect=self.download)
        patchers = (
            patch.multiple(
                prepare_assets,
                ROOT=self.root,
                settings=Mock(return_value={"base_model": self.repository}),
                HfApi=Mock(return_value=self.api),
                hf_hub_download=self.downloader,
                reference_matches=Mock(return_value=False),
            ),
            patch.object(
                prepare_assets.AutoTokenizer, "from_pretrained", self.tokenizer_loader
            ),
            patch.object(data, "encode_row", self.probe),
            patch("sys.argv", ["prepare_assets.py"]),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def download(self, repository, filename, revision):
        self.assertEqual(repository, self.repository)
        self.assertEqual(revision, self.revision)
        self.assertIn(filename, self.names)
        return self.download_directory / filename

    def test_default_setup_downloads_metadata_without_a_local_tokenizer_directory(self):
        with redirect_stdout(io.StringIO()):
            prepare_assets.main()
            prepare_assets.main()
        self.api.model_info.assert_called_once_with(self.repository, revision=None)
        self.assertEqual(self.downloader.call_count, len(self.names))
        self.assertEqual(
            {call.args[1] for call in self.downloader.call_args_list}, self.names
        )
        self.assertEqual(
            self.tokenizer_loader.call_args.kwargs,
            {"local_files_only": True, "trust_remote_code": False},
        )
        self.probe.assert_called_once_with(
            self.tokenizer, "<s_total>42</s_total>", 256, 8192
        )
        manifest = prepare_assets.verify_assets(self.root / "assets")
        self.assertEqual(
            manifest["source"],
            {
                "kind": "huggingface",
                "repository": self.repository,
                "revision": self.revision,
            },
        )
        self.assertFalse(manifest["remote_code_executed"])
        self.assertTrue(
            (self.root / "assets" / "tokenizer" / "tokenizer.json").is_file()
        )

    def test_missing_remote_processor_fails_before_downloading_anything(self):
        self.info.siblings = [SimpleNamespace(rfilename="tokenizer.json")]
        with self.assertRaisesRegex(ValueError, "lacks required metadata"):
            prepare_assets.main()
        self.downloader.assert_not_called()
        self.tokenizer_loader.assert_not_called()
        self.probe.assert_not_called()
        self.assertFalse((self.root / "assets").exists())


if __name__ == "__main__":
    unittest.main()
