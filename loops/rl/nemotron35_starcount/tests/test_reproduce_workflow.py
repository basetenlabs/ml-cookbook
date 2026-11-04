"""Exercise direct offline CLIs and check the documented script arguments."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import shlex
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_data  # noqa: E402
import preflight  # noqa: E402


class DirectWorkflowTests(unittest.TestCase):
    def test_generator_cli_routes_options_without_authentication(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "generated"
            report = {"compute_provisioned": False, "splits": {}}
            with patch("generate_data.generate_dataset", return_value=report) as generate:
                with patch("common.load_env", side_effect=AssertionError("No credential reads")):
                    with patch("common.api_request", side_effect=AssertionError("No API requests")):
                        with redirect_stdout(StringIO()) as output:
                            result = generate_data.main(["--rebuild", "--output-dir", str(destination)])
            self.assertEqual(result, 0)
            generate.assert_called_once_with(destination, rebuild=True)
            self.assertFalse(json.loads(output.getvalue())["compute_provisioned"])

    def test_offline_preflight_does_not_authenticate_or_check_remote_capabilities(self):
        audit = {"compute_provisioned": False, "splits": {}}
        with patch("preflight.settings", return_value={}), patch("preflight.prepare", return_value=({}, audit)):
            with patch("preflight.load_env", side_effect=AssertionError("No credential reads")):
                with patch("preflight.validate_catalog", side_effect=AssertionError("No API requests")):
                    with redirect_stdout(StringIO()) as output:
                        result = preflight.main(["--offline"])
        self.assertEqual(result, 0)
        displayed = json.loads(output.getvalue())
        self.assertTrue(displayed["offline"])
        self.assertFalse(displayed["capability_checked"])
        self.assertFalse(displayed["compute_provisioned"])

    def test_failed_local_preflight_stays_offline_and_returns_failure(self):
        with patch("preflight.settings", return_value={}):
            with patch("preflight.prepare", side_effect=ValueError("Frozen input checksum mismatch")):
                with patch("preflight.load_env", side_effect=AssertionError("No credential reads")):
                    with patch("preflight.validate_catalog", side_effect=AssertionError("No API requests")):
                        with redirect_stderr(StringIO()) as errors:
                            result = preflight.main(["--offline"])
        self.assertEqual(result, 1)
        self.assertIn("checksum mismatch", errors.getvalue())

    def test_public_workflow_uses_existing_scripts_and_supported_flags(self):
        readme = (ROOT / "README.md").read_text().replace("\\\n", " ")
        commands = re.findall(r"uv run --locked python ([A-Za-z_]+\.py)([^\n]*)", readme)
        self.assertTrue(commands)
        for script, tail in commands:
            with self.subTest(script=script, tail=tail):
                source = ROOT / script
                self.assertTrue(source.is_file(), f"Documented script is missing: {script}")
                tree = ast.parse(source.read_text())
                supported = {argument.value for call in ast.walk(tree)
                             if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                             and call.func.attr == "add_argument"
                             for argument in call.args
                             if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                             and argument.value.startswith("--")}
                flags = {word.split("=", 1)[0] for word in shlex.split(tail) if word.startswith("--")}
                self.assertTrue(flags <= supported, f"Unknown flags for {script}: {flags - supported}")
        first = {script: next(index for index, command in enumerate(commands) if command[0] == script)
                 for script in ("generate_data.py", "prepare_assets.py", "preflight.py", "train.py")}
        self.assertLess(first["generate_data.py"], first["prepare_assets.py"])
        self.assertLess(first["prepare_assets.py"], first["preflight.py"])
        self.assertLess(first["preflight.py"], first["train.py"])
        training = next(tail for script, tail in commands if script == "train.py")
        self.assertIn("--steps 3", training)
        self.assertNotIn("--resume", training)
        self.assertNotIn("--smoke-then-full", training)
        self.assertTrue(any(script == "train.py" and "--steps" not in tail and "--resume" not in tail
                            for script, tail in commands))
        self.assertNotIn("recipes/loops/", readme)
        self.assertNotIn("--accept-managed-differences", readme)


if __name__ == "__main__":
    unittest.main()
