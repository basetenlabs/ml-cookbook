"""Simulated full training loop using SDK data types but no remote services."""

from contextlib import ExitStack, redirect_stdout
from io import StringIO
import asyncio
import json
import logging
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from baseten.loops import ModelInput, TensorData


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train  # noqa: E402
from core import warmup_learning_rates  # noqa: E402


class Future:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class FakeRun:
    url = "https://unit.test/never-published"

    def __init__(self):
        self.summary, self.logs, self.finished = {}, [], None

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, values, **kwargs):
        self.logs.append((dict(values), dict(kwargs)))

    def finish(self, exit_code):
        self.finished = exit_code


class FakeTable:
    def __init__(self, columns):
        self.columns = columns

    def add_data(self, *args):
        pass


class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=False):
        return "correct" if tokens[0] == 20 else "incorrect"


class FakeCheckpoint:
    def __init__(self, checkpoint_id):
        self.checkpoint_id = checkpoint_id

    def model_dump(self, **kwargs):
        return {"checkpoint_id": self.checkpoint_id}


class FakeSampler:
    def __init__(self, client, version):
        self.client, self.version = client, version

    async def sample_async(self, prompt, num_samples, **kwargs):
        self.client.sampling_loop_ids.add(id(asyncio.get_running_loop()))
        params = kwargs["sampling_params"]
        self.client.events.append(("sample", self.version, num_samples, prompt.to_ints()[1]))
        if self.version != self.client.policy_version:
            raise AssertionError("sampling stale weights")
        self.client.sample_temperatures.append((num_samples, params.temperature))
        sequences = [SimpleNamespace(tokens=[20 + index, 90], logprobs=[-2.0, -2.0],
                                     stop_reason="length" if index == 7 else "stop",
                                     routed_experts=None)
                     for index in range(num_samples)]
        return SimpleNamespace(sequences=sequences, policy_version=self.version,
                               prompt_token_ids=prompt.to_ints(),
                               prompt_logprobs=[0.0] * prompt.length)

    def close(self):
        self.client.events.append(("sampler_close", self.version))

    async def aclose(self):
        self.client.sampling_loop_ids.add(id(asyncio.get_running_loop()))
        self.client.events.append(("sampler_aclose", self.version))


class FakeClient:
    run_id = "owned-unit-test-run"

    def __init__(self, fail_update=None, fail_checkpoint_step=None, hidden_checkpoint_step=None):
        self.events, self.sample_temperatures, self.optimizer_params = [], [], []
        self.sampling_loop_ids = set()
        self._version, self.step = 0, 0
        self.custom_since_step, self.forward_since_step = 0, 0
        self.fail_update, self.checkpoints = fail_update, []
        self.fail_checkpoint_step, self.hidden_checkpoint_step = fail_checkpoint_step, hidden_checkpoint_step

    @property
    def policy_version(self):
        return self._version

    def load_state_with_optimizer(self, path):
        self.step = int(path.removeprefix("unit-checkpoint-"))
        self._version = self.step
        self.events.append(("restore", self.step))
        return Future(SimpleNamespace(step=self.step))

    def save_weights_and_get_sampling_client(self, name):
        self.events.append(("snapshot", self.step, name))
        return FakeSampler(self, self._version)

    def forward(self, datums, loss_fn, loss_fn_config):
        self.events.append(("previous", self.step, len(datums)))
        self.forward_since_step += 1
        outputs = []
        for datum in datums:
            targets = datum.loss_fn_inputs["target_tokens"]
            self.assert_causal(datum)
            values = [None if token == -100 else -2.0 for token in targets.data]
            outputs.append({"logprobs": TensorData(data=values, dtype="float64", shape=targets.shape)})
        return Future(SimpleNamespace(metrics={"skip_token_normalization": 1.0}, loss_fn_outputs=outputs))

    @staticmethod
    def assert_causal(datum):
        tokens = datum.model_input.to_ints()
        targets = datum.loss_fn_inputs["target_tokens"].data
        if len(tokens) != len(targets) or targets[:2] != [-100, -100]:
            raise AssertionError("wrong causal shapes or prompt mask")
        if targets[2] != tokens[3] or targets[-1] != 90:
            raise AssertionError("first response did not align with preceding prompt logit")

    def forward_backward_custom(self, datums, callback):
        self.events.append(("custom", self.step, len(datums)))
        self.custom_since_step += 1
        currents = [torch.tensor([math.nan if token == -100 else -2.0
                                  for token in datum.loss_fn_inputs["target_tokens"].data],
                                 dtype=torch.float64, requires_grad=True)
                    for datum in datums]
        loss, metrics = callback(datums, currents)
        if metrics["global_valid_sequences"] != 16 * 8:
            raise AssertionError("global valid-sequence denominator changed with minibatch")
        if metrics["valid_sequences"] != 2 * 8:
            raise AssertionError("generation-capped samples below the context limit were masked")
        loss.backward()
        for datum, current in zip(datums, currents):
            if current.grad is None or not bool(torch.isfinite(current.grad).all()):
                raise AssertionError("SDK requires a finite gradient for every LP tensor")
            inactive = torch.tensor([token == -100 for token in datum.loss_fn_inputs["target_tokens"].data])
            if not bool((current.grad[inactive] == 0).all()):
                raise AssertionError("inactive tokens contributed gradients")
        return Future(SimpleNamespace(metrics={**metrics, "skip_token_normalization": 1.0}, loss=loss.item()))

    def optim_step(self, params):
        if self.forward_since_step != 8 or self.custom_since_step != 8:
            raise AssertionError("optimizer ran before all 128 responses were scored/backpropagated")
        self.forward_since_step = self.custom_since_step = 0
        self.events.append(("optimizer", self.step + 1))
        self.optimizer_params.append(params)
        self.step += 1
        self._version += 1
        metrics = {"skipped": 1.0} if self.step == self.fail_update else {"grad_norm": 0.2}
        return Future(SimpleNamespace(metrics=metrics))

    def save_state(self, name):
        self.events.append(("checkpoint", self.step, name))
        if self.step == self.fail_checkpoint_step:
            raise RuntimeError("unit checkpoint save failure")
        self.checkpoints.append(FakeCheckpoint(name))
        return Future(SimpleNamespace(path=f"unit-checkpoint-{self.step}"))

    def list_checkpoints(self):
        self.events.append(("registry", self.step))
        return [checkpoint for checkpoint in self.checkpoints
                if checkpoint.checkpoint_id != f"step-{self.hidden_checkpoint_step}"]

    def close(self):
        self.events.append(("client_close", self.step))


class FakeService:
    def __init__(self, client):
        self.client = client

    def create_lora_training_client(self, **kwargs):
        self.client.events.append(("create", kwargs))
        return self.client

    def close(self):
        self.client.events.append(("service_close",))


class TrainingLoopTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(train.__file__).parent / "recipe.json").read_text())
        self.audit = {"unit_test_fixed_dataset": True}
        self.examples = {
            "train": [self.example(index) for index in range(1024)],
            "validation": [self.example(2000 + index) for index in range(256)],
        }

    @staticmethod
    def example(index):
        ids = [10, 1000 + index, 11]
        return {"prompt": ModelInput.from_ints(ids), "expected_prompt_ids": ids,
                "row": {"index": index}, "ground_truth": 1}

    def exercise(self, output_root, *, name, resume_step=None, fail_update=None, steps=None,
                 smoke_then_full=False, fail_checkpoint_step=None, hidden_checkpoint_step=None,
                 expected_error=None):
        client = FakeClient(fail_update, fail_checkpoint_step, hidden_checkpoint_step)
        run, api_calls = FakeRun(), []
        argv = ["train.py", "--name", name]
        if smoke_then_full:
            argv.append("--smoke-then-full")
        if steps is not None:
            argv.extend(["--steps", str(steps)])
        if resume_step is not None:
            resume = output_root / "unit-resume.json"
            resume.write_text(json.dumps({
                "fingerprint": train.fingerprint(self.config, self.audit),
                "completed_updates": resume_step,
                "next_prompt_offset": resume_step * 16,
                "trainer_checkpoint": f"unit-checkpoint-{resume_step}",
                "source_run_id": "previous-owned-unit-test-run",
            }))
            argv.extend(["--resume", str(resume)])

        def api(path, **kwargs):
            api_calls.append((path, kwargs.get("method", "GET")))
            return {"run_id": client.run_id}

        def owned_cleanup(run_id, output):
            self.assertEqual(run_id, client.run_id)
            client.events.append(("owned_cleanup", run_id))
            report = {"run_id": run_id, "requested": True, "verified": True,
                      "sampler_deactivation_requested": False}
            train.write_json(output / "deactivation.json", report)
            return report

        original_write_json = train.write_json

        def tracked_write_json(path, value):
            original_write_json(path, value)
            if path.name == "smoke_success.json":
                client.events.append(("smoke_success", client.step))

        with ExitStack() as stack:
            stack.enter_context(patch.object(train, "ROOT", output_root))
            stack.enter_context(patch.object(train, "settings", return_value=self.config))
            stack.enter_context(patch.object(train, "prepare", return_value=(self.examples, self.audit)))
            stack.enter_context(patch.object(train, "validate_catalog", return_value={"unit_test": True}))
            stack.enter_context(patch.object(train, "load_env"))
            stack.enter_context(patch.object(train, "api_request", side_effect=api))
            stack.enter_context(patch.object(train, "cleanup_owned_run", side_effect=owned_cleanup))
            stack.enter_context(patch.object(train, "write_json", side_effect=tracked_write_json))
            stack.enter_context(patch.object(train, "ServiceClient", return_value=FakeService(client)))
            stack.enter_context(patch.object(
                train, "reward", side_effect=lambda text, row: int(text == "correct")))
            stack.enter_context(patch.object(train.wandb, "init", return_value=run))
            stack.enter_context(patch.object(train.wandb, "Table", FakeTable))
            stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()))
            stack.enter_context(patch.dict(train.os.environ, {
                "WANDB_API_KEY": "unit-test-placeholder", "BASETEN_API_KEY": "unit-test-placeholder",
            }, clear=True))
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(redirect_stdout(StringIO()))
            if fail_update is None and expected_error is None:
                train.main()
            else:
                with self.assertRaisesRegex(RuntimeError, expected_error or "skipped update"):
                    train.main()
        return client, run, api_calls, output_root / "outputs" / name

    def test_full_fifty_update_loop_grouping_order_schedule_and_owned_cleanup(self):
        with TemporaryDirectory(prefix="starcount-fake-train-") as directory:
            client, run, api_calls, output = self.exercise(Path(directory), name="baseten_starcount_fake50")
            self.assertEqual(client.step, 50)
            self.assertEqual(len(client.optimizer_params), 50)
            for params, expected in zip(client.optimizer_params, warmup_learning_rates()):
                self.assertAlmostEqual(params.learning_rate, expected)
                self.assertEqual((params.beta1, params.beta2, params.eps), (0.9, 0.99, 1e-8))
                self.assertEqual((params.weight_decay, params.grad_clip_norm), (0, 1))
            training_samples = [event for event in client.events if event[0] == "sample" and event[2] == 8]
            self.assertEqual(len(training_samples), 50 * 16)
            self.assertEqual([event[3] for event in training_samples], list(range(1000, 1800)))
            self.assertEqual([event[1] for event in training_samples],
                             [step for step in range(50) for _ in range(16)])
            self.assertEqual(len([event for event in client.events if event[0] == "custom"]), 50 * 8)
            self.assertEqual(len([event for event in client.events if event[0] == "snapshot"]), 51)
            self.assertEqual(len([event for event in client.events if event[0] == "sampler_aclose"]), 51)
            self.assertEqual(len(client.sampling_loop_ids), 1)
            self.assertEqual(len([event for event in client.events if event[0] == "checkpoint"]), 5)
            self.assertEqual({temperature for count, temperature in client.sample_temperatures
                              if count == 8}, {1.0})
            self.assertEqual({temperature for count, temperature in client.sample_temperatures
                              if count == 1}, {0.0})
            completion = json.loads((output / "completion.json").read_text())
            self.assertTrue(completion["complete"])
            self.assertEqual(completion["updates"], 50)
            self.assertEqual(completion["recipe_version"], self.config["recipe_version"])
            self.assertEqual(run.summary["training_api"], "baseten-loops")
            resume = json.loads((output / "resume.json").read_text())
            self.assertEqual(resume["next_prompt_offset"], 800)
            self.assertEqual(resume["trainer_checkpoint"], "unit-checkpoint-50")
            self.assertFalse((output / "smoke_success.json").exists())
            self.assertEqual(run.finished, 0)
            self.assertIn(("owned_cleanup", client.run_id), client.events)
            self.assertEqual(api_calls, [("/v1/loops/runs/owned-unit-test-run", "GET")])

    def test_smoke_gate_proves_validation_and_registered_checkpoint_before_update_two(self):
        with TemporaryDirectory(prefix="starcount-fake-smoke-full-") as directory:
            client, run, api_calls, output = self.exercise(
                Path(directory), name="baseten_starcount_fake_smoke_full", smoke_then_full=True)
            self.assertEqual(client.step, 50)
            self.assertEqual(len([event for event in client.events if event[0] == "create"]), 1)
            validation_at_one = [index for index, event in enumerate(client.events)
                                 if event[0] == "sample" and event[1:3] == (1, 1)]
            self.assertEqual(len(validation_at_one), 256)
            checkpoint_index = client.events.index(("checkpoint", 1, "step-1"))
            registry_index = client.events.index(("registry", 1))
            success_index = client.events.index(("smoke_success", 1))
            next_rollout_index = next(index for index, event in enumerate(client.events)
                                      if event[0] == "sample" and event[1:3] == (1, 8))
            self.assertLess(max(validation_at_one), checkpoint_index)
            self.assertLess(checkpoint_index, registry_index)
            self.assertLess(registry_index, success_index)
            self.assertLess(success_index, next_rollout_index)
            self.assertLess(success_index, client.events.index(("optimizer", 2)))
            self.assertEqual(len([event for event in client.events if event[0] == "checkpoint"]), 6)
            smoke = json.loads((output / "smoke_success.json").read_text())
            self.assertTrue(smoke["passed"])
            self.assertEqual(smoke["validation_examples"], 256)
            self.assertEqual(smoke["trainer_checkpoint"], "unit-checkpoint-1")
            self.assertTrue(smoke["checkpoint_registered"])
            self.assertEqual(smoke["run_id"], client.run_id)
            self.assertTrue(run.summary["smoke_passed"])
            self.assertEqual(run.finished, 0)
            self.assertIn(("owned_cleanup", client.run_id), client.events)

    def test_smoke_checkpoint_save_or_registration_failure_prevents_update_two(self):
        cases = ({"fail_checkpoint_step": 1, "expected_error": "unit checkpoint save failure"},
                 {"hidden_checkpoint_step": 1, "expected_error": "absent from the checkpoint registry"})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), TemporaryDirectory(prefix="starcount-fake-smoke-fail-") as directory:
                client, run, api_calls, output = self.exercise(
                    Path(directory), name="baseten_starcount_fake_smoke_fail", smoke_then_full=True, **kwargs)
                self.assertEqual(client.step, 1)
                self.assertEqual(len(client.optimizer_params), 1)
                self.assertFalse(any(event[0] == "sample" and event[1:3] == (1, 8)
                                     for event in client.events))
                self.assertFalse((output / "smoke_success.json").exists())
                self.assertFalse((output / "completion.json").exists())
                self.assertEqual(json.loads((output / "failure.json").read_text())["completed_updates"], 1)
                self.assertEqual(run.finished, 1)
                self.assertIn(("owned_cleanup", client.run_id), client.events)

    def test_standalone_one_update_smoke_preserves_original_behavior(self):
        with TemporaryDirectory(prefix="starcount-fake-one-update-") as directory:
            client, run, _, output = self.exercise(
                Path(directory), name="baseten_starcount_fake_one_update", steps=1)
            self.assertEqual(client.step, 1)
            self.assertEqual(client.events.count(("checkpoint", 1, "step-1")), 1)
            self.assertEqual(json.loads((output / "completion.json").read_text())["updates"], 1)
            self.assertFalse((output / "smoke_success.json").exists())
            self.assertEqual(run.finished, 0)

    def test_smoke_then_full_rejects_short_target_or_resume_before_provisioning(self):
        for extra in (("--steps", "1"), ("--resume", "never-read.json")):
            with self.subTest(extra=extra), patch.object(train, "settings", return_value=self.config), \
                    patch.object(train, "ServiceClient") as service, patch.object(sys, "argv", [
                        "train.py", "--smoke-then-full", *extra]), \
                    redirect_stdout(StringIO()), patch("sys.stderr", new_callable=StringIO):
                with self.assertRaises(SystemExit):
                    train.main()
                service.assert_not_called()

    def test_base_version_header_may_be_missing_but_post_update_header_is_required(self):
        for expected_version, reported_version, accepted in ((0, None, True), (0, -1, False), (1, None, False)):
            with self.subTest(expected_version=expected_version, reported_version=reported_version):
                client = FakeClient()
                sampler = FakeSampler(client, 0)

                async def sample_with_header(*args, **kwargs):
                    response = await sampler.sample_async(*args, **kwargs)
                    response.policy_version = reported_version
                    return response

                coroutine = train.rollouts(SimpleNamespace(sample_async=sample_with_header),
                    [self.examples["validation"][0]], self.config, update=0, training=False,
                    expected_version=expected_version, timeout=1, tokenizer=FakeTokenizer())
                with patch.object(train, "reward", return_value=1):
                    if accepted:
                        groups = asyncio.run(coroutine)
                        self.assertEqual(len(groups), 1)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "synchronized policy version"):
                            asyncio.run(coroutine)

    def test_rollout_mask_uses_total_context_not_generation_finish_reason(self):
        cases = ((1510, 256, "length", True, True),
                 (3840, 256, "length", True, False),
                 (4016, 80, "stop", True, False),
                 (691, 80, "stop", True, True),
                 (3840, 256, "length", False, True))
        for prompt_length, response_length, stop_reason, filtering, valid in cases:
            with self.subTest(prompt_length=prompt_length, response_length=response_length,
                              stop_reason=stop_reason, filtering=filtering):
                ids = [10] * prompt_length
                example = {"prompt": ModelInput.from_ints(ids), "expected_prompt_ids": ids,
                           "row": {}, "ground_truth": 1}

                async def sample(prompt, **kwargs):
                    sequence = SimpleNamespace(tokens=[20] * response_length,
                        logprobs=[-2.0] * response_length, stop_reason=stop_reason)
                    return SimpleNamespace(sequences=[sequence], policy_version=0,
                        prompt_token_ids=ids, prompt_logprobs=[0.0] * prompt_length)

                config = {**self.config, "overlong_filtering": filtering}
                with patch.object(train, "reward", return_value=0):
                    groups = asyncio.run(train.rollouts(SimpleNamespace(sample_async=sample),
                        [example], config, update=0, training=False, expected_version=0,
                        timeout=1, tokenizer=FakeTokenizer()))
                self.assertEqual(groups[0][0]["valid"], valid)
                self.assertEqual(groups[0][0]["context_truncated"],
                                 prompt_length + response_length == 4096)

    def test_mask_revision_changes_resume_fingerprint(self):
        legacy = {key: value for key, value in self.config.items()
                  if key != "overlong_filtering_basis"}
        self.assertNotEqual(train.fingerprint(legacy, self.audit),
                            train.fingerprint(self.config, self.audit))

    def test_interrupt_records_failure_and_deactivates_owned_run(self):
        with TemporaryDirectory(prefix="starcount-fake-interrupt-") as directory:
            name = "baseten_starcount_fake_interrupt"
            with patch.object(FakeClient, "optim_step", side_effect=KeyboardInterrupt), \
                    self.assertRaises(KeyboardInterrupt):
                self.exercise(Path(directory), name=name, steps=1)
            output = Path(directory) / "outputs" / name
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["exception_type"], "KeyboardInterrupt")
            self.assertEqual(failure["completed_updates"], 0)
            self.assertFalse((output / "completion.json").exists())
            self.assertEqual(json.loads((output / "deactivation.json").read_text()),
                             {"run_id": "owned-unit-test-run", "requested": True,
                              "verified": True, "sampler_deactivation_requested": False})

    def test_owned_creation_hint_is_recorded_before_sdk_readiness_failure(self):
        with TemporaryDirectory(prefix="starcount-recorder-") as directory:
            output = Path(directory)
            recorder = train.RunRecorder(output)
            record = logging.LogRecord("baseten.loops.service_client", logging.INFO, __file__, 1,
                "To reconnect: export LOOPS_REUSE_FROM_RUN_ID=owned-created-before-readiness", (), None)
            recorder.emit(record)
            self.assertEqual(recorder.run_id, "owned-created-before-readiness")
            state = json.loads((output / "loops_run.json").read_text())
            self.assertEqual(state["run_id"], recorder.run_id)

    def test_provisioning_failure_still_invokes_owned_resource_cleanup(self):
        def fail_after_creation(*args, **kwargs):
            logging.getLogger("baseten.loops.service_client").warning(
                "To reconnect: export LOOPS_REUSE_FROM_RUN_ID=owned-unit-test-run")
            raise RuntimeError("unit SDK readiness failure")

        with TemporaryDirectory(prefix="starcount-fake-provisioning-") as directory, \
                patch.object(FakeService, "create_lora_training_client", side_effect=fail_after_creation):
            client, run, _, output = self.exercise(Path(directory),
                name="baseten_starcount_fake_provisioning", steps=1,
                expected_error="unit SDK readiness failure")
            self.assertEqual(client.step, 0)
            self.assertIn(("owned_cleanup", client.run_id), client.events)
            self.assertEqual(json.loads((output / "failure.json").read_text())["completed_updates"], 0)
            self.assertTrue(json.loads((output / "deactivation.json").read_text())["verified"])
            self.assertEqual(run.finished, 1)

    def test_resume_restores_optimizer_then_continues_correct_prompt_offset(self):
        with TemporaryDirectory(prefix="starcount-fake-resume-") as directory:
            client, run, api_calls, output = self.exercise(
                Path(directory), name="baseten_starcount_fake_resume", resume_step=40)
            self.assertEqual(client.step, 50)
            self.assertEqual(len(client.optimizer_params), 10)
            self.assertLess(client.events.index(("restore", 40)),
                            client.events.index(("snapshot", 40, "initial-40")))
            training_samples = [event for event in client.events if event[0] == "sample" and event[2] == 8]
            self.assertEqual([event[3] for event in training_samples], list(range(1640, 1800)))
            self.assertEqual(training_samples[0][1], 40)
            self.assertEqual(json.loads((output / "completion.json").read_text())["updates"], 50)
            self.assertEqual(run.finished, 0)
            self.assertIn(("owned_cleanup", client.run_id), client.events)

    def test_skipped_optimizer_update_aborts_and_deactivates_only_owned_run(self):
        with TemporaryDirectory(prefix="starcount-fake-failure-") as directory:
            client, run, api_calls, output = self.exercise(
                Path(directory), name="baseten_starcount_fake_fail", fail_update=1)
            self.assertEqual(len(client.optimizer_params), 1)
            failure = json.loads((output / "failure.json").read_text())
            self.assertFalse(failure["complete"])
            self.assertEqual(failure["completed_updates"], 0)
            self.assertFalse((output / "completion.json").exists())
            self.assertEqual(run.finished, 1)
            self.assertIn(("owned_cleanup", client.run_id), client.events)


if __name__ == "__main__":
    unittest.main()
