"""Torch-only loss checks; no trainer, network, sampling, or allocation calls."""

from pathlib import Path
import math
import sys
from types import SimpleNamespace
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss import (  # noqa: E402
    PPOLossConfig,
    build_ppo_record,
    make_ppo_loss,
    valid_sequence_count,
)
from core import leave_one_out_advantages  # noqa: E402


EPS = 1e-8


def datum(record):
    return SimpleNamespace(loss_fn_inputs={"target_tokens": SimpleNamespace(
        data=list(record.target_tokens), shape=[record.input_length])})


def record(advantage=1, length=1, prompt_length=2, previous=None, sampler=None, **kwargs):
    previous = [-2.0] * length if previous is None else previous
    sampler = previous if sampler is None else sampler
    return build_ppo_record(prompt_length=prompt_length,
                            response_token_ids=list(range(20, 20 + length)),
                            advantage=advantage, previous_trainer_logprobs=previous,
                            generation_sampler_logprobs=sampler, **kwargs)


def current_tensor(rec, ratios=None, masked_value=0):
    values = [float(masked_value)] * rec.input_length
    ratios = [1.0] * rec.response_length if ratios is None else ratios
    for offset, ratio in enumerate(ratios):
        index = rec.response_offset + offset
        if rec.loss_mask[index]:
            values[index] = rec.previous_trainer_logprobs[index] + math.log(ratio)
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


class PPORecordTests(unittest.TestCase):
    def test_causal_labels_match_input_prompt_plus_response_without_final_token(self):
        rec = record(advantage=0.7, length=2, prompt_length=3,
                     previous=[-0.1, -0.2], sampler=[-0.3, -0.4])
        self.assertEqual(rec.input_length, 3 + 2 - 1)
        self.assertEqual(rec.response_offset, 3 - 1)
        self.assertEqual(rec.target_tokens, (-100, -100, 20, 21))
        self.assertEqual(rec.advantages, (0, 0, 0.7, 0.7))
        self.assertEqual(rec.previous_trainer_logprobs, (0, 0, -0.1, -0.2))
        self.assertEqual(rec.generation_sampler_logprobs, (0, 0, -0.3, -0.4))
        self.assertEqual(rec.loss_mask, (0, 0, 1, 1))

    def test_explicit_context_mask_excludes_sample_but_retains_scalar_group_advantage(self):
        rec = record(advantage=3, length=2, stop_reason="length", sample_mask=False)
        self.assertFalse(rec.sample_mask)
        self.assertFalse(rec.valid_sequence)
        self.assertEqual(rec.advantage, 3)
        self.assertEqual(rec.target_tokens, (-100, -100, -100))
        self.assertEqual(rec.advantages, (0, 0, 0))
        self.assertEqual(rec.previous_trainer_logprobs, (0, 0, 0))
        self.assertEqual(rec.generation_sampler_logprobs, (0, 0, 0))

    def test_snapshots_are_detached_and_immutable(self):
        previous = torch.tensor([-0.1, -0.2], requires_grad=True)
        sampler = torch.tensor([-0.3, -0.4], requires_grad=True)
        rec = record(length=2, previous=previous, sampler=sampler)
        before = rec.previous_trainer_logprobs
        with torch.no_grad():
            previous.fill_(-9)
            sampler.fill_(-8)
        self.assertEqual(rec.previous_trainer_logprobs, before)
        lp = current_tensor(rec)
        loss, _ = make_ppo_loss([rec], global_valid_sequence_count=1)([datum(rec)], [lp])
        loss.backward()
        self.assertIsNone(previous.grad)
        self.assertIsNone(sampler.grad)

    def test_image_response_position_is_inactive_and_zero_advantage_remains_valid(self):
        rec = record(advantage=0, length=3, excluded_response_positions=[1])
        self.assertEqual(rec.target_tokens, (-100, 20, -100, 22))
        self.assertEqual(rec.loss_mask, (0, 1, 0, 1))
        self.assertEqual(rec.previous_trainer_logprobs, (0, -2, 0, -2))
        self.assertTrue(rec.valid_sequence)
        self.assertEqual(valid_sequence_count([rec, record(stop_reason="length", sample_mask=False)]), 1)

    def test_empty_response_and_bad_shapes(self):
        self.assertTrue(record(length=0).valid_sequence)
        self.assertEqual(record(length=0).active_token_count, 0)
        with self.assertRaises(ValueError):
            record(length=2, previous=[-1])
        with self.assertRaises(ValueError):
            record(previous=[math.nan])
        with self.assertRaises(ValueError):
            record(prompt_length=0)


class PPOLossTests(unittest.TestCase):
    def run_loss(self, records, currents, global_count=None):
        if global_count is None:
            global_count = valid_sequence_count(records)
        return make_ppo_loss(records, global_valid_sequence_count=global_count)(
            [datum(rec) for rec in records], currents)

    def test_analytic_unclipped_gradient_with_global_denominator(self):
        rec = record(advantage=2, length=2)
        lp = current_tensor(rec)
        loss, metrics = self.run_loss([rec], [lp], global_count=3)
        denominator = (2 + EPS) * (3 + EPS)
        self.assertAlmostEqual(loss.item(), -4 / denominator)
        loss.backward()
        self.assertEqual(lp.grad[0].item(), 0)
        torch.testing.assert_close(lp.grad[1:], torch.full((2,), -2 / denominator, dtype=lp.dtype))
        self.assertEqual(metrics["active_tokens"], 2)

    def test_asymmetric_clipping_and_pessimistic_gradient(self):
        cases = ((1, 1.25, -1.25, -1.25),
                 (1, 1.29, -1.28, 0),
                 (-1, 0.79, 0.8, 0),
                 (-1, 0.85, 0.85, 0.85),
                 (1, 0.7, -0.7, -0.7),
                 (-1, 1.4, 1.4, 1.4))
        denominator = (1 + EPS) ** 2
        for advantage, ratio, expected_loss, expected_gradient in cases:
            with self.subTest(advantage=advantage, ratio=ratio):
                rec = record(advantage)
                lp = current_tensor(rec, [ratio])
                loss, _ = self.run_loss([rec], [lp])
                self.assertAlmostEqual(loss.item(), expected_loss / denominator)
                loss.backward()
                self.assertAlmostEqual(lp.grad[-1].item(), expected_gradient / denominator)

    def test_detached_token_tis_weights_are_clamped_to_half_and_two(self):
        previous = [-2.0] * 3
        sampler = [prev - math.log(weight) for prev, weight in zip(previous, (0.1, 1, 4))]
        rec = record(advantage=2, length=3, previous=previous, sampler=sampler)
        lp = current_tensor(rec)
        loss, metrics = self.run_loss([rec], [lp])
        denominator = (3 + EPS) * (1 + EPS)
        self.assertAlmostEqual(loss.item(), -7 / denominator)
        loss.backward()
        torch.testing.assert_close(lp.grad[1:], torch.tensor([-1, -2, -4], dtype=lp.dtype) / denominator)
        self.assertEqual(metrics["tis_oob_tokens"], 2)
        self.assertEqual(metrics["tis_weight_sum"], 3.5)

    def test_tis_overflow_maps_to_zero_before_clamp(self):
        rec = record(previous=[0.0], sampler=[-1000.0])
        lp = current_tensor(rec)
        loss, metrics = self.run_loss([rec], [lp])
        self.assertAlmostEqual(loss.item(), -0.5 / ((1 + EPS) ** 2))
        self.assertEqual(metrics["tis_weight_sum"], 0.5)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(lp.grad).all()))

    def test_sequence_mean_weights_short_and_long_completions_equally(self):
        short, long = record(1, length=1), record(3, length=3)
        short_lp, long_lp = current_tensor(short), current_tensor(long)
        loss, _ = self.run_loss([short, long], [short_lp, long_lp])
        expected = (-1 / (1 + EPS) - 9 / (3 + EPS)) / (2 + EPS)
        self.assertAlmostEqual(loss.item(), expected)
        loss.backward()
        self.assertAlmostEqual(short_lp.grad[-1].item(), -1 / ((1 + EPS) * (2 + EPS)))
        for gradient in long_lp.grad[1:]:
            self.assertAlmostEqual(gradient.item(), -3 / ((3 + EPS) * (2 + EPS)))

    def test_zero_advantage_sequence_stays_in_denominator_and_autograd(self):
        active, zero = record(1), record(0)
        active_lp, zero_lp = current_tensor(active), current_tensor(zero)
        loss, metrics = self.run_loss([active, zero], [active_lp, zero_lp])
        self.assertAlmostEqual(loss.item(), -1 / ((1 + EPS) * (2 + EPS)))
        self.assertEqual(metrics["valid_sequences"], 2)
        self.assertEqual(metrics["zero_advantage_sequences"], 1)
        loss.backward()
        self.assertIsNotNone(zero_lp.grad)
        self.assertTrue(bool((zero_lp.grad == 0).all()))

    def test_masked_sample_is_excluded_and_has_zero_grad_even_with_nan_slots(self):
        active, overlong = record(2), record(3, length=2, stop_reason="length", sample_mask=False)
        active_lp = current_tensor(active, masked_value=math.nan)
        overlong_lp = current_tensor(overlong, masked_value=math.nan)
        loss, metrics = self.run_loss([active, overlong], [active_lp, overlong_lp])
        self.assertAlmostEqual(loss.item(), -2 / ((1 + EPS) ** 2))
        self.assertEqual(metrics["valid_sequences"], 1)
        loss.backward()
        self.assertEqual(active_lp.grad[0].item(), 0)
        self.assertTrue(bool((overlong_lp.grad == 0).all()))

    def test_zero_active_batch_has_finite_zero_and_grad_for_every_tensor(self):
        records = [record(sample_mask=False), record(stop_reason="length", sample_mask=False)]
        currents = [current_tensor(rec, masked_value=math.nan) for rec in records]
        loss, metrics = self.run_loss(records, currents, global_count=0)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(metrics["valid_sequences"], 0)
        loss.backward()
        for lp in currents:
            self.assertIsNotNone(lp.grad)
            self.assertTrue(bool((lp.grad == 0).all()))

    def test_capped_negative_completion_trains_below_context_limit_and_full_context_is_excluded(self):
        negative_advantage = leave_one_out_advantages([1] + [0] * 7)[1]
        capped = record(negative_advantage, length=256, prompt_length=1300, stop_reason="length")
        full_context = record(negative_advantage, length=256, prompt_length=3840,
                              stop_reason="length", sample_mask=False)
        self.assertLess(capped.input_length + 1, 4096)
        self.assertEqual(full_context.input_length + 1, 4096)
        self.assertTrue(capped.sample_mask)
        self.assertEqual(capped.active_token_count, 256)
        self.assertFalse(full_context.sample_mask)
        self.assertEqual(full_context.active_token_count, 0)
        self.assertEqual(valid_sequence_count([capped, full_context]), 1)
        capped_lp = current_tensor(capped)
        full_context_lp = current_tensor(full_context, masked_value=math.nan)
        loss, metrics = self.run_loss([capped, full_context], [capped_lp, full_context_lp])
        self.assertGreater(loss.item(), 0)
        self.assertEqual(metrics["valid_sequences"], 1)
        self.assertEqual(metrics["active_tokens"], 256)
        loss.backward()
        expected_gradient = -negative_advantage / ((256 + EPS) * (1 + EPS))
        self.assertGreater(expected_gradient, 0)
        torch.testing.assert_close(capped_lp.grad[capped.response_offset:],
            torch.full((256,), expected_gradient, dtype=capped_lp.dtype))
        self.assertTrue(bool((capped_lp.grad[:capped.response_offset] == 0).all()))
        self.assertIsNotNone(full_context_lp.grad)
        self.assertTrue(bool((full_context_lp.grad == 0).all()))

    def test_zero_token_sample_counts_in_global_sequence_denominator(self):
        active, empty = record(), record(length=0)
        active_lp, empty_lp = current_tensor(active), current_tensor(empty)
        self.assertEqual(valid_sequence_count([active, empty]), 2)
        loss, _ = self.run_loss([active, empty], [active_lp, empty_lp])
        self.assertAlmostEqual(loss.item(), -1 / ((1 + EPS) * (2 + EPS)))
        loss.backward()
        self.assertIsNotNone(empty_lp.grad)
        self.assertTrue(bool((empty_lp.grad == 0).all()))

    def test_minibatch_losses_sum_to_full_global_sequence_loss(self):
        records = [record(1), record(-2, length=2)]
        currents = [current_tensor(rec) for rec in records]
        full_loss, _ = self.run_loss(records, currents)
        parts = [self.run_loss([rec], [lp], global_count=2)[0]
                 for rec, lp in zip(records, currents)]
        torch.testing.assert_close(sum(parts), full_loss)

    def test_nonfinite_active_probs_shape_or_record_order_fail(self):
        rec = record()
        with self.assertRaises(ValueError):
            self.run_loss([rec], [torch.tensor([0, math.nan], requires_grad=True)])
        with self.assertRaises(ValueError):
            self.run_loss([rec], [torch.tensor([0.0], requires_grad=True)])
        with self.assertRaises(ValueError):
            make_ppo_loss([rec], global_valid_sequence_count=0)
        callback = make_ppo_loss([rec], global_valid_sequence_count=1)
        other = record(sample_mask=False)
        with self.assertRaises(ValueError):
            callback([datum(other)], [current_tensor(rec)])
        with self.assertRaises(ValueError):
            PPOLossConfig(ratio_min=1.1)

    def test_real_sdk_datum_contract_uses_only_target_tokens(self):
        try:
            from baseten.loops import Datum, ModelInput, TensorData
        except ImportError:
            self.skipTest("SDK optional for local torch algebra checks")
        rec = record(length=2, prompt_length=3)
        sdk_datum = Datum(model_input=ModelInput.from_ints([10, 999, 11, 20]),
                          loss_fn_inputs={"target_tokens": TensorData(
                              data=list(rec.target_tokens), dtype="int64", shape=[rec.input_length])})
        lp = current_tensor(rec)
        loss, _ = make_ppo_loss([rec], global_valid_sequence_count=1)([sdk_datum], [lp])
        loss.backward()
        self.assertEqual(sdk_datum.model_input.length, rec.input_length)
        self.assertIsNotNone(lp.grad)


def scalar_policy_oracle(records, current_values, global_count, config):
    """Independent scalar, piecewise surrogate and analytical LP derivatives.

    Positive advantages cap only the upper ratio; negative advantages floor
    only the lower ratio. This avoids reproducing the tensor max/clamp graph.
    No torch operations or automatic differentiation compute the oracle.
    """
    sequence_losses, gradients = [], []
    for rec, values in zip(records, current_values):
        gradient = [0.0] * len(values)
        gradients.append(gradient)
        active = [index for index, mask in enumerate(rec.loss_mask) if mask]
        if not rec.sample_mask or not active:
            continue
        scale = 1 / ((len(active) + config.reduction_epsilon) *
                     (global_count + config.reduction_epsilon))
        token_losses = []
        for index in active:
            advantage = rec.advantages[index]
            ratio = math.exp(values[index] - rec.previous_trainer_logprobs[index])
            try:
                importance = math.exp(rec.previous_trainer_logprobs[index] -
                                      rec.generation_sampler_logprobs[index])
            except OverflowError:
                importance = 0.0
            if not math.isfinite(importance):
                importance = 0.0
            weight = min(config.tis_max, max(config.tis_min, importance))
            effective_ratio = (min(ratio, config.ratio_max) if advantage >= 0
                               else max(ratio, config.ratio_min))
            token_losses.append(-advantage * effective_ratio * weight)
            clipped = ((advantage > 0 and ratio > config.ratio_max) or
                       (advantage < 0 and ratio < config.ratio_min))
            if not clipped:
                gradient[index] = -advantage * ratio * weight * scale
        sequence_losses.append(math.fsum(token_losses) /
                               (len(active) + config.reduction_epsilon))
    return math.fsum(sequence_losses) / (global_count + config.reduction_epsilon), gradients


class IndependentPolicyMathTests(unittest.TestCase):
    def test_all_binary_counts_match_closed_form_leave_one_out_variance(self):
        for successes in range(9):
            rewards = [1.0] * successes + [0.0] * (8 - successes)
            expected = []
            for reward in rewards:
                other_successes = successes - reward
                # Seven Bernoulli observations: sample variance k(7-k)/(7*6).
                variance = other_successes * (7 - other_successes) / 42
                value = reward - other_successes / 7
                if variance:
                    value /= math.sqrt(variance) + 1e-6
                expected.append(min(50, max(-50, value)))
            actual = leave_one_out_advantages(rewards)
            for value, target in zip(actual, expected):
                self.assertAlmostEqual(value, target, places=12)

    @staticmethod
    def fixtures():
        records = [
            record(1.5, length=3, prompt_length=3,
                   sampler=[-2 - math.log(weight) for weight in (0.1, 1, 4)]),
            record(-2, length=3,
                   sampler=[-2 - math.log(weight) for weight in (4, 0.1, 1)]),
            record(0, length=2),
            record(7, length=2, stop_reason="length", sample_mask=False),
            record(3, length=0),
            record(0.25, length=3, excluded_response_positions=[1]),
        ]
        ratios = ([0.65, 1.1, 1.5], [0.7, 0.85, 1.6], [1, 1],
                  [1, 1], [], [1.1, 100, 0.9])
        currents = [current_tensor(rec, values, masked_value=math.nan)
                    for rec, values in zip(records, ratios)]
        return records, currents

    def test_loss_and_gradients_match_independent_piecewise_scalar_oracle(self):
        configs = (PPOLossConfig(), PPOLossConfig(ratio_min=0.75, ratio_max=1.35,
            tis_min=0.25, tis_max=1.75, reduction_epsilon=1e-5))
        for config in configs:
            with self.subTest(config=config):
                records, currents = self.fixtures()
                # Include two valid sequences outside this submitted minibatch.
                global_count = sum(rec.sample_mask for rec in records) + 2
                expected_loss, expected_gradients = scalar_policy_oracle(records,
                    [current.detach().tolist() for current in currents], global_count, config)
                actual_loss, metrics = make_ppo_loss(records,
                    global_valid_sequence_count=global_count, config=config)(
                        [datum(rec) for rec in records], currents)
                self.assertAlmostEqual(actual_loss.item(), expected_loss, places=12)
                self.assertEqual(metrics["valid_sequences"], 5)
                self.assertEqual(metrics["global_valid_sequences"], 7)
                actual_loss.backward()
                for current, expected in zip(currents, expected_gradients):
                    torch.testing.assert_close(current.grad,
                        torch.tensor(expected, dtype=current.dtype), rtol=1e-12, atol=1e-12)

    def test_callback_gradients_match_scalar_loss_finite_differences(self):
        records, currents = self.fixtures()
        config = PPOLossConfig()
        global_count = sum(rec.sample_mask for rec in records)
        values = [current.detach().tolist() for current in currents]
        loss, _ = make_ppo_loss(records, global_valid_sequence_count=global_count)(
            [datum(rec) for rec in records], currents)
        loss.backward()
        delta = 1e-6
        for row, (rec, current) in enumerate(zip(records, currents)):
            for index, active in enumerate(rec.loss_mask):
                if not active:
                    self.assertEqual(current.grad[index].item(), 0)
                    continue
                initial = values[row][index]
                values[row][index] = initial + delta
                above = scalar_policy_oracle(records, values, global_count, config)[0]
                values[row][index] = initial - delta
                below = scalar_policy_oracle(records, values, global_count, config)[0]
                values[row][index] = initial
                self.assertAlmostEqual(current.grad[index].item(), (above - below) / (2 * delta),
                                       places=7)


if __name__ == "__main__":
    unittest.main()
