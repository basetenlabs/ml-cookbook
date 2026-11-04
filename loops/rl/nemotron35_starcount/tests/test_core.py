"""CPU-only checks of GRPO algebra and full-sequence/next-token masks."""

from pathlib import Path
import math
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    LearningRateSchedule,
    RewardNormalization,
    causal_loss_view,
    group_advantages,
    grouped_advantages,
    learning_rates,
    loss_normalization_metadata,
    mask_sequence,
    context_limit_reached,
    leave_one_out_advantages,
    grouped_leave_one_out_advantages,
    warmup_learning_rates,
    reduce_masked_losses,
)


class GroupAdvantagesTests(unittest.TestCase):
    def assertSequenceAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected):
            self.assertAlmostEqual(value, target)

    def test_leave_one_out_matches_direct_baselines(self):
        rewards = [0, 0, 1, 1, 0, 1, 0, 1]
        expected = [(reward - (sum(rewards) - reward) / 7) for reward in rewards]
        actual = group_advantages(rewards, normalization=None)
        self.assertSequenceAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(actual), 0)

    def test_mixed_binary_population_normalization(self):
        config = RewardNormalization(0, 0, "inside_sqrt")
        actual = group_advantages([0, 0, 0, 0, 1, 1, 1, 1], normalization=config)
        self.assertSequenceAlmostEqual(actual, [-8 / 7] * 4 + [8 / 7] * 4)

    def test_unbiased_variance_is_explicit(self):
        population = group_advantages([0, 0, 0, 0, 1, 1, 1, 1],
                                     normalization=RewardNormalization(0, 0, "inside_sqrt"))
        sample = group_advantages([0, 0, 0, 0, 1, 1, 1, 1],
                                 normalization=RewardNormalization(1, 0, "inside_sqrt"))
        self.assertSequenceAlmostEqual(sample, [value * math.sqrt(7 / 8)
                                               for value in population])

    def test_epsilon_placement_is_explicit(self):
        inside = group_advantages([0, 1], normalization=RewardNormalization(0, 0.25, "inside_sqrt"))
        outside = group_advantages([0, 1], normalization=RewardNormalization(0, 0.25, "outside_sqrt"))
        clamp = group_advantages([0, 1], normalization=RewardNormalization(0, 1, "clamp_std"))
        self.assertSequenceAlmostEqual(inside, [-1 / math.sqrt(0.5), 1 / math.sqrt(0.5)])
        self.assertSequenceAlmostEqual(outside, [-1 / 0.75, 1 / 0.75])
        self.assertSequenceAlmostEqual(clamp, [-1, 1])

    def test_constant_zero_and_one_groups_have_no_signal(self):
        for reward in (0, 1):
            for placement in ("inside_sqrt", "outside_sqrt", "clamp_std"):
                for epsilon in (0, 1e-8):
                    with self.subTest(reward=reward, placement=placement, epsilon=epsilon):
                        self.assertEqual(group_advantages([reward] * 8,
                                         normalization=RewardNormalization(1, epsilon, placement)),
                                         [0.0] * 8)

    def test_normalization_is_of_rewards_not_rloo_advantages(self):
        values = group_advantages([0, 1], normalization=RewardNormalization(0, 0, "inside_sqrt"))
        self.assertEqual(values, [-2, 2])

    def test_standard_mean_baseline_without_leave_one_out(self):
        self.assertEqual(group_advantages([0, 1], normalization=None,
                                        use_leave_one_out_baseline=False), [-0.5, 0.5])

    def test_batch_layout_is_sixteen_independent_eight_response_groups(self):
        groups = [[0] * 8, [1] * 8] + [[0] * 4 + [1] * 4] * 14
        output = grouped_advantages(groups, normalization=RewardNormalization(0, 0, "inside_sqrt"))
        self.assertEqual(len(output), 16)
        self.assertTrue(all(len(group) == 8 for group in output))
        self.assertEqual(output[:2], [[0.0] * 8, [0.0] * 8])
        self.assertAlmostEqual(output[2][4], 8 / 7)
        with self.assertRaises(ValueError):
            grouped_advantages(groups[:15], normalization=None)
        with self.assertRaises(ValueError):
            grouped_advantages([[0] * 7] * 16, normalization=None)

    def test_invalid_and_nonfinite_groups_fail(self):
        for rewards in ([], [1], [0, math.nan], [0, math.inf], [0, -math.inf]):
            with self.subTest(rewards=rewards), self.assertRaises(ValueError):
                group_advantages(rewards, normalization=None)
        with self.assertRaises(ValueError):
            group_advantages([0, 1], normalization=RewardNormalization(2, 0, "inside_sqrt"))
        with self.assertRaises(ValueError):
            RewardNormalization(0, -1, "inside_sqrt")
        with self.assertRaises(ValueError):
            RewardNormalization(0, math.nan, "inside_sqrt")


class LearningRateTests(unittest.TestCase):
    def schedule(self, decay="cosine"):
        return LearningRateSchedule(warmup_steps=1, decay_end_step=50,
                                    peak_lr=3e-6, min_lr=3e-8,
                                    warmup_start_lr=0, decay=decay)

    def test_sample_rates_and_explicit_boundaries(self):
        schedule = self.schedule()
        self.assertEqual(schedule.at(0), 0)
        self.assertEqual(schedule.at(1), 3e-6)
        self.assertAlmostEqual(schedule.at(50), 3e-8)
        self.assertAlmostEqual(schedule.at(100), 3e-8)
        one_based = learning_rates(schedule, first_step=1)
        zero_based = learning_rates(schedule, first_step=0)
        self.assertEqual(len(one_based), 50)
        self.assertEqual(one_based[0], 3e-6)
        self.assertAlmostEqual(one_based[-1], 3e-8)
        self.assertEqual(zero_based[0], 0)
        self.assertGreater(zero_based[-1], 3e-8)

    def test_decay_kind_is_configurable(self):
        linear = self.schedule("linear")
        cosine = self.schedule("cosine")
        constant = self.schedule("constant")
        self.assertAlmostEqual(linear.at(25), 3e-8 + (25 / 49) * (3e-6 - 3e-8))
        self.assertNotAlmostEqual(linear.at(25), cosine.at(25), places=12)
        self.assertEqual(constant.at(50), 3e-6)

    def test_invalid_schedules_fail(self):
        with self.assertRaises(ValueError):
            LearningRateSchedule(1, 1, 3e-6, 3e-8, 0, "cosine")
        with self.assertRaises(ValueError):
            self.schedule().at(-1)
        with self.assertRaises(ValueError):
            learning_rates(self.schedule(), first_step=0, num_steps=-1)


class LeaveOneOutAdvantageTests(unittest.TestCase):
    def test_single_success_retains_raw_positive_advantage_when_other_std_is_zero(self):
        actual = leave_one_out_advantages([1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(actual[0], 1)
        expected_negative = -(1 / 7) / (math.sqrt(1 / 7) + 1e-6)
        for value in actual[1:]:
            self.assertAlmostEqual(value, expected_negative)
        # Per-response normalization does not preserve zero group sum.
        self.assertNotAlmostEqual(sum(actual), 0)

    def test_single_failure_retains_raw_negative_advantage(self):
        actual = leave_one_out_advantages([0, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(actual[0], -1)
        expected_positive = (1 / 7) / (math.sqrt(1 / 7) + 1e-6)
        for value in actual[1:]:
            self.assertAlmostEqual(value, expected_positive)

    def test_other_seven_reward_sample_variance_matches_direct_calculation(self):
        rewards = [0, 0, 1, 1, 0, 1, 0, 1]
        actual = leave_one_out_advantages(rewards)
        for index, value in enumerate(actual):
            others = rewards[:index] + rewards[index + 1:]
            baseline = sum(others) / 7
            sample_std = math.sqrt(sum((reward - baseline) ** 2 for reward in others) / 6)
            expected = (rewards[index] - baseline) / (sample_std + 1e-6)
            self.assertAlmostEqual(value, expected)

    def test_constant_groups_and_advantage_clamp(self):
        self.assertEqual(leave_one_out_advantages([0] * 8), [0] * 8)
        self.assertEqual(leave_one_out_advantages([1] * 8), [0] * 8)
        self.assertEqual(leave_one_out_advantages([100] + [0] * 7)[0], 50)
        self.assertEqual(leave_one_out_advantages([-100] + [0] * 7)[0], -50)
        with self.assertRaises(ValueError):
            leave_one_out_advantages([0, 1])
        with self.assertRaises(ValueError):
            leave_one_out_advantages([0, math.nan, 1])

    def test_group_layout_and_constant_schedule_start_at_preupdate_step(self):
        groups = grouped_leave_one_out_advantages([[1] + [0] * 7] * 16)
        self.assertEqual(len(groups), 16)
        self.assertEqual(groups[0][0], 1)
        rates = warmup_learning_rates()
        self.assertEqual(len(rates), 50)
        self.assertEqual(rates[0], 3e-8)
        self.assertAlmostEqual(rates[1], 3e-8 + (3e-6 - 3e-8) / 10)
        self.assertEqual(rates[10], 3e-6)
        self.assertEqual(rates[-1], 3e-6)


class ContextTruncationTests(unittest.TestCase):
    def test_generation_cap_below_context_limit_is_not_truncated(self):
        self.assertFalse(context_limit_reached(
            prompt_length=1510, response_length=256, max_seq_len=4096))

    def test_exact_total_context_limit_is_truncated(self):
        self.assertTrue(context_limit_reached(
            prompt_length=3840, response_length=256, max_seq_len=4096))

    def test_early_stop_and_image_expanded_context_lengths(self):
        self.assertFalse(context_limit_reached(
            prompt_length=691, response_length=80, max_seq_len=4096))
        self.assertTrue(context_limit_reached(
            prompt_length=4095, response_length=1, max_seq_len=4096))

    def test_invalid_lengths_and_budget_overflow_fail(self):
        for prompt, response, maximum in ((0, 1, 4096), (2, -1, 4096),
                                           (2, 1, 0), (4096, 1, 4096)):
            with self.subTest(prompt=prompt, response=response, maximum=maximum):
                with self.assertRaises(ValueError):
                    context_limit_reached(prompt_length=prompt,
                        response_length=response, max_seq_len=maximum)


class SequenceMaskTests(unittest.TestCase):
    def test_prompt_image_and_padding_targets_are_inactive(self):
        sequence = mask_sequence([10, 11, 999, 20, 998, 21, 0], response_start=3,
                                 response_end=6, advantage=0.75,
                                 response_logprobs=[-0.1, -0.3, -0.2],
                                 excluded_target_positions=[2, 4])
        self.assertEqual(sequence.target_ids, (-100, -100, -100, 20, -100, 21, -100))
        self.assertEqual(sequence.loss_mask, (0, 0, 0, 1, 0, 1, 0))
        self.assertEqual(sequence.advantages, (0, 0, 0, 0.75, 0, 0.75, 0))
        self.assertEqual(sequence.old_logprobs, (0, 0, 0, -0.1, 0, -0.2, 0))

    def test_first_response_target_aligns_with_previous_logit(self):
        sequence = mask_sequence([10, 11, 999, 20, 21], response_start=3,
                                 advantage=1.5, response_logprobs=[-0.1, -0.2])
        shifted = causal_loss_view(sequence)
        self.assertEqual(shifted.input_ids, (10, 11, 999, 20))
        self.assertEqual(shifted.target_ids, (-100, -100, 20, 21))
        self.assertEqual(shifted.loss_mask, (0, 0, 1, 1))
        self.assertEqual(shifted.advantages, (0, 0, 1.5, 1.5))
        self.assertEqual(shifted.old_logprobs, (0, 0, -0.1, -0.2))
        self.assertEqual(shifted.target_ids[3 - 1], 20)

    def test_zero_advantage_does_not_change_token_mask(self):
        sequence = mask_sequence([10, 20], response_start=1,
                                 advantage=0, response_logprobs=[-0.1])
        self.assertEqual(sequence.loss_mask, (0, 1))
        self.assertEqual(sequence.advantages, (0, 0))

    def test_empty_response_has_zero_active_tokens(self):
        sequence = mask_sequence([10, 11], response_start=2,
                                 advantage=0, response_logprobs=[])
        self.assertEqual(sequence.target_ids, (-100, -100))
        self.assertEqual(causal_loss_view(sequence).loss_mask, (0,))

    def test_bad_offsets_alignment_and_finite_values_fail(self):
        with self.assertRaises(ValueError):
            mask_sequence([10, 20], response_start=1, advantage=1, response_logprobs=[])
        with self.assertRaises(ValueError):
            mask_sequence([10, 20], response_start=3, advantage=1, response_logprobs=[])
        with self.assertRaises(ValueError):
            mask_sequence([10, 20], response_start=1, advantage=math.nan, response_logprobs=[-0.1])
        with self.assertRaises(ValueError):
            mask_sequence([10, 20], response_start=1, advantage=1, response_logprobs=[math.inf])
        with self.assertRaises(ValueError):
            causal_loss_view(mask_sequence([10, 20], response_start=0,
                                          advantage=1, response_logprobs=[-0.1, -0.2]))


class LossNormalizationTests(unittest.TestCase):
    def test_metadata_reports_raw_counts(self):
        metadata = loss_normalization_metadata([[0, 1, 1], [0, 1], [0, 0]])
        self.assertEqual(metadata.sequence_count, 3)
        self.assertEqual(metadata.active_sequence_count, 2)
        self.assertEqual(metadata.active_token_count, 3)
        self.assertEqual(metadata.active_tokens_by_sequence, (2, 1, 0))

    def test_reductions_preserve_their_distinct_denominators(self):
        losses, masks = [[999, 2, 4], [999, 9], [999]], [[0, 1, 1], [0, 1], [0]]
        self.assertEqual(reduce_masked_losses(losses, masks, reduction="token_mean"), 5)
        self.assertEqual(reduce_masked_losses(losses, masks, reduction="sequence_mean",
                                             include_empty_sequences=False), 6)
        self.assertEqual(reduce_masked_losses(losses, masks, reduction="sequence_mean",
                                             include_empty_sequences=True), 4)
        self.assertEqual(reduce_masked_losses(losses, masks, reduction="fixed_tokens",
                                             fixed_tokens_per_sequence=5), 1)

    def test_all_inactive_and_empty_batches_return_finite_zero(self):
        for losses, masks in (([[math.nan, math.inf]], [[0, 0]]), ([], [])):
            for reduction, kwargs in (("token_mean", {}),
                                      ("sequence_mean", {"include_empty_sequences": True}),
                                      ("sequence_mean", {"include_empty_sequences": False}),
                                      ("fixed_tokens", {"fixed_tokens_per_sequence": 256})):
                result = reduce_masked_losses(losses, masks, reduction=reduction, **kwargs)
                self.assertEqual(result, 0)
                self.assertTrue(math.isfinite(result))

    def test_invalid_masks_nonfinite_active_losses_and_missing_policies_fail(self):
        with self.assertRaises(ValueError):
            loss_normalization_metadata([[0, 2]])
        with self.assertRaises(ValueError):
            reduce_masked_losses([[math.nan]], [[1]], reduction="token_mean")
        with self.assertRaises(ValueError):
            reduce_masked_losses([[1]], [[1]], reduction="sequence_mean")
        with self.assertRaises(ValueError):
            reduce_masked_losses([[1]], [[1]], reduction="fixed_tokens")
        with self.assertRaises(ValueError):
            reduce_masked_losses([[1, 2]], [[1]], reduction="token_mean")


if __name__ == "__main__":
    unittest.main()
