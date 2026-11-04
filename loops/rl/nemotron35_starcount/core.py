"""Pure Python GRPO algebra, causal loss masks, and schedule utilities.

Variance, epsilon, schedule, and loss-reduction settings are explicit rather
than inferred from the name "GRPO".
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence


EpsilonPlacement = Literal["inside_sqrt", "outside_sqrt", "clamp_std"]
Decay = Literal["constant", "linear", "cosine"]
LossReduction = Literal["token_mean", "sequence_mean", "fixed_tokens"]


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class RewardNormalization:
    """Explicit reward standard-deviation convention.

    ``ddof=0`` uses population variance; ``ddof=1`` uses sample variance.
    No convention is the implicit default. Epsilon may be added to variance,
    added to standard deviation, or used as a minimum standard deviation.
    """

    ddof: int
    epsilon: float
    epsilon_placement: EpsilonPlacement

    def __post_init__(self) -> None:
        _integer(self.ddof, "ddof")
        epsilon = _finite(self.epsilon, "epsilon")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        object.__setattr__(self, "epsilon", epsilon)
        if self.epsilon_placement not in {"inside_sqrt", "outside_sqrt", "clamp_std"}:
            raise ValueError("unsupported epsilon placement")


def group_advantages(
    rewards: Sequence[float],
    *,
    normalization: RewardNormalization | None,
    use_leave_one_out_baseline: bool = True,
) -> list[float]:
    """Return advantages for one prompt's independently sampled responses.

    RLOO gives ``(r_i - mean(r)) * n / (n - 1)``. When enabled,
    normalization divides this by the standard deviation of *rewards*, not
    the standard deviation of the already computed RLOO advantages. Passing
    ``normalization=None`` explicitly disables reward normalization.
    """

    values = [_finite(reward, "reward") for reward in rewards]
    count = len(values)
    if not count:
        raise ValueError("a reward group must contain at least one response")
    if use_leave_one_out_baseline and count < 2:
        raise ValueError("leave-one-out baseline requires at least two responses")
    mean = _finite(math.fsum(values) / count, "reward mean")
    centered = [_finite(value - mean, "centered reward") for value in values]
    multiplier = count / (count - 1) if use_leave_one_out_baseline else 1.0
    if normalization is not None:
        divisor = count - normalization.ddof
        if divisor <= 0:
            raise ValueError("reward group size must exceed variance ddof")
        variance = _finite(math.fsum(value * value for value in centered) / divisor,
                           "reward variance")
        if normalization.epsilon_placement == "inside_sqrt":
            scale = math.sqrt(variance + normalization.epsilon)
        elif normalization.epsilon_placement == "outside_sqrt":
            scale = math.sqrt(variance) + normalization.epsilon
        else:
            scale = max(math.sqrt(variance), normalization.epsilon)
        scale = _finite(scale, "reward normalization scale")
        if scale == 0:
            # Constant groups have no learning signal, even with epsilon=0.
            if any(centered):
                raise ValueError("nonconstant rewards have zero normalization scale")
            return [0.0] * count
        multiplier /= scale
    return [_finite(value * multiplier, "advantage") for value in centered]


def grouped_advantages(
    reward_groups: Sequence[Sequence[float]],
    *,
    normalization: RewardNormalization | None,
    num_prompts: int = 16,
    responses_per_prompt: int = 8,
    use_leave_one_out_baseline: bool = True,
) -> list[list[float]]:
    """Validate update layout before computing each prompt's own baseline."""

    _integer(num_prompts, "num_prompts", minimum=1)
    _integer(responses_per_prompt, "responses_per_prompt", minimum=1)
    if len(reward_groups) != num_prompts:
        raise ValueError(f"expected {num_prompts} prompt groups")
    if any(len(group) != responses_per_prompt for group in reward_groups):
        raise ValueError(f"each prompt must have {responses_per_prompt} responses")
    return [group_advantages(group, normalization=normalization,
                             use_leave_one_out_baseline=use_leave_one_out_baseline)
            for group in reward_groups]


def leave_one_out_advantages(rewards: Sequence[float]) -> list[float]:
    """Compute normalized leave-one-out advantages for one complete group.

    Both baseline and sample standard deviation exclude the current response.
    For eight responses, the other seven rewards have sample variance
    ``(mean(r**2) - mean(r)**2) * 7 / 6``. Zero standard deviation leaves the
    raw advantage unchanged. Advantages are then clipped to +/-50.

    Include all responses here, even samples excluded later from policy loss.
    This is intentionally distinct from whole-group GRPO normalization.
    """

    values = [_finite(reward, "reward") for reward in rewards]
    if len(values) < 3:
        raise ValueError("leave-one-out sample variance needs at least three responses")
    result = []
    for index, value in enumerate(values):
        others = values[:index] + values[index + 1:]
        count = len(others)
        mean = _finite(math.fsum(others) / count, "leave-one-out reward baseline")
        mean_square = _finite(math.fsum(reward * reward for reward in others) / count,
                              "leave-one-out reward second moment")
        variance = _finite((mean_square - mean * mean) * count / (count - 1),
                           "leave-one-out reward variance")
        # Clamp negative roundoff variance to zero before taking its square root.
        std = math.sqrt(max(0.0, variance))
        advantage = _finite(value - mean, "leave-one-out raw advantage")
        if std > 0:
            advantage /= std + 1e-6
        result.append(max(-50.0, min(50.0, _finite(advantage, "leave-one-out advantage"))))
    return result


def grouped_leave_one_out_advantages(
    reward_groups: Sequence[Sequence[float]],
    *,
    num_prompts: int = 16,
    responses_per_prompt: int = 8,
) -> list[list[float]]:
    _integer(num_prompts, "num_prompts", minimum=1)
    _integer(responses_per_prompt, "responses_per_prompt", minimum=3)
    if len(reward_groups) != num_prompts:
        raise ValueError(f"expected {num_prompts} prompt groups")
    if any(len(group) != responses_per_prompt for group in reward_groups):
        raise ValueError(f"each prompt must have {responses_per_prompt} responses")
    return [leave_one_out_advantages(group) for group in reward_groups]


def context_limit_reached(*, prompt_length: int, response_length: int,
                          max_seq_len: int) -> bool:
    """Check whether prompt and completion fill the total context budget.

    Sum all prompt and completion token counts and compare with max_seq_len.
    Hitting max_new_tokens is NOT enough to mask a
    response; generation-capped failures still supply negative advantages.
    Prompt length includes the expanded image tokens, not an image placeholder.
    """

    _integer(prompt_length, "prompt_length", minimum=1)
    _integer(response_length, "response_length")
    _integer(max_seq_len, "max_seq_len", minimum=1)
    total = prompt_length + response_length
    if total > max_seq_len:
        raise ValueError("sequence exceeds the total context budget")
    return total == max_seq_len


@dataclass(frozen=True)
class LearningRateSchedule:
    """Explicit optimizer-step boundaries for a warmup and decay schedule.

    ``step=0`` has ``warmup_start_lr``, ``step=warmup_steps`` has ``peak_lr``,
    and ``step=decay_end_step`` has ``min_lr`` for linear/cosine decay.
    ``learning_rates`` requires a first step, so zero-based and one-based SDK
    update conventions cannot be silently interchanged. Configuration must
    establish decay shape, endpoints, and warmup start.
    """

    warmup_steps: int
    decay_end_step: int
    peak_lr: float
    min_lr: float
    warmup_start_lr: float
    decay: Decay

    def __post_init__(self) -> None:
        _integer(self.warmup_steps, "warmup_steps")
        _integer(self.decay_end_step, "decay_end_step", minimum=1)
        if self.decay_end_step <= self.warmup_steps:
            raise ValueError("decay_end_step must exceed warmup_steps")
        for name in ("peak_lr", "min_lr", "warmup_start_lr"):
            value = _finite(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.min_lr > self.peak_lr or self.warmup_start_lr > self.peak_lr:
            raise ValueError("minimum and warmup-start rates cannot exceed peak_lr")
        if self.decay not in {"constant", "linear", "cosine"}:
            raise ValueError("unsupported learning-rate decay")

    def at(self, step: int) -> float:
        _integer(step, "step")
        if self.warmup_steps and step <= self.warmup_steps:
            fraction = step / self.warmup_steps
            return self.warmup_start_lr + fraction * (self.peak_lr - self.warmup_start_lr)
        if self.decay == "constant":
            return self.peak_lr
        fraction = min(1.0, max(0.0, (step - self.warmup_steps) /
                               (self.decay_end_step - self.warmup_steps)))
        multiplier = 1.0 - fraction if self.decay == "linear" else (
            0.5 * (1.0 + math.cos(math.pi * fraction)))
        return self.min_lr + multiplier * (self.peak_lr - self.min_lr)


def learning_rates(
    schedule: LearningRateSchedule,
    *,
    num_steps: int = 50,
    first_step: int,
) -> list[float]:
    """Materialize rates using the caller's explicit update-step convention."""

    _integer(num_steps, "num_steps")
    _integer(first_step, "first_step")
    return [schedule.at(step) for step in range(first_step, first_step + num_steps)]


def warmup_learning_rates(num_steps: int = 50) -> list[float]:
    """Constant 3e-6 LR after ten updates of warmup beginning at 3e-8.

    Update one uses scheduler step zero; update eleven uses step ten. The
    configured minimum LR (2e-9) has no effect for this constant decay style.
    """

    schedule = LearningRateSchedule(warmup_steps=10, decay_end_step=50,
                                    peak_lr=3e-6, min_lr=2e-9,
                                    warmup_start_lr=3e-8, decay="constant")
    return learning_rates(schedule, num_steps=num_steps, first_step=0)


@dataclass(frozen=True)
class MaskedSequence:
    """Per-target arrays at full sequence positions, before causal shifting."""

    token_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    advantages: tuple[float, ...]
    old_logprobs: tuple[float, ...]
    loss_mask: tuple[int, ...]


@dataclass(frozen=True)
class CausalLossView:
    """Logit slot t predicts target token t+1; all arrays have length T-1."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    advantages: tuple[float, ...]
    old_logprobs: tuple[float, ...]
    loss_mask: tuple[int, ...]


def mask_sequence(
    token_ids: Sequence[int],
    *,
    response_start: int,
    advantage: float,
    response_logprobs: Sequence[float],
    response_end: int | None = None,
    excluded_target_positions: Iterable[int] = (),
) -> MaskedSequence:
    """Mask prompt, image, and padding targets without shifting token offsets.

    Response log probabilities correspond to response *tokens*, beginning at
    ``response_start``. Excluded positions are absolute full-sequence target
    positions, including any image tokens. Inactive positions always carry
    label -100, advantage 0, and old log probability 0.
    """

    tokens = tuple(_integer(token, "token id") for token in token_ids)
    start = _integer(response_start, "response_start")
    end = len(tokens) if response_end is None else _integer(response_end, "response_end")
    if start > end or end > len(tokens):
        raise ValueError("response offsets must satisfy 0 <= start <= end <= sequence length")
    value = _finite(advantage, "advantage")
    logprobs = tuple(_finite(prob, "response log probability") for prob in response_logprobs)
    if len(logprobs) != end - start:
        raise ValueError("response log probability count must match response token count")
    excluded = {_integer(position, "excluded target position")
                for position in excluded_target_positions}
    if any(position >= len(tokens) for position in excluded):
        raise ValueError("excluded target position exceeds sequence length")
    active = tuple(int(start <= position < end and position not in excluded)
                   for position in range(len(tokens)))
    return MaskedSequence(
        token_ids=tokens,
        target_ids=tuple(token if mask else -100 for token, mask in zip(tokens, active)),
        advantages=tuple(value if mask else 0.0 for mask in active),
        old_logprobs=tuple(logprobs[position - start] if mask else 0.0
                          for position, mask in enumerate(active)),
        loss_mask=active,
    )


def causal_loss_view(sequence: MaskedSequence) -> CausalLossView:
    """Align each next-token target with the preceding token's logit slot."""

    lengths = {len(sequence.token_ids), len(sequence.target_ids),
               len(sequence.advantages), len(sequence.old_logprobs), len(sequence.loss_mask)}
    if len(lengths) != 1:
        raise ValueError("masked sequence arrays must have the same length")
    if sequence.loss_mask and sequence.loss_mask[0]:
        raise ValueError("the first token has no preceding causal logit slot")
    return CausalLossView(
        input_ids=sequence.token_ids[:-1],
        target_ids=sequence.target_ids[1:],
        advantages=sequence.advantages[1:],
        old_logprobs=sequence.old_logprobs[1:],
        loss_mask=sequence.loss_mask[1:],
    )


@dataclass(frozen=True)
class LossNormalizationMetadata:
    """Raw counts; an SDK's loss denominator must be confirmed independently."""

    sequence_count: int
    active_sequence_count: int
    active_token_count: int
    active_tokens_by_sequence: tuple[int, ...]


def _validate_masks(masks: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    result = []
    for row in masks:
        values = tuple(row)
        if any(value not in (0, 1) for value in values):
            raise ValueError("loss masks must contain only 0 and 1")
        result.append(tuple(int(value) for value in values))
    return result


def loss_normalization_metadata(masks: Sequence[Sequence[int]]) -> LossNormalizationMetadata:
    rows = _validate_masks(masks)
    counts = tuple(sum(row) for row in rows)
    return LossNormalizationMetadata(len(counts), sum(count > 0 for count in counts),
                                     sum(counts), counts)


def reduce_masked_losses(
    losses: Sequence[Sequence[float]],
    masks: Sequence[Sequence[int]],
    *,
    reduction: LossReduction,
    include_empty_sequences: bool | None = None,
    fixed_tokens_per_sequence: int | None = None,
) -> float:
    """Explicit reductions for checking the selected SDK loss semantics.

    Token mean weights long responses more heavily. Sequence mean first
    averages each response's active tokens and then averages responses.
    Fixed-token reduction uses ``num_sequences * fixed_tokens_per_sequence``.
    A zero-active batch returns 0; inactive losses do not enter arithmetic.
    """

    rows = _validate_masks(masks)
    if len(losses) != len(rows) or any(len(loss) != len(mask)
                                     for loss, mask in zip(losses, rows)):
        raise ValueError("losses and masks must have matching shapes")
    counts = [sum(row) for row in rows]
    sums = [math.fsum(_finite(loss, "active token loss")
                      for loss, active in zip(loss_row, mask) if active)
            for loss_row, mask in zip(losses, rows)]
    if reduction == "token_mean":
        denominator = sum(counts)
        return math.fsum(sums) / denominator if denominator else 0.0
    if reduction == "sequence_mean":
        if include_empty_sequences is None:
            raise ValueError("sequence mean requires explicit empty-sequence policy")
        terms = [total / count if count else 0.0 for total, count in zip(sums, counts)
                 if count or include_empty_sequences]
        return math.fsum(terms) / len(terms) if terms else 0.0
    if reduction == "fixed_tokens":
        if fixed_tokens_per_sequence is None:
            raise ValueError("fixed-token reduction requires an explicit denominator")
        _integer(fixed_tokens_per_sequence, "fixed_tokens_per_sequence", minimum=1)
        denominator = len(rows) * fixed_tokens_per_sequence
        return math.fsum(sums) / denominator if denominator else 0.0
    raise ValueError("unsupported loss reduction")
