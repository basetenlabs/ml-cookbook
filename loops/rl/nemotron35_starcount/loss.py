"""Clipped PPO/TIS policy loss for Loops' client-side custom-loss callback.

Only torch is needed here. Loops datums are duck-typed and the callback never
adds unsupported fields to ``loss_fn_inputs``. Previous trainer and generation
sampler probabilities are immutable snapshots, separate from current trainer
probabilities supplied by the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Sequence

import torch


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _frozen_logprobs(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = values.detach().cpu().tolist()
    return tuple(_finite(value, name) for value in values)


@dataclass(frozen=True)
class PPORecord:
    """Frozen arrays aligned with next-token labels, not unshifted token IDs."""

    target_tokens: tuple[int, ...]
    response_offset: int
    response_length: int
    advantages: tuple[float, ...]
    previous_trainer_logprobs: tuple[float, ...]
    generation_sampler_logprobs: tuple[float, ...]
    loss_mask: tuple[int, ...]
    sample_mask: bool
    advantage: float

    @property
    def input_length(self) -> int:
        return len(self.target_tokens)

    @property
    def active_token_count(self) -> int:
        return sum(self.loss_mask)

    @property
    def valid_sequence(self) -> bool:
        # The global sequence denominator is sample_mask.sum(), independent
        # of token count. A zero-token sample contributes zero loss but counts.
        return self.sample_mask


def build_ppo_record(
    *,
    prompt_length: int,
    response_token_ids: Sequence[int],
    advantage: float,
    previous_trainer_logprobs: Sequence[float],
    generation_sampler_logprobs: Sequence[float],
    stop_reason: str | None = None,
    sample_mask: bool = True,
    excluded_response_positions: Iterable[int] = (),
) -> PPORecord:
    """Build labels for prompt + response[:-1], with first response at P-1.

    Caller-supplied log probabilities contain exactly L response-token values.
    Prompt/image labels, omitted response positions, and samples explicitly
    excluded with ``sample_mask=False`` get label -100 and zero loss inputs.
    The caller determines whether the full context limit was reached.
    ``stop_reason`` is accepted as metadata for compatibility; a generation
    cap (``'length'``) below that context limit does not exclude the sample.
    Compute group advantages using all eight rewards before this loss mask.
    """

    _integer(prompt_length, "prompt_length", minimum=1)
    if not isinstance(sample_mask, bool):
        raise ValueError("sample_mask must be a bool")
    tokens = tuple(_integer(token, "response token id") for token in response_token_ids)
    value = _finite(advantage, "advantage")
    previous = _frozen_logprobs(previous_trainer_logprobs, "previous trainer log probability")
    sampler = _frozen_logprobs(generation_sampler_logprobs, "generation sampler log probability")
    if len(previous) != len(tokens) or len(sampler) != len(tokens):
        raise ValueError("both frozen log-probability arrays must match response length")
    excluded = {_integer(position, "excluded response position")
                for position in excluded_response_positions}
    if any(position >= len(tokens) for position in excluded):
        raise ValueError("excluded response position exceeds response length")
    response_mask = tuple(int(sample_mask and position not in excluded)
                          for position in range(len(tokens)))
    prefix_length = prompt_length - 1
    zeros = (0.0,) * prefix_length
    return PPORecord(
        target_tokens=(-100,) * prefix_length + tuple(
            token if mask else -100 for token, mask in zip(tokens, response_mask)),
        response_offset=prefix_length,
        response_length=len(tokens),
        advantages=zeros + tuple(value if mask else 0.0 for mask in response_mask),
        previous_trainer_logprobs=zeros + tuple(
            prob if mask else 0.0 for prob, mask in zip(previous, response_mask)),
        generation_sampler_logprobs=zeros + tuple(
            prob if mask else 0.0 for prob, mask in zip(sampler, response_mask)),
        loss_mask=(0,) * prefix_length + response_mask,
        sample_mask=sample_mask,
        advantage=value,
    )


def valid_sequence_count(records: Sequence[PPORecord]) -> int:
    """Count valid samples including samples with zero advantage."""

    return sum(record.valid_sequence for record in records)


@dataclass(frozen=True)
class PPOLossConfig:
    ratio_min: float = 0.8
    ratio_max: float = 1.28
    tis_min: float = 0.5
    tis_max: float = 2.0
    reduction_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("ratio_min", "ratio_max", "tis_min", "tis_max", "reduction_epsilon"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not 0 < self.ratio_min <= 1 <= self.ratio_max:
            raise ValueError("PPO clipping bounds must be positive and contain 1")
        if not 0 <= self.tis_min <= self.tis_max or self.tis_max <= 0:
            raise ValueError("TIS bounds must satisfy 0 <= min <= max and max > 0")
        if self.reduction_epsilon <= 0:
            raise ValueError("reduction_epsilon must be positive")


def _validate_record(record: PPORecord) -> None:
    size = record.input_length
    if any(len(values) != size for values in (
        record.advantages, record.previous_trainer_logprobs,
        record.generation_sampler_logprobs, record.loss_mask,
    )):
        raise ValueError("PPO record arrays must have the same length")
    if any(mask not in (0, 1) for mask in record.loss_mask):
        raise ValueError("PPO record loss mask must contain only 0 and 1")
    if not record.sample_mask and record.active_token_count:
        raise ValueError("a masked sample cannot have active tokens")
    for target, mask, advantage, previous, sampler in zip(
        record.target_tokens, record.loss_mask, record.advantages,
        record.previous_trainer_logprobs, record.generation_sampler_logprobs,
    ):
        if mask:
            _integer(target, "active target token id")
            _finite(advantage, "active advantage")
            _finite(previous, "active previous trainer log probability")
            _finite(sampler, "active generation sampler log probability")
        elif target != -100 or advantage != 0 or previous != 0 or sampler != 0:
            raise ValueError("inactive record targets must be -100 with zero loss inputs")


def _datum_targets(datum: Any) -> tuple[int, ...]:
    try:
        tensor_data = datum.loss_fn_inputs["target_tokens"]
        if list(tensor_data.shape) != [len(tensor_data.data)]:
            raise ValueError("custom PPO requires one-dimensional target_tokens")
        return tuple(tensor_data.data)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("datum must contain target_tokens TensorData") from error


def make_ppo_loss(
    records: Sequence[PPORecord],
    *,
    global_valid_sequence_count: int,
    config: PPOLossConfig | None = None,
) -> Callable[[Sequence[Any], Sequence[torch.Tensor]], tuple[torch.Tensor, dict[str, float]]]:
    """Bind one submitted minibatch to the SDK custom-loss callback contract.

    Every minibatch in the update must use the *same* global valid-sequence
    count, allowing accumulated gradients to equal the full-batch loss.
    Use a token mean inside each sequence and a global sequence mean outside,
    adding 1e-8 to both denominators. There is no KL penalty.
    """

    batch = tuple(records)
    if not batch:
        raise ValueError("a PPO callback batch must contain at least one record")
    _integer(global_valid_sequence_count, "global_valid_sequence_count")
    for record in batch:
        _validate_record(record)
    local_valid_sequences = valid_sequence_count(batch)
    if global_valid_sequence_count < local_valid_sequences:
        raise ValueError("global valid-sequence count cannot be smaller than this minibatch")
    cfg = PPOLossConfig() if config is None else config

    def callback(
        datums: Sequence[Any], logprobs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if len(datums) != len(batch) or len(logprobs) != len(batch):
            raise ValueError("callback datums and logprobs must match bound PPO records")
        terms = []
        anchors = []
        metrics = {
            "valid_sequences": float(local_valid_sequences),
            "global_valid_sequences": float(global_valid_sequence_count),
            "active_tokens": 0.0,
            "zero_advantage_sequences": 0.0,
            "ratio_sum": 0.0,
            "tis_weight_sum": 0.0,
            "clipped_tokens": 0.0,
            "tis_oob_tokens": 0.0,
        }
        for datum, current_full, record in zip(datums, logprobs, batch):
            if _datum_targets(datum) != record.target_tokens:
                raise ValueError("datum target_tokens do not match bound record order/masks")
            if not isinstance(current_full, torch.Tensor) or current_full.ndim != 1:
                raise ValueError("current log probabilities must be one-dimensional torch tensors")
            if current_full.numel() != record.input_length:
                raise ValueError("current log-probability shape must match datum target_tokens")
            if not current_full.is_floating_point() or not current_full.requires_grad:
                raise ValueError("current log probabilities must be floating tensors requiring gradients")
            # Empty differentiable slices anchor every SDK tensor to autograd,
            # producing zero gradients for unused slots without reading NaNs.
            anchors.append(current_full[:0].sum())
            if not record.valid_sequence or not record.active_token_count:
                continue
            mask = torch.tensor(record.loss_mask, dtype=torch.bool, device=current_full.device)
            current = current_full[mask]
            if not bool(torch.isfinite(current).all()):
                raise ValueError("active current trainer log probabilities must be finite")
            previous = torch.tensor(record.previous_trainer_logprobs,
                                    dtype=current.dtype, device=current.device)[mask].detach()
            sampler = torch.tensor(record.generation_sampler_logprobs,
                                   dtype=current.dtype, device=current.device)[mask].detach()
            advantages = torch.tensor(record.advantages,
                                      dtype=current.dtype, device=current.device)[mask]
            ratios = torch.exp(current - previous)
            if not bool(torch.isfinite(ratios).all()):
                raise ValueError("active PPO likelihood ratios must be finite")
            clipped_ratios = torch.clamp(ratios, min=cfg.ratio_min, max=cfg.ratio_max)
            loss1 = -advantages * ratios
            loss2 = -advantages * clipped_ratios
            tis_raw = torch.exp(previous - sampler).detach()
            # Map nonfinite importance weights to zero before truncated IS clipping.
            tis_raw = torch.nan_to_num(tis_raw, nan=0.0, posinf=0.0, neginf=0.0)
            tis_weights = torch.clamp(tis_raw, min=cfg.tis_min, max=cfg.tis_max)
            token_losses = torch.maximum(loss1, loss2) * tis_weights
            sequence_loss = token_losses.sum() / (current.numel() + cfg.reduction_epsilon)
            if not bool(torch.isfinite(sequence_loss)):
                raise ValueError("PPO sequence loss must be finite")
            terms.append(sequence_loss)
            metrics["active_tokens"] += float(current.numel())
            metrics["zero_advantage_sequences"] += float(record.advantage == 0)
            metrics["ratio_sum"] += float(ratios.detach().sum())
            metrics["tis_weight_sum"] += float(tis_weights.sum())
            metrics["clipped_tokens"] += float((loss2 > loss1).sum())
            metrics["tis_oob_tokens"] += float(((tis_raw < cfg.tis_min) |
                                                (tis_raw > cfg.tis_max)).sum())
        loss = torch.stack(anchors).sum()
        if terms:
            loss = loss + torch.stack(terms).sum() / (
                global_valid_sequence_count + cfg.reduction_epsilon)
        metrics["ppo_loss"] = float(loss.detach())
        return loss, metrics

    return callback
