"""Preference-optimization losses implemented directly with PyTorch.

The functions in this module intentionally operate below trainer abstractions.
They expose completion masking, sequence log-probability aggregation, reference
correction, target margins, and the SFT anchor used by different objectives.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F


Reduction = Literal["none", "mean", "sum"]
Normalization = Literal["sum", "mean"]


@dataclass(frozen=True)
class CompletionLogProbs:
    """Aggregated log probabilities and completion-token counts."""

    values: torch.Tensor
    token_counts: torch.Tensor


@dataclass(frozen=True)
class PreferenceLossOutput:
    """Loss plus detached diagnostics useful for training logs."""

    loss: torch.Tensor
    per_example_loss: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    preference_margin: torch.Tensor


def _validate_same_shape(**tensors: torch.Tensor) -> None:
    shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Expected tensors with identical shapes, got {shapes}.")


def _validate_hyperparameters(
    beta: float,
    label_smoothing: float,
    sft_weight: float,
) -> None:
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}.")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError(
            "label_smoothing must be in [0, 0.5) so the chosen response remains preferred."
        )
    if sft_weight < 0:
        raise ValueError(f"sft_weight must be non-negative, got {sft_weight}.")


def _reduce(losses: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}.")


def _add_sft_anchor(
    losses: torch.Tensor,
    policy_chosen_logps: torch.Tensor,
    chosen_sft_logps: Optional[torch.Tensor],
    sft_weight: float,
) -> torch.Tensor:
    if sft_weight == 0.0:
        return losses

    anchor_logps = policy_chosen_logps if chosen_sft_logps is None else chosen_sft_logps
    _validate_same_shape(losses=losses, chosen_sft_logps=anchor_logps)
    return losses + sft_weight * (-anchor_logps)


def completion_logps_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: Optional[torch.Tensor] = None,
    *,
    label_pad_token_id: int = -100,
    normalization: Normalization = "sum",
) -> CompletionLogProbs:
    """Compute causal-LM log probabilities over completion tokens only.

    Args:
        logits: Unnormalized model scores with shape ``(..., sequence, vocab)``.
        labels: Token ids with shape ``(..., sequence)``. Positions equal to
            ``label_pad_token_id`` are always ignored.
        completion_mask: Optional mask with the same shape as ``labels``. Truthy
            entries mark completion tokens; prompt and padding positions should
            be false. When omitted, all non-pad label positions are included.
        label_pad_token_id: Ignore index used by the language-model labels.
        normalization: ``"sum"`` for sequence log-probability (the common DPO
            convention) or ``"mean"`` for length-normalized log-probability
            (required by the SimPO formulation and used by ORPO here).

    Returns:
        Aggregated log probabilities and the number of selected completion tokens.

    Raises:
        ValueError: If shapes are inconsistent or an example contains no selected
            completion token after the causal shift.
    """

    if logits.ndim < 3:
        raise ValueError("logits must have shape (..., sequence, vocab).")
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} is incompatible with labels "
            f"shape {tuple(labels.shape)}."
        )
    if logits.shape[-2] < 2:
        raise ValueError("At least two sequence positions are required for causal shifting.")
    if completion_mask is not None and completion_mask.shape != labels.shape:
        raise ValueError(
            f"completion_mask shape {tuple(completion_mask.shape)} must match labels "
            f"shape {tuple(labels.shape)}."
        )
    if normalization not in {"sum", "mean"}:
        raise ValueError(f"Unsupported normalization: {normalization!r}.")

    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:].clone()
    selected = shifted_labels.ne(label_pad_token_id)
    if completion_mask is not None:
        selected = selected & completion_mask[..., 1:].to(dtype=torch.bool)

    safe_labels = shifted_labels.masked_fill(~selected, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    selected_float = selected.to(dtype=token_logps.dtype)
    token_counts = selected.sum(dim=-1)
    if torch.any(token_counts == 0):
        raise ValueError("Every example must contain at least one completion token.")

    values = (token_logps * selected_float).sum(dim=-1)
    if normalization == "mean":
        values = values / token_counts.to(dtype=values.dtype)

    return CompletionLogProbs(values=values, token_counts=token_counts)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    sft_weight: float = 0.0,
    chosen_sft_logps: Optional[torch.Tensor] = None,
    reduction: Reduction = "mean",
) -> PreferenceLossOutput:
    """Compute DPO with optional conservative labels and an SFT anchor.

    DPO compares the policy preference margin with the frozen reference-policy
    margin. Sequence-summed completion log probabilities are conventional. If an
    SFT anchor is enabled, pass length-normalized ``chosen_sft_logps`` to prevent
    long responses from receiving a disproportionately large auxiliary penalty.
    """

    _validate_same_shape(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
    )
    _validate_hyperparameters(beta, label_smoothing, sft_weight)

    policy_margin = policy_chosen_logps - policy_rejected_logps
    frozen_reference_chosen = reference_chosen_logps.detach()
    frozen_reference_rejected = reference_rejected_logps.detach()
    reference_margin = frozen_reference_chosen - frozen_reference_rejected
    relative_margin = policy_margin - reference_margin
    scaled_margin = beta * relative_margin
    losses = (
        -(1.0 - label_smoothing) * F.logsigmoid(scaled_margin)
        - label_smoothing * F.logsigmoid(-scaled_margin)
    )
    losses = _add_sft_anchor(
        losses, policy_chosen_logps, chosen_sft_logps, sft_weight
    )

    return PreferenceLossOutput(
        loss=_reduce(losses, reduction),
        per_example_loss=losses,
        chosen_rewards=(
            beta * (policy_chosen_logps - frozen_reference_chosen)
        ).detach(),
        rejected_rewards=(
            beta * (policy_rejected_logps - frozen_reference_rejected)
        ).detach(),
        preference_margin=relative_margin.detach(),
    )


def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    *,
    beta: float = 2.0,
    gamma_beta_ratio: float = 0.5,
    label_smoothing: float = 0.0,
    sft_weight: float = 0.0,
    chosen_sft_logps: Optional[torch.Tensor] = None,
    reduction: Reduction = "mean",
) -> PreferenceLossOutput:
    """Compute reference-free SimPO with a target reward margin.

    Inputs should be completion-length-normalized log probabilities. The target
    margin is represented as ``gamma / beta`` following the public SimPO trainer.
    An optional SFT anchor can preserve chosen-response likelihood during strong
    preference optimization.
    """

    _validate_same_shape(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
    )
    _validate_hyperparameters(beta, label_smoothing, sft_weight)
    if gamma_beta_ratio < 0:
        raise ValueError(
            f"gamma_beta_ratio must be non-negative, got {gamma_beta_ratio}."
        )

    policy_margin = policy_chosen_logps - policy_rejected_logps
    target_adjusted_margin = policy_margin - gamma_beta_ratio
    scaled_margin = beta * target_adjusted_margin
    losses = (
        -(1.0 - label_smoothing) * F.logsigmoid(scaled_margin)
        - label_smoothing * F.logsigmoid(-scaled_margin)
    )
    losses = _add_sft_anchor(
        losses, policy_chosen_logps, chosen_sft_logps, sft_weight
    )

    return PreferenceLossOutput(
        loss=_reduce(losses, reduction),
        per_example_loss=losses,
        chosen_rewards=(beta * policy_chosen_logps).detach(),
        rejected_rewards=(beta * policy_rejected_logps).detach(),
        preference_margin=target_adjusted_margin.detach(),
    )


def _log1mexp(log_prob: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(log_prob)) stably for log probabilities near zero."""

    if not log_prob.is_floating_point():
        raise TypeError("log probabilities must use a floating-point dtype.")

    # Exact log probability zero would produce log(0). Clamping by machine
    # epsilon keeps ORPO diagnostics finite without changing ordinary inputs.
    upper_bound = -torch.finfo(log_prob.dtype).eps
    safe_log_prob = torch.clamp(log_prob, max=upper_bound)
    log_half = -0.6931471805599453
    return torch.where(
        safe_log_prob < log_half,
        torch.log1p(-torch.exp(safe_log_prob)),
        torch.log(-torch.expm1(safe_log_prob)),
    )


def orpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    sft_weight: float = 1.0,
    chosen_sft_logps: Optional[torch.Tensor] = None,
    reduction: Reduction = "mean",
) -> PreferenceLossOutput:
    """Compute ORPO as chosen-response NLL plus an odds-ratio preference loss.

    Inputs should be completion-length-normalized log probabilities. Standard
    ORPO uses ``sft_weight=1``; setting it to zero is exposed only for controlled
    ablations. The ``_log1mexp`` branch avoids NaNs when a log probability is very
    close to zero.
    """

    _validate_same_shape(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
    )
    _validate_hyperparameters(beta, label_smoothing=0.0, sft_weight=sft_weight)

    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        _log1mexp(policy_chosen_logps) - _log1mexp(policy_rejected_logps)
    )
    odds_ratio_losses = -F.logsigmoid(log_odds)
    anchor_logps = (
        policy_chosen_logps if chosen_sft_logps is None else chosen_sft_logps
    )
    _validate_same_shape(
        policy_chosen_logps=policy_chosen_logps, chosen_sft_logps=anchor_logps
    )
    losses = sft_weight * (-anchor_logps) + beta * odds_ratio_losses

    return PreferenceLossOutput(
        loss=_reduce(losses, reduction),
        per_example_loss=losses,
        chosen_rewards=(beta * policy_chosen_logps).detach(),
        rejected_rewards=(beta * policy_rejected_logps).detach(),
        preference_margin=log_odds.detach(),
    )
