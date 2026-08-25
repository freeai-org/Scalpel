"""探测与蒸馏共享的概率指标。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class LogitMetrics:
    reference_ce: float
    candidate_ce: float
    ce_delta: float
    kl: float
    top1_agreement: float


@dataclass(slots=True)
class BoundaryKDLoss:
    """Field-weighted KL at the deleted-layer boundary."""

    total: torch.Tensor
    boundary_kl: torch.Tensor


def distribution_kl(
    candidate_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """计算 ``KL(reference || candidate)``，并乘回 ``T²`` 保持梯度尺度。"""

    candidate_log_probs = F.log_softmax(candidate_logits.float() / temperature, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits.float() / temperature, dim=-1)
    reference_probs = reference_log_probs.exp()
    token_kl = torch.sum(
        reference_probs * (reference_log_probs - candidate_log_probs),
        dim=-1,
    )
    return token_kl.mean() * temperature * temperature


def compare_logits(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> LogitMetrics:
    reference_ce = F.cross_entropy(
        reference_logits.reshape(-1, reference_logits.shape[-1]).float(),
        targets.reshape(-1),
    )
    candidate_ce = F.cross_entropy(
        candidate_logits.reshape(-1, candidate_logits.shape[-1]).float(),
        targets.reshape(-1),
    )
    agreement = (
        reference_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1)
    ).float().mean()
    kl = distribution_kl(candidate_logits, reference_logits, temperature)
    return LogitMetrics(
        reference_ce=float(reference_ce.item()),
        candidate_ce=float(candidate_ce.item()),
        ce_delta=float((candidate_ce - reference_ce).item()),
        kl=float(kl.item()),
        top1_agreement=float(agreement.item()),
    )


def normalized_hidden_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """方向上的余弦距离，避免隐藏态绝对尺度主导辅助损失。"""

    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    return (1.0 - (student * teacher).sum(dim=-1)).mean()


def normalized_jensen_shannon(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return weighted Jensen-Shannon divergence normalized to ``[0, 1]``.

    Both inputs are distributions over the same vocabulary and may have any
    leading dimensions.  ``weights`` must match those leading dimensions.
    """

    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    candidate_log_probs = F.log_softmax(candidate_logits.float(), dim=-1)
    mixture_log_probs = torch.logaddexp(
        reference_log_probs,
        candidate_log_probs,
    ) - math.log(2.0)
    reference_probs = reference_log_probs.exp()
    candidate_probs = candidate_log_probs.exp()
    token_js = 0.5 * (
        torch.sum(
            reference_probs * (reference_log_probs - mixture_log_probs),
            dim=-1,
        )
        + torch.sum(
            candidate_probs * (candidate_log_probs - mixture_log_probs),
            dim=-1,
        )
    )
    if weights is None:
        mean_js = token_js.mean()
    else:
        aligned_weights = weights.to(token_js.device, torch.float32)
        if aligned_weights.shape != token_js.shape:
            raise ValueError(
                f"weights shape {aligned_weights.shape} != token JS shape "
                f"{token_js.shape}"
            )
        mean_js = (token_js * aligned_weights).sum() / aligned_weights.sum().clamp_min(
            1e-12
        )
    return (mean_js / math.log(2.0)).clamp(0.0, 1.0)


def field_weighted_boundary_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_labels: torch.Tensor,
    target_weights: torch.Tensor,
    temperature: float = 1.0,
) -> BoundaryKDLoss:
    """Match teacher layer ``i`` with student layer ``i-1``.

    ``teacher_logits`` and ``student_logits`` must come from the same frozen
    logit lens (final norm plus LM head).  The labels are used only to mask
    non-assistant positions; recovery has no hard-label CE term.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have the same shape")
    if target_labels.shape != student_logits.shape[:-1]:
        raise ValueError("target_labels must match the logits leading dimensions")
    if target_weights.shape != target_labels.shape:
        raise ValueError("target_weights must match target_labels")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    target_weights = target_weights.to(student_logits.device, torch.float32)
    target_labels = target_labels.to(student_logits.device)
    valid_mask = target_labels != -100
    effective_weights = target_weights * valid_mask.float()
    # Give every sample equal weight, independent of response length.  This
    # preserves the batch-size-1 objective when several micro-batches are
    # combined into one dynamically padded micro-batch.
    denominator = effective_weights.sum(dim=-1).clamp_min(1.0)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    boundary_kl_per_token = torch.sum(
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs),
        dim=-1,
    ) * temperature * temperature
    boundary_kl = (
        (boundary_kl_per_token * effective_weights).sum(dim=-1) / denominator
    ).mean()
    return BoundaryKDLoss(
        total=boundary_kl,
        boundary_kl=boundary_kl,
    )
