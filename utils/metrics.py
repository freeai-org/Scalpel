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
class WeightedKDLoss:
    """最终 logits 上的字段加权 CE 与 teacher-student KL。"""

    total: torch.Tensor
    hard_ce: torch.Tensor
    soft_kl: torch.Tensor


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


def field_weighted_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_labels: torch.Tensor,
    target_weights: torch.Tensor,
    temperature: float = 1.0,
) -> WeightedKDLoss:
    """计算最终 token 分布上的字段加权恢复目标。

    ``target_labels`` 里值为 ``-100`` 的位置会被忽略；其它位置同时计算
    hard-label CE 和 ``KL(P_teacher || P_student)``。每条样本先按自己的
    有效权重归一化，再对 batch 求平均，避免长回答压过短回答。
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
    vocab_size = student_logits.shape[-1]
    hard_per_token = F.cross_entropy(
        student_logits.reshape(-1, vocab_size),
        target_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(target_labels)
    hard_loss = ((hard_per_token * effective_weights).sum(dim=-1) / denominator).mean()

    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    soft_per_token = torch.sum(
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs),
        dim=-1,
    ) * temperature * temperature
    soft_loss = ((soft_per_token * effective_weights).sum(dim=-1) / denominator).mean()
    return WeightedKDLoss(
        total=hard_loss + soft_loss,
        hard_ce=hard_loss,
        soft_kl=soft_loss,
    )


class _ChunkedWeightedKDLoss(torch.autograd.Function):
    """Exact CE+KL with bounded temporary memory over the token dimension.

    Qwen3-VL has a large vocabulary.  Materializing float32 teacher/student
    probabilities for every supervised token at once can consume several GiB.
    The forward and backward formulas are separable by token, so process small
    token blocks while retaining only the native-dtype logits required for the
    student gradient.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target_labels: torch.Tensor,
        target_weights: torch.Tensor,
        temperature: float,
        token_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, vocab_size = student_logits.shape
        valid_mask = target_labels != -100
        effective_weights = target_weights.float() * valid_mask.float()
        denominator = effective_weights.sum(dim=-1).clamp_min(1.0)
        normalized_weights = effective_weights / denominator.unsqueeze(-1)
        normalized_weights = normalized_weights / batch_size

        hard_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        soft_loss = torch.zeros_like(hard_loss)
        for start in range(0, token_count, token_chunk_size):
            stop = min(start + token_chunk_size, token_count)
            student = student_logits[:, start:stop].float()
            teacher = teacher_logits[:, start:stop].float()
            labels = target_labels[:, start:stop]
            weights = normalized_weights[:, start:stop]

            hard_per_token = F.cross_entropy(
                student.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(labels)
            hard_loss.add_((hard_per_token * weights).sum())

            teacher_log_probs = F.log_softmax(teacher / temperature, dim=-1)
            student_log_probs = F.log_softmax(student / temperature, dim=-1)
            soft_per_token = torch.sum(
                teacher_log_probs.exp()
                * (teacher_log_probs - student_log_probs),
                dim=-1,
            ) * temperature * temperature
            soft_loss.add_((soft_per_token * weights).sum())

        ctx.save_for_backward(
            student_logits,
            teacher_logits,
            target_labels,
            normalized_weights,
        )
        ctx.temperature = temperature
        ctx.token_chunk_size = token_chunk_size
        return hard_loss, soft_loss

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_hard: torch.Tensor,
        grad_soft: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        student_logits, teacher_logits, target_labels, normalized_weights = (
            ctx.saved_tensors
        )
        temperature = ctx.temperature
        token_chunk_size = ctx.token_chunk_size
        token_count = student_logits.shape[1]
        grad_student = torch.empty_like(student_logits)

        for start in range(0, token_count, token_chunk_size):
            stop = min(start + token_chunk_size, token_count)
            student = student_logits[:, start:stop].float()
            teacher = teacher_logits[:, start:stop].float()
            labels = target_labels[:, start:stop]
            weights = normalized_weights[:, start:stop].unsqueeze(-1)

            student_probs = F.softmax(student / temperature, dim=-1)
            hard_grad = F.softmax(student, dim=-1)
            valid_mask = labels != -100
            safe_labels = labels.masked_fill(~valid_mask, 0)
            hard_grad.scatter_add_(
                dim=-1,
                index=safe_labels.unsqueeze(-1),
                src=-valid_mask.to(hard_grad.dtype).unsqueeze(-1),
            )
            teacher_probs = F.softmax(teacher / temperature, dim=-1)
            soft_grad = (student_probs - teacher_probs) * temperature
            chunk_grad = weights * (
                hard_grad * grad_hard.float()
                + soft_grad * grad_soft.float()
            )
            grad_student[:, start:stop].copy_(chunk_grad)

        return grad_student, None, None, None, None, None


def memory_efficient_field_weighted_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_labels: torch.Tensor,
    target_weights: torch.Tensor,
    temperature: float = 1.0,
    token_chunk_size: int = 4,
) -> WeightedKDLoss:
    """Mathematically equivalent, token-chunked variant of the KD objective."""

    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape [batch, tokens, vocabulary]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have the same shape")
    if target_labels.shape != student_logits.shape[:-1]:
        raise ValueError("target_labels must match the logits leading dimensions")
    if target_weights.shape != target_labels.shape:
        raise ValueError("target_weights must match target_labels")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    if teacher_logits.requires_grad:
        raise ValueError("teacher_logits must be detached")

    target_labels = target_labels.to(student_logits.device)
    target_weights = target_weights.to(student_logits.device, torch.float32)
    hard_loss, soft_loss = _ChunkedWeightedKDLoss.apply(
        student_logits,
        teacher_logits,
        target_labels,
        target_weights,
        float(temperature),
        int(token_chunk_size),
    )
    return WeightedKDLoss(
        total=hard_loss + soft_loss,
        hard_ce=hard_loss,
        soft_kl=soft_loss,
    )
