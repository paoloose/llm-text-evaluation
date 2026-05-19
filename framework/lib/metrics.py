"""Metrics computation for accuracy, robustness, and comparative analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from .types import EvaluatedSample, TaskType


@dataclass
class DatasetMetrics:
    """Aggregated accuracy metrics for a single dataset evaluation.

    Attributes:
        total: Total number of samples evaluated.
        correct: Number of correctly predicted samples.
        failed: Number of samples where parsing failed (predicted is None).
        accuracy: Overall accuracy (correct / total).
        tasks: Per-task breakdown — ``{task_name: {correct, total, accuracy}}``.
        avg_latency_ms: Average prediction latency in milliseconds.
        total_time_s: Total wall-clock evaluation time in seconds.
    """

    total: int
    correct: int
    failed: int
    accuracy: float
    tasks: dict[str, dict[str, float]]
    avg_latency_ms: float
    total_time_s: float

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "failed": self.failed,
            "accuracy": round(self.accuracy, 4),
            "tasks": {
                k: {sk: round(sv, 4) for sk, sv in v.items()}
                for k, v in self.tasks.items()
            },
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_time_s": round(self.total_time_s, 2),
        }


@dataclass
class RobustnessMetrics:
    """Comparative robustness metrics between baseline and attacked results.

    Attributes:
        accuracy_drop: Accuracy_baseline - Accuracy_attacked (positive = worse).
        flip_rate: Fraction of baseline-correct samples that became incorrect.
        consistency: Fraction of samples where the prediction didn't change.
        positive_transfer: Fraction of baseline-correct samples that remain
            correct under attack.
        negative_transfer: Fraction of baseline-incorrect samples that produce
            the *same* wrong answer under attack.
        rank_consistency: Spearman rank correlation between answer-rank vectors
            (when logprobs are available).  None otherwise.
    """

    accuracy_drop: float
    flip_rate: float
    consistency: float
    positive_transfer: float
    negative_transfer: float
    rank_consistency: float | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "accuracy_drop": round(self.accuracy_drop, 4),
            "flip_rate": round(self.flip_rate, 4),
            "consistency": round(self.consistency, 4),
            "positive_transfer": round(self.positive_transfer, 4),
            "negative_transfer": round(self.negative_transfer, 4),
        }
        if self.rank_consistency is not None:
            d["rank_consistency"] = round(self.rank_consistency, 4)
        return d


def compute_accuracy(results: list[EvaluatedSample]) -> DatasetMetrics:
    """Compute accuracy metrics from a list of evaluated samples.

    Args:
        results: List of EvaluatedSample objects from a single dataset run.

    Returns:
        DatasetMetrics with overall and per-task accuracy.
    """
    if not results:
        return DatasetMetrics(
            total=0,
            correct=0,
            failed=0,
            accuracy=0.0,
            tasks={},
            avg_latency_ms=0.0,
            total_time_s=0.0,
        )

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    failed = sum(1 for r in results if r.predicted is None)

    # Per-task stats
    task_correct: dict[str, int] = defaultdict(int)
    task_total: dict[str, int] = defaultdict(int)
    for r in results:
        task_total[r.task.value] += 1
        if r.correct:
            task_correct[r.task.value] += 1

    tasks = {
        task: {
            "correct": task_correct.get(task, 0),
            "total": task_total[task],
            "accuracy": task_correct.get(task, 0) / task_total[task] if task_total[task] > 0 else 0.0,
        }
        for task in task_total
    }

    # Timing
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    total_time = sum(latencies) / 1000.0 if latencies else 0.0

    return DatasetMetrics(
        total=total,
        correct=correct,
        failed=failed,
        accuracy=correct / total if total > 0 else 0.0,
        tasks=tasks,
        avg_latency_ms=avg_latency,
        total_time_s=total_time,
    )


def _spearman_rho(
    x: list[float],
    y: list[float],
) -> float | None:
    """Spearman rank correlation between two vectors.

    Returns None if either vector is constant (no variation).
    """
    n = len(x)
    if n < 3:
        return None

    def ranks(arr: list[float]) -> list[float]:
        indexed = sorted(enumerate(arr), key=lambda p: p[1])
        result = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and indexed[j][1] == indexed[i][1]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                result[indexed[k][0]] = avg
            i = j
        return result

    rx = ranks(x)
    ry = ranks(y)

    mx = mean(rx)
    my = mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx)
        * sum((b - my) ** 2 for b in ry)
    ) ** 0.5

    if den == 0:
        return None
    return num / den


def _build_rank_vector(
    results: dict[int, EvaluatedSample],
    sample_ids: list[int],
) -> list[float] | None:
    """Build an answer-rank vector from per-sample logprobs.

    For each sample, ranks answer choices by their logprob (highest first).

    Returns None if any sample lacks logprobs (mixed availability would
    distort Spearman).
    """
    ranks: list[float] = []

    for sid in sample_ids:
        r = results[sid]
        probs = r.logprobs
        if not (probs and probs.choice_logprobs):
            return None
        sorted_indices = sorted(probs.choice_logprobs, key=lambda k: probs.choice_logprobs[k], reverse=True)
        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        if r.predicted is not None:
            ranks.append(rank_map.get(r.predicted, float(len(sorted_indices) + 1)))
        else:
            ranks.append(float(len(sorted_indices) + 1))

    return ranks


def compute_robustness(
    baseline_results: list[EvaluatedSample],
    attacked_results: list[EvaluatedSample],
) -> RobustnessMetrics:
    """Compute robustness metrics comparing baseline vs. attacked results.

    Matches samples by their sample_id. Requires that both lists contain
    results for the same set of sample IDs.

    Args:
        baseline_results: Results from the baseline (unperturbed) dataset.
        attacked_results: Results from the attacked (perturbed) dataset.

    Returns:
        RobustnessMetrics with accuracy drop, flip rate, consistency,
        positive/negative transfer, and rank consistency (if logprobs available).
    """
    baseline_map = {r.sample_id: r for r in baseline_results}
    attacked_map = {r.sample_id: r for r in attacked_results}

    common_ids = set(baseline_map) & set(attacked_map)

    if not common_ids:
        return RobustnessMetrics(
            accuracy_drop=0.0,
            flip_rate=0.0,
            consistency=0.0,
            positive_transfer=0.0,
            negative_transfer=0.0,
        )

    n = len(common_ids)

    # Accuracy drop
    baseline_correct = sum(
        1 for sid in common_ids if baseline_map[sid].correct
    )
    attacked_correct = sum(
        1 for sid in common_ids if attacked_map[sid].correct
    )

    baseline_acc = baseline_correct / n
    attacked_acc = attacked_correct / n
    accuracy_drop = baseline_acc - attacked_acc

    # Flip rate: baseline correct → attacked incorrect
    flips = sum(
        1
        for sid in common_ids
        if baseline_map[sid].correct and not attacked_map[sid].correct
    )
    flip_rate = flips / baseline_correct if baseline_correct > 0 else 0.0

    # Consistency: same prediction regardless of correctness
    consistent = sum(
        1
        for sid in common_ids
        if baseline_map[sid].predicted == attacked_map[sid].predicted
    )
    consistency = consistent / n

    # --- Positive transfer ---
    # |F_baseline ∩ F_attacked| / |F_baseline|
    pos_transfer = sum(
        1
        for sid in common_ids
        if baseline_map[sid].correct and attacked_map[sid].correct
    )
    positive_transfer = (
        pos_transfer / baseline_correct if baseline_correct > 0 else 0.0
    )

    # --- Negative transfer ---
    # |¬F_baseline ∩ ¬F_attacked_same| / |¬F_baseline|
    baseline_incorrect = n - baseline_correct
    neg_same = sum(
        1
        for sid in common_ids
        if not baseline_map[sid].correct
        and not attacked_map[sid].correct
        and baseline_map[sid].predicted == attacked_map[sid].predicted
    )
    negative_transfer = (
        neg_same / baseline_incorrect if baseline_incorrect > 0 else 0.0
    )

    # --- Rank consistency (Spearman via logprobs) ---
    sorted_ids = sorted(common_ids)
    baseline_ranks = _build_rank_vector(baseline_map, sorted_ids)
    attacked_ranks = _build_rank_vector(attacked_map, sorted_ids)
    rank_consistency = (
        _spearman_rho(baseline_ranks, attacked_ranks)
        if baseline_ranks and attacked_ranks
        else None
    )

    return RobustnessMetrics(
        accuracy_drop=accuracy_drop,
        flip_rate=flip_rate,
        consistency=consistency,
        positive_transfer=positive_transfer,
        negative_transfer=negative_transfer,
        rank_consistency=rank_consistency,
    )
