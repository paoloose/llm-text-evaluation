"""Report generation: BenchmarkResult, JSON and Markdown export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .attacks import AttackType
from .metrics import (
    DatasetMetrics,
    RobustnessMetrics,
    compute_accuracy,
    compute_robustness,
    compute_robustness_per_task,
)
from .types import EvaluatedSample


@dataclass
class DatasetResult:
    """Evaluation results for a single (model, dataset) pair.

    Attributes:
        dataset_file: Filename of the evaluated dataset.
        attack: Attack applied to this dataset, or None for baseline.
        results: Per-sample evaluation results.
        metrics: Lazily computed accuracy metrics.
        robustness: Lazily computed robustness metrics (None for baseline).
    """

    dataset_file: str
    attack: AttackType | None
    results: list[EvaluatedSample]
    _metrics: DatasetMetrics | None = field(default=None, repr=False)
    _robustness: RobustnessMetrics | None = field(default=None, repr=False)
    _per_task_robustness: dict[str, RobustnessMetrics] | None = field(
        default=None, repr=False
    )
    _pairwise_robustness: dict[str, RobustnessMetrics] | None = field(
        default=None, repr=False
    )

    @property
    def metrics(self) -> DatasetMetrics:
        if self._metrics is None:
            self._metrics = compute_accuracy(self.results)
        return self._metrics

    @property
    def stats(self) -> dict:
        """Summary stats dict for quick inspection."""
        d: dict = {
            "dataset": self.dataset_file,
            "attack": (
                {"type": self.attack.attack_name, "label": self.attack.label}
                if self.attack
                else None
            ),
            "metrics": self.metrics.to_dict(),
        }
        if self._robustness is not None:
            d["robustness"] = self._robustness.to_dict()
        return d

    @property
    def attack_label(self) -> str:
        """Short label used as a JSON key for this dataset (e.g. ``"french_base"``)."""
        if self.attack is None:
            return "baseline"
        return self.attack.label or self.attack.attack_name

    def compute_robustness_against(
        self, baseline: DatasetResult
    ) -> None:
        """Compute robustness metrics relative to a baseline.

        Args:
            baseline: The baseline DatasetResult to compare against.
        """
        self._robustness = compute_robustness(baseline.results, self.results)
        self._per_task_robustness = compute_robustness_per_task(
            baseline.results, self.results
        )


@dataclass
class ModelResult:
    """All evaluation results for a single model across datasets.

    Attributes:
        model_name: Full model identifier.
        provider: Provider name (e.g., "ollama", "openrouter").
        evaluated_datasets: Results per dataset.
    """

    model_name: str
    provider: str
    evaluated_datasets: list[DatasetResult] = field(default_factory=list)

    @property
    def stats(self) -> dict:
        return {
            "model": self.model_name,
            "provider": self.provider,
            "datasets": [ds.stats for ds in self.evaluated_datasets],
        }


@dataclass
class BenchmarkResult:
    """Top-level benchmark results container.

    Attributes:
        models: Results per model.
        is_finished: Whether all (model, dataset) pairs completed.
        baseline_file: Filename of the baseline dataset.
        started_at: ISO 8601 timestamp of when the benchmark started.
        finished_at: ISO 8601 timestamp of when the benchmark finished.
    """

    models: list[ModelResult]
    is_finished: bool
    baseline_file: str = ""
    started_at: str = ""
    finished_at: str = ""
    base_dir: str | None = None

    def __iter__(self):
        return iter(self.models)

    def _compute_all_robustness(self) -> None:
        """Compute robustness metrics for all attacked datasets.

        Also computes per-task robustness and pairwise attack-vs-attack
        robustness (e.g. French↔Chinese).
        """
        for model_result in self.models:
            baseline = None
            attacked: list[DatasetResult] = []

            for ds in model_result.evaluated_datasets:
                if ds.attack is None:
                    baseline = ds
                else:
                    attacked.append(ds)

            if baseline is None:
                continue

            # -- baseline-vs-attack (existing) --
            for ds in attacked:
                if ds._robustness is None:
                    ds.compute_robustness_against(baseline)

            # -- pairwise attack-vs-attack --
            for i, ds_a in enumerate(attacked):
                if ds_a._pairwise_robustness is None:
                    ds_a._pairwise_robustness = {}
                for j, ds_b in enumerate(attacked):
                    if i == j:
                        continue
                    if ds_b.dataset_file not in ds_a._pairwise_robustness:
                        ds_a._pairwise_robustness[ds_b.dataset_file] = (
                            compute_robustness(ds_b.results, ds_a.results)
                        )

    def save(self, path: str | Path, per_sample: bool = False) -> None:
        """Export benchmark results to file.

        File format is determined by extension:
        - .json → structured JSON with all stats
        - .md or .txt → human-readable scientific summary

        If ``per_sample=True``, JSON exports include per-sample predictions
        for all (model, dataset) pairs, enabling interactive sankey, upset,
        beeswarm, and sunburst visualizations in downstream consumers.

        If ``base_dir`` is set on the BenchmarkResult and ``path`` is not
        absolute, the path is resolved relative to ``base_dir``.

        Args:
            path: Output file path.
            per_sample: Include per-sample outcomes in JSON output.
        """
        path = Path(path)
        if self.base_dir and not path.is_absolute():
            path = Path(self.base_dir) / path

        self._compute_all_robustness()

        suffix = path.suffix
        if suffix == ".json":
            if per_sample:
                self._save_json_with_samples(str(path))
            else:
                self._save_json(str(path))
        elif suffix in (".md", ".txt"):
            self._save_report(str(path))
        elif suffix == ".html":
            self._save_html(str(path))
        else:
            if per_sample:
                self._save_json_with_samples(str(path))
            else:
                self._save_json(str(path))

    def _save_html(self, path: str) -> None:
        """Export as a self-contained interactive HTML report."""
        from .report_html import build_html
        html = build_html(self)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def _save_json(self, path: str) -> None:
        """Export as structured JSON."""
        # Count total samples from first model's baseline
        total_samples = 0
        for model_result in self.models:
            for ds in model_result.evaluated_datasets:
                if ds.attack is None:
                    total_samples = len(ds.results)
                    break
            if total_samples > 0:
                break

        output: dict[str, Any] = {
            "benchmark_info": {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "is_finished": self.is_finished,
                "baseline_dataset": self.baseline_file,
                "total_samples": total_samples,
            },
            "models": [],
        }

        for model_result in self.models:
            model_data: dict[str, Any] = {
                "model": model_result.model_name,
                "provider": model_result.provider,
                "datasets": [],
            }

            for ds in model_result.evaluated_datasets:
                ds_data: dict[str, Any] = {
                    "file": ds.dataset_file,
                    "attack": (
                        {"type": ds.attack.attack_name, "label": ds.attack.label}
                        if ds.attack
                        else None
                    ),
                    "metrics": ds.metrics.to_dict(),
                    "robustness": (
                        ds._robustness.to_dict() if ds._robustness else None
                    ),
                }
                if ds._per_task_robustness:
                    ds_data["robustness_per_task"] = {
                        task: rm.to_dict()
                        for task, rm in ds._per_task_robustness.items()
                    }
                if ds._pairwise_robustness:
                    ds_data["pairwise_robustness"] = {
                        other: rm.to_dict()
                        for other, rm in ds._pairwise_robustness.items()
                    }
                model_data["datasets"].append(ds_data)

            output["models"].append(model_data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _save_json_with_samples(self, path: str) -> None:
        """Export as JSON including per-sample predictions.

        Produces a file with two top-level sections:
        ``aggregates`` — the same compact summary as ``_save_json``.
        ``per_sample`` — a dict keyed by ``sample_id`` containing the
        task, expected answer, and per-model/per-attack outcomes
        (predicted index, correctness, and latency).

        This format enables downstream interactive visualizations
        (sankey, upset, beeswarm, sunburst) directly from the JSON.
        """
        total_samples = 0
        for model_result in self.models:
            for ds in model_result.evaluated_datasets:
                if ds.attack is None:
                    total_samples = len(ds.results)
                    break
            if total_samples > 0:
                break

        # -- Build the aggregates section (same as _save_json) --
        aggregates: dict[str, Any] = {
            "benchmark_info": {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "is_finished": self.is_finished,
                "baseline_dataset": self.baseline_file,
                "total_samples": total_samples,
            },
            "models": [],
        }

        for model_result in self.models:
            model_data: dict = {
                "model": model_result.model_name,
                "provider": model_result.provider,
                "datasets": [],
            }
            for ds in model_result.evaluated_datasets:
                ds_data: dict = {
                    "file": ds.dataset_file,
                    "attack": (
                        {"type": ds.attack.attack_name, "label": ds.attack.label}
                        if ds.attack
                        else None
                    ),
                    "metrics": ds.metrics.to_dict(),
                    "robustness": (
                        ds._robustness.to_dict() if ds._robustness else None
                    ),
                }
                if ds._per_task_robustness:
                    ds_data["robustness_per_task"] = {
                        task: rm.to_dict()
                        for task, rm in ds._per_task_robustness.items()
                    }
                if ds._pairwise_robustness:
                    ds_data["pairwise_robustness"] = {
                        other: rm.to_dict()
                        for other, rm in ds._pairwise_robustness.items()
                    }
                model_data["datasets"].append(ds_data)
            aggregates["models"].append(model_data)

        # -- Build per_sample section --
        per_sample: dict[str, Any] = {}

        for model_result in self.models:
            for ds in model_result.evaluated_datasets:
                label = ds.attack_label
                for r in ds.results:
                    sid = str(r.sample_id)
                    if sid not in per_sample:
                        per_sample[sid] = {
                            "sample_id": r.sample_id,
                            "task": r.task.value,
                            "expected": r.expected,
                        }
                    sample_entry = per_sample[sid]
                    sample_entry.setdefault("models", {})
                    sample_entry["models"].setdefault(
                        model_result.model_name, {}
                    )[label] = {
                        "predicted": r.predicted,
                        "correct": r.correct,
                        "latency_ms": round(r.latency_ms, 2),
                    }

        output = {
            "benchmark_info": aggregates.pop("benchmark_info"),
            "aggregates": aggregates,
            "per_sample": per_sample,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _save_report(self, path: str) -> None:
        """Export as human-readable scientific summary."""
        lines: list[str] = []
        lines.append("# LLM Verbal Reasoning Evaluation: Results Report")
        lines.append("")
        lines.append(f"**Benchmark started:** {self.started_at}")
        lines.append(f"**Benchmark finished:** {self.finished_at}")
        lines.append(f"**Baseline dataset:** {self.baseline_file}")
        lines.append(f"**Status:** {'Completed' if self.is_finished else 'Partial'}")
        lines.append("")

        for model_result in self.models:
            lines.append(f"## Model: {model_result.model_name}")
            lines.append(f"**Provider:** {model_result.provider}")
            lines.append("")

            # Find baseline metrics
            baseline_ds = None
            for ds in model_result.evaluated_datasets:
                if ds.attack is None:
                    baseline_ds = ds
                    break

            # Accuracy summary table
            lines.append("### Accuracy Summary")
            lines.append("")
            lines.append(
                "| Dataset | Attack | Total | Correct | Failed | Accuracy | "
                "Avg Latency (ms) |"
            )
            lines.append(
                "|---------|--------|-------|---------|--------|----------|"
                "------------------|"
            )

            for ds in model_result.evaluated_datasets:
                m = ds.metrics
                attack_str = (
                    f"{ds.attack.attack_name} ({ds.attack.label})"
                    if ds.attack
                    else "—"
                )
                lines.append(
                    f"| {ds.dataset_file} | {attack_str} | "
                    f"{m.total} | {m.correct} | {m.failed} | "
                    f"{m.accuracy:.2%} | {m.avg_latency_ms:.1f} |"
                )

            lines.append("")

            # Per-task accuracy for each dataset
            for ds in model_result.evaluated_datasets:
                m = ds.metrics
                if not m.tasks:
                    continue
                attack_str = (
                    f"{ds.attack.attack_name} ({ds.attack.label})"
                    if ds.attack
                    else "baseline"
                )
                lines.append(f"### Per-Task Accuracy — {attack_str}")
                lines.append("")
                lines.append("| Task | Correct | Total | Accuracy |")
                lines.append("|------|---------|-------|----------|")
                for task, info in sorted(m.tasks.items()):
                    lines.append(
                        f"| {task} | {info['correct']:.0f} | {info['total']:.0f} | "
                        f"{info['accuracy']:.2%} |"
                    )
                lines.append("")

            # Robustness table
            attacked_datasets = [
                ds for ds in model_result.evaluated_datasets if ds.attack is not None
            ]
            if attacked_datasets and any(ds._robustness for ds in attacked_datasets):
                lines.append("### Robustness Metrics")
                lines.append("")
                lines.append(
                    "| Attack | Acc. Drop (Δ) | Flip Rate | Consistency | "
                    "Pos. Transfer | Neg. Transfer | Rank Cons. |"
                )
                lines.append(
                    "|--------|---------------|-----------|-------------|"
                    "--------------|--------------|------------|"
                )
                for ds in attacked_datasets:
                    if ds._robustness:
                        r = ds._robustness
                        attack_str = (
                            f"{ds.attack.attack_name} ({ds.attack.label})"
                            if ds.attack
                            else "—"
                        )
                        rank_str = (
                            f"{r.rank_consistency:.3f}"
                            if r.rank_consistency is not None
                            else "—"
                        )
                        lines.append(
                            f"| {attack_str} | {r.accuracy_drop:+.2%} | "
                            f"{r.flip_rate:.2%} | {r.consistency:.2%} | "
                            f"{r.positive_transfer:.2%} | {r.negative_transfer:.2%} | "
                            f"{rank_str} |"
                        )
                lines.append("")

            lines.append("---")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
