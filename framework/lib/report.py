"""Report generation: BenchmarkResult, JSON and Markdown export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .attacks import AttackType
from .metrics import DatasetMetrics, RobustnessMetrics, compute_accuracy, compute_robustness
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

    def compute_robustness_against(
        self, baseline: DatasetResult
    ) -> None:
        """Compute robustness metrics relative to a baseline.

        Args:
            baseline: The baseline DatasetResult to compare against.
        """
        self._robustness = compute_robustness(baseline.results, self.results)


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

    def __iter__(self):
        return iter(self.models)

    def _compute_all_robustness(self) -> None:
        """Compute robustness metrics for all attacked datasets."""
        for model_result in self.models:
            # Find baseline
            baseline = None
            for ds in model_result.evaluated_datasets:
                if ds.attack is None:
                    baseline = ds
                    break

            if baseline is None:
                continue

            for ds in model_result.evaluated_datasets:
                if ds.attack is not None and ds._robustness is None:
                    ds.compute_robustness_against(baseline)

    def save(self, path: str | Path) -> None:
        """Export benchmark results to file.

        File format is determined by extension:
        - .json → structured JSON with all stats
        - .md or .txt → human-readable scientific summary

        Args:
            path: Output file path.
        """
        path = Path(path)
        # Compute robustness lazily before saving
        self._compute_all_robustness()

        suffix = path.suffix
        if suffix == ".json":
            self._save_json(str(path))
        elif suffix in (".md", ".txt"):
            self._save_report(str(path))
        else:
            # Default to JSON
            self._save_json(str(path))

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

        output = {
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
                model_data["datasets"].append(ds_data)

            output["models"].append(model_data)

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

            # Per-task accuracy for baseline
            if baseline_ds:
                lines.append("### Per-Task Accuracy (Baseline)")
                lines.append("")
                lines.append("| Task | Accuracy |")
                lines.append("|------|----------|")
                for task, acc in sorted(baseline_ds.metrics.accuracy_by_task.items()):
                    lines.append(f"| {task} | {acc:.2%} |")
                lines.append("")

            # Robustness table
            attacked_datasets = [
                ds for ds in model_result.evaluated_datasets if ds.attack is not None
            ]
            if attacked_datasets and any(ds._robustness for ds in attacked_datasets):
                lines.append("### Robustness Metrics")
                lines.append("")
                lines.append(
                    "| Attack | Accuracy Drop (Δ) | Flip Rate | Consistency |"
                )
                lines.append(
                    "|--------|-------------------|-----------|-------------|"
                )
                for ds in attacked_datasets:
                    if ds._robustness:
                        r = ds._robustness
                        attack_str = (
                            f"{ds.attack.attack_name} ({ds.attack.label})"
                            if ds.attack
                            else "—"
                        )
                        lines.append(
                            f"| {attack_str} | {r.accuracy_drop:+.2%} | "
                            f"{r.flip_rate:.2%} | {r.consistency:.2%} |"
                        )
                lines.append("")

            lines.append("---")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
