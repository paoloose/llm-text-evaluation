"""Benchmark orchestration: the main entry point for running evaluations.

Coordinates dataset loading, model evaluation, partial results,
and metric computation across all (model, dataset) combinations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .attacks import AttackType
from .dataset import Dataset, load_dataset, validate_alignment
from .partial import load_partial_results, save_partial_results
from .prompt import build_messages, parse_batch_response, parse_single_response
from .providers.base import BaseProvider
from .report import BenchmarkResult, DatasetResult, ModelResult
from .types import EvaluatedSample

logger = logging.getLogger("llm_verbal_framework")


class Benchmark:
    """Main benchmark orchestrator.

    Manages dataset loading, validation, model evaluation with concurrency
    control, partial results persistence, and result aggregation.

    Args:
        baseline: Path to the baseline (unperturbed) dataset.
        attacked: List of (path, AttackType) tuples for attacked datasets.
        models: List of provider instances to evaluate.
        concurrency: Maximum number of parallel LLM requests.
        partial_results_dir: Directory to store/load partial results.
    """

    def __init__(
        self,
        baseline: str | Path,
        attacked: list[tuple[str | Path, AttackType]] | None = None,
        models: list[BaseProvider] | None = None,
        concurrency: int = 4,
        partial_results_dir: str | Path = ".partial",
    ) -> None:
        self._concurrency = concurrency
        self._partial_dir = partial_results_dir
        self._models = models or []

        # Load datasets
        logger.info("Loading baseline dataset: %s", baseline)
        self._baseline = load_dataset(baseline, attack=None)
        logger.info(
            "Baseline loaded: %d samples, %d tasks",
            len(self._baseline),
            len({s.task for s in self._baseline.samples}),
        )

        self._attacked: list[Dataset] = []
        for path, attack in attacked or []:
            logger.info("Loading attacked dataset: %s (%s)", path, attack.attack_name)
            ds = load_dataset(path, attack=attack)
            validate_alignment(self._baseline, ds)
            self._attacked.append(ds)
            logger.info("  Loaded and validated: %d samples", len(ds))

        self._all_datasets = [self._baseline] + self._attacked

    def run(self) -> BenchmarkResult:
        """Run the full evaluation pipeline.

        Executes all (model, dataset) combinations with concurrency control
        and partial result persistence.

        Returns:
            BenchmarkResult with all evaluation data.
        """
        return asyncio.run(self._run_async())

    async def _run_async(self) -> BenchmarkResult:
        """Async implementation of the benchmark pipeline."""
        started_at = datetime.now(timezone.utc).isoformat()
        model_results: list[ModelResult] = []
        all_finished = True

        for provider in self._models:
            logger.info(
                "Evaluating model: %s (%s)", provider.model, provider.provider_name
            )
            model_result = ModelResult(
                model_name=provider.model,
                provider=provider.provider_name,
            )

            for dataset in self._all_datasets:
                ds_result = await self._evaluate_dataset(provider, dataset, started_at)
                model_result.evaluated_datasets.append(ds_result)

                expected = len(dataset)
                completed = len(ds_result.results)
                if completed < expected:
                    all_finished = False
                    logger.warning(
                        "  %s: %d/%d completed (incomplete)",
                        dataset.filename,
                        completed,
                        expected,
                    )
                else:
                    logger.info(
                        "  %s: %d/%d completed (%.1f%% accuracy)",
                        dataset.filename,
                        completed,
                        expected,
                        ds_result.metrics.accuracy * 100,
                    )

            model_results.append(model_result)

        finished_at = datetime.now(timezone.utc).isoformat()

        return BenchmarkResult(
            models=model_results,
            is_finished=all_finished,
            baseline_file=self._baseline.filename,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def _evaluate_dataset(
        self,
        provider: BaseProvider,
        dataset: Dataset,
        started_at: str,
    ) -> DatasetResult:
        """Evaluate a single (model, dataset) combination.

        Loads partial results, determines remaining work, evaluates
        in batches with concurrency control, and persists progress.
        """
        # Load partial results
        existing = load_partial_results(
            self._partial_dir,
            dataset.filename,
            provider.model_slug,
        )
        completed_ids = {r.sample_id for r in existing}
        all_results = list(existing)

        # Determine remaining samples
        remaining = [s for s in dataset.samples if s.id not in completed_ids]

        if not remaining:
            logger.info("  %s: all samples already completed", dataset.filename)
            return DatasetResult(
                dataset_file=dataset.filename,
                attack=dataset.attack,
                results=all_results,
            )

        logger.info(
            "  %s: %d remaining (of %d total, %d cached)",
            dataset.filename,
            len(remaining),
            len(dataset),
            len(completed_ids),
        )

        # Group into batches
        batches: list[list] = []
        for i in range(0, len(remaining), provider.batch_size):
            batches.append(remaining[i : i + provider.batch_size])

        # Evaluate with concurrency control
        semaphore = asyncio.Semaphore(self._concurrency)
        batch_counter = len(completed_ids) // max(provider.batch_size, 1)

        for batch_idx, batch_samples in enumerate(batches):
            batch_id = batch_counter + batch_idx
            sample_ids = [s.id for s in batch_samples]
            logger.info("    Batch %d: Evaluating samples %s", batch_id, sample_ids)

            batch_results = await self._evaluate_batch(
                provider, batch_samples, batch_id, semaphore
            )
            all_results.extend(batch_results)

            correct = sum(1 for r in batch_results if r.correct)
            logger.info("    Batch %d: Completed (%d/%d correct)", batch_id, correct, len(batch_samples))

            # Persist partial results after each batch
            save_partial_results(
                partial_dir=self._partial_dir,
                dataset_filename=dataset.filename,
                model_name=provider.model,
                model_slug=provider.model_slug,
                provider_name=provider.provider_name,
                results=all_results,
                total_samples=len(dataset),
                started_at=started_at,
            )

        return DatasetResult(
            dataset_file=dataset.filename,
            attack=dataset.attack,
            results=all_results,
        )

    async def _evaluate_batch(
        self,
        provider: BaseProvider,
        samples: list,
        batch_id: int,
        semaphore: asyncio.Semaphore,
    ) -> list[EvaluatedSample]:
        """Evaluate a batch of samples against a provider.

        Uses the semaphore for concurrency control.
        """
        async with semaphore:
            messages, response_format = build_messages(samples)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("    Batch %d: Request messages:\n%s", batch_id, json.dumps(messages, ensure_ascii=False, indent=4))
                logger.debug("    Batch %d: Response format:\n%s", batch_id, json.dumps(response_format, ensure_ascii=False, indent=4))

            start_time = time.perf_counter()
            try:
                raw_response, prompt_tokens, completion_tokens = await provider.complete(
                    messages, response_format
                )
            except Exception as e:
                logger.error(
                    "Batch %d failed: %s: %s", batch_id, type(e).__name__, e
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                timestamp = datetime.now(timezone.utc).isoformat()

                # Mark all samples in batch as failed
                return [
                    EvaluatedSample(
                        sample_id=s.id,
                        task=s.task,
                        expected=s.answer,
                        predicted=None,
                        correct=False,
                        raw_response=f"ERROR: {type(e).__name__}: {e}",
                        latency_ms=elapsed_ms,
                        batch_id=batch_id,
                        timestamp=timestamp,
                    )
                    for s in samples
                ]

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            timestamp = datetime.now(timezone.utc).isoformat()
            per_sample_ms = elapsed_ms / len(samples)

            # Parse response
            results: list[EvaluatedSample] = []

            if len(samples) == 1:
                predicted = parse_single_response(raw_response)
                s = samples[0]
                results.append(
                    EvaluatedSample(
                        sample_id=s.id,
                        task=s.task,
                        expected=s.answer,
                        predicted=predicted,
                        correct=predicted == s.answer,
                        raw_response=raw_response,
                        latency_ms=elapsed_ms,
                        batch_id=batch_id,
                        timestamp=timestamp,
                    )
                )
            else:
                expected_ids = [s.id for s in samples]
                parsed = parse_batch_response(raw_response, expected_ids)
                for s in samples:
                    predicted = parsed.get(s.id)
                    results.append(
                        EvaluatedSample(
                            sample_id=s.id,
                            task=s.task,
                            expected=s.answer,
                            predicted=predicted,
                            correct=predicted == s.answer,
                            raw_response=raw_response,
                            latency_ms=per_sample_ms,
                            batch_id=batch_id,
                            timestamp=timestamp,
                        )
                    )

            return results
