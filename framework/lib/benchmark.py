"""Benchmark orchestration: the main entry point for running evaluations.

Coordinates dataset loading, model evaluation, partial results,
and metric computation across all (model, dataset) combinations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .attacks import AttackType
from .dataset import Dataset, load_dataset
from .partial import load_partial_results, save_partial_results
from .perturb import generate_perturbed_dataset
from .prompt import build_messages, parse_batch_response, parse_single_response
from .providers.base import BaseProvider
from .report import BenchmarkResult, DatasetResult, ModelResult
from .types import EvaluatedSample

logger = logging.getLogger('llm_verbal_framework')


@dataclass
class _RetryState:
    """In-memory retry and error tracking for a single model session.

    Tracked per (dataset_filename, sample_id).  Errors are counted per
    failed *batch attempt* (not per sample), so a batch of 3 failing once
    adds 1 to ``total_errors``.

    After ``retry_times + 1`` total failures for a given sample (i.e. the
    batch was attempted that many times and failed every time), that sample
    is skipped for the remainder of the session.

    When ``total_errors >= max_errors`` the session is aborted entirely.

    Attributes:
        retry_times: Max retries per sample (from provider).
        max_errors: Max total batch failures before abort (from provider).
        sample_attempts: (dataset_filename, sample_id) → failed attempt count.
        total_errors: Cumulative count of failed batch attempts.
        aborted: True once total_errors >= max_errors.
    """

    retry_times: int
    max_errors: int
    sample_attempts: dict[tuple[str, int], int] = field(default_factory=dict)
    total_errors: int = 0
    aborted: bool = False

    def record_failure(
        self, dataset_filename: str, sample_ids: list[int]
    ) -> None:
        """Record one failed batch attempt for every sample in *sample_ids*."""
        for sid in sample_ids:
            key = (dataset_filename, sid)
            self.sample_attempts[key] = self.sample_attempts.get(key, 0) + 1
        self.total_errors += 1
        if self.total_errors >= self.max_errors:
            self.aborted = True

    def should_skip(
        self, dataset_filename: str, sample_id: int
    ) -> bool:
        """Return True if *sample_id* has been retried too many times."""
        key = (dataset_filename, sample_id)
        return self.sample_attempts.get(key, 0) > self.retry_times


class Benchmark:
    """Main benchmark orchestrator.

    Manages dataset loading, validation, model evaluation with concurrency
    control, partial results persistence, and result aggregation.

    Args:
        baseline: Path to the baseline (unperturbed) dataset.
        attacks: List of AttackType instances specifying perturbations to apply
            or pre-computed files to load (via ``load_from``).
        models: List of provider instances to evaluate.
        concurrency: Maximum number of parallel LLM requests.
        partial_results_dir: Directory to store/load partial results.
            Relative to ``base_dir`` when set.
        base_dir: Optional base directory. When set, all output paths
            (partial results and result.save) are resolved relative to it.
            The directory is created if it does not exist.
    """

    def __init__(
        self,
        baseline: str | Path,
        attacks: list[AttackType] | None = None,
        models: list[BaseProvider] | None = None,
        concurrency: int = 4,
        partial_results_dir: str | Path = 'partial',
        base_dir: str | Path | None = None,
    ) -> None:
        self._concurrency = concurrency
        self._models = models or []

        if base_dir is not None:
            base_dir = Path(base_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            self._partial_dir = base_dir / partial_results_dir
            self._base_dir = str(base_dir)
        else:
            self._partial_dir = Path(partial_results_dir)
            self._base_dir = None

        # Load datasets
        logger.info('Loading baseline dataset: %s', baseline)
        self._baseline = load_dataset(baseline, attack=None)
        logger.info(
            'Baseline loaded: %d samples, %d tasks',
            len(self._baseline),
            len({s.task for s in self._baseline.samples}),
        )

        self._attacked: list[Dataset] = []
        for attack in attacks or []:
            logger.info(
                'Processing attack: %s (%s)', attack.attack_name, attack.label
            )
            ds = generate_perturbed_dataset(self._baseline, attack)
            self._attacked.append(ds)
            logger.info('  Prepared: %d samples', len(ds))

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
                'Evaluating model: %s (%s)', provider.display_name, provider.provider_name
            )
            model_result = ModelResult(
                model_name=provider.display_name,
                provider=provider.provider_name,
            )
            retry_state = _RetryState(
                retry_times=provider.retry_times,
                max_errors=provider.max_errors,
            )

            for dataset in self._all_datasets:
                if retry_state.aborted:
                    all_finished = False
                    break

                ds_result = await self._evaluate_dataset(
                    provider, dataset, started_at, retry_state
                )
                model_result.evaluated_datasets.append(ds_result)

                expected = len(dataset)
                completed = len(ds_result.results)
                if retry_state.aborted or completed < expected:
                    all_finished = False
                    logger.warning(
                        '  %s: %d/%d completed (incomplete)',
                        dataset.filename,
                        completed,
                        expected,
                    )

                    if retry_state.aborted:
                        logger.warning(
                            '  %s: ABORTING model — %d errors',
                            dataset.filename,
                            retry_state.total_errors,
                        )
                        break
                else:
                    logger.info(
                        '  %s: %d/%d completed (%.1f%% accuracy)',
                        dataset.filename,
                        completed,
                        expected,
                        ds_result.metrics.accuracy * 100,
                    )

            model_results.append(model_result)

        finished_at = datetime.now(timezone.utc).isoformat()

        result = BenchmarkResult(
            models=model_results,
            is_finished=all_finished,
            baseline_file=self._baseline.filename,
            started_at=started_at,
            finished_at=finished_at,
            base_dir=self._base_dir,
        )
        result._compute_all_robustness()
        return result

    async def _evaluate_dataset(
        self,
        provider: BaseProvider,
        dataset: Dataset,
        started_at: str,
        retry_state: _RetryState,
    ) -> DatasetResult:
        """Evaluate a single (model, dataset) combination.

        Loads partial results, filters out already-evaluated and
        retry-exhausted samples, evaluates remaining in batches with
        concurrency control, and persists progress.
        """
        # Load partial results (filter out any stale error markers)
        existing = load_partial_results(
            self._partial_dir,
            dataset.filename,
            provider.model_slug,
        )
        sessionless = [r for r in existing if not r.raw_response.startswith('ERROR:')]
        completed_ids = {r.sample_id for r in sessionless}
        all_results = list(sessionless)

        # Determine remaining samples — skip those already completed and
        # those whose retry budget was exhausted in THIS session.
        remaining = [
            s for s in dataset.samples
            if s.id not in completed_ids
            and not retry_state.should_skip(dataset.filename, s.id)
        ]

        if not remaining:
            logger.info('  %s: all samples already completed', dataset.filename)
            return DatasetResult(
                dataset_file=dataset.filename,
                attack=dataset.attack,
                results=all_results,
            )

        logger.info(
            '  %s: %d remaining (of %d total, %d cached)',
            dataset.filename,
            len(remaining),
            len(dataset),
            len(completed_ids),
        )

        # Group into batches
        batches: list[list] = []
        for i in range(0, len(remaining), provider.batch_size):
            batches.append(remaining[i : i + provider.batch_size])

        semaphore = asyncio.Semaphore(self._concurrency)
        batch_counter = len(completed_ids) // max(provider.batch_size, 1)

        for batch_idx, batch_samples in enumerate(batches):
            if retry_state.aborted:
                break

            batch_id = batch_counter + batch_idx
            sample_ids = [s.id for s in batch_samples]
            logger.info('    Batch %d: Evaluating samples %s', batch_id, sample_ids)

            batch_results = await self._evaluate_batch(
                provider, batch_samples, batch_id, semaphore, retry_state, dataset.filename
            )
            if batch_results is None:
                # All retries exhausted — no results to add or persist.
                # The error was already logged inside _evaluate_batch.
                continue

            all_results.extend(batch_results)

            correct = sum(1 for r in batch_results if r.correct)
            logger.info(
                '    Batch %d: Completed (%d/%d correct)',
                batch_id, correct, len(batch_samples),
            )

            # Persist ONLY successfully evaluated samples
            save_partial_results(
                partial_dir=self._partial_dir,
                dataset_filename=dataset.filename,
                model_name=provider.display_name,
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
        retry_state: _RetryState,
        dataset_filename: str,
    ) -> list[EvaluatedSample] | None:
        """Evaluate a batch of samples with retry logic.

        Returns a list of EvaluatedSample on success, or None if all
        retries were exhausted.
        """
        async with semaphore:
            messages, response_format = build_messages(samples)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    '    Batch %d: Request messages:\n%s',
                    batch_id, json.dumps(messages, ensure_ascii=False, indent=4),
                )
                logger.debug(
                    '    Batch %d: Response format:\n%s',
                    batch_id, json.dumps(response_format, ensure_ascii=False, indent=4),
                )

            sample_ids = [s.id for s in samples]

            for attempt in range(retry_state.retry_times + 1):
                start_time = time.perf_counter()
                try:
                    raw_response, prompt_tokens, completion_tokens = await provider.complete(
                        messages, response_format
                    )
                    logger.debug(
                        '    Batch %d: Raw response:\n%s', batch_id, raw_response,
                    )
                except Exception as e:
                    logger.error(
                        '    Batch %d attempt %d/%d failed: %s: %s',
                        batch_id, attempt + 1, retry_state.retry_times + 1,
                        type(e).__name__, e,
                    )
                    if attempt == retry_state.retry_times:
                        # All retries exhausted
                        retry_state.record_failure(dataset_filename, sample_ids)
                        logger.warning(
                            '    Batch %d: all %d attempts failed — skipping %d samples',
                            batch_id, retry_state.retry_times + 1, len(samples),
                        )
                        return None
                    continue

                # --- success path ---
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                timestamp = datetime.now(timezone.utc).isoformat()
                per_sample_ms = elapsed_ms / len(samples)

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
