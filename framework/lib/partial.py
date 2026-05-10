"""Partial results persistence and resumption.

Saves evaluation progress after each batch so that interrupted runs
can resume without re-evaluating already-completed samples.
Uses atomic writes (temp file + rename) for crash safety.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .types import EvaluatedSample, TaskType


def _partial_filename(
    dataset_filename: str,
    model_slug: str,
) -> str:
    """Build the partial results filename.

    Pattern: partial.{dataset_filename}.{model_slug}.json
    """
    return f"partial.{dataset_filename}.{model_slug}.json"


def load_partial_results(
    partial_dir: str | Path,
    dataset_filename: str,
    model_slug: str,
) -> list[EvaluatedSample]:
    """Load previously saved partial results.

    Args:
        partial_dir: Directory containing partial result files.
        dataset_filename: Base filename of the dataset.
        model_slug: URL-safe model identifier.

    Returns:
        List of previously evaluated samples, or empty list if none exist.
    """
    fname = _partial_filename(dataset_filename, model_slug)
    fpath = Path(partial_dir) / fname

    if not fpath.exists():
        return []

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    results: list[EvaluatedSample] = []
    for item in data.get("results", []):
        try:
            results.append(
                EvaluatedSample(
                    sample_id=item["sample_id"],
                    task=TaskType(item["task"]),
                    expected=item["expected"],
                    predicted=item["predicted"],
                    correct=item["correct"],
                    raw_response=item["raw_response"],
                    latency_ms=item["latency_ms"],
                    batch_id=item["batch_id"],
                    timestamp=item.get("timestamp", ""),
                )
            )
        except (KeyError, ValueError):
            continue

    return results


def save_partial_results(
    partial_dir: str | Path,
    dataset_filename: str,
    model_name: str,
    model_slug: str,
    provider_name: str,
    results: list[EvaluatedSample],
    total_samples: int,
    started_at: str,
) -> None:
    """Save partial results to disk with atomic write.

    Args:
        partial_dir: Directory to store partial result files.
        dataset_filename: Base filename of the dataset.
        model_name: Full model name for metadata.
        model_slug: URL-safe model identifier for filename.
        provider_name: Provider name for metadata.
        results: All evaluated samples so far (including newly completed).
        total_samples: Total number of samples in the dataset.
        started_at: ISO 8601 timestamp of when evaluation started.
    """
    os.makedirs(partial_dir, exist_ok=True)

    fname = _partial_filename(dataset_filename, model_slug)
    fpath = Path(partial_dir) / fname

    data = {
        "model": model_name,
        "provider": provider_name,
        "dataset_file": dataset_filename,
        "started_at": started_at,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_samples": total_samples,
        "completed_samples": len(results),
        "results": [
            {
                "sample_id": r.sample_id,
                "task": r.task.value,
                "expected": r.expected,
                "predicted": r.predicted,
                "correct": r.correct,
                "raw_response": r.raw_response,
                "latency_ms": round(r.latency_ms, 2),
                "batch_id": r.batch_id,
                "timestamp": r.timestamp,
            }
            for r in results
        ],
    }

    # Atomic write: write to temp file in same directory, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=partial_dir, prefix=".partial_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, fpath)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
