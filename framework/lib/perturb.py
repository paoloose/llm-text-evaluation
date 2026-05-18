"""Perturbation pipeline stage.

Produces perturbed (attacked) datasets from a baseline by either loading a
pre-computed file (when ``load_from`` is set on the attack) or applying the
perturbation strategy on-the-fly with progress persistence for resumption.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .attacks import AttackType
from .dataset import Dataset, load_dataset, validate_alignment
from .errors import log_error
from .types import Sample, TaskType

logger = logging.getLogger("llm_verbal_framework")


async def generate_perturbed_dataset(
    baseline: Dataset,
    attack: AttackType,
    partial_dir: str | Path,
    session_id: str,
) -> Dataset:
    """Produce a perturbed dataset from a baseline.

    If ``attack.load_from`` is set the dataset is loaded directly and validated
    for structural alignment (same IDs, answers, option counts).  Otherwise the
    attack's ``perturb()`` is called asynchronously with progress persisted to
    ``{partial_dir}/perturbations/{attack_name}/{baseline}.{label}.json`` so
    that interrupted perturbations can resume.

    Args:
        baseline: The baseline (unperturbed) dataset.
        attack: The attack/perturbation instance.
        partial_dir: Root directory for partial result files.
        session_id: Benchmark session identifier for error logging.

    Returns:
        A validated Dataset tagged with the attack metadata.

    Raises:
        FileNotFoundError: If ``load_from`` points to a non-existent file.
        NotImplementedError: If ``load_from`` is not set and the attack does
            not implement ``perturb()``.
        ValueError: If the loaded dataset fails structural validation or the
            perturbed dataset fails ID alignment.
    """
    if attack.load_from:
        logger.info(
            "Loading perturbed dataset from: %s (%s)", attack.load_from, attack.attack_name
        )
        ds = load_dataset(attack.load_from, attack=attack)
        _validate_loaded_structure(baseline, ds)
        return ds

    label = attack.label or "default"
    perturb_path = (
        Path(partial_dir) / "perturbations" / attack.attack_name
        / f"{label}.json"
    )

    existing = _load_perturbation_partial(perturb_path)
    existing_ids = {s.id for s in existing}
    remaining = [s for s in baseline.samples if s.id not in existing_ids]

    if not remaining:
        logger.info(
            "Perturbation '%s/%s': all %d samples already cached",
            attack.attack_name, label, len(existing),
        )
    else:
        logger.info(
            "Perturbation '%s/%s': %d cached, %d remaining",
            attack.attack_name, label, len(existing), len(remaining),
        )
        try:
            new_samples = await attack.perturb(remaining)
        except Exception as exc:
            log_error(
                partial_dir,
                session_id,
                phase="perturbation",
                error_type="translation_failed",
                provider=getattr(getattr(attack, "model", None), "provider_name", ""),
                model=getattr(getattr(attack, "model", None), "display_name", ""),
                dataset=baseline.filename,
                attack_type=attack.attack_name,
                attack_label=attack.label or "",
                sample_ids=[s.id for s in remaining],
                exception=exc,
            )
            raise
        existing.extend(new_samples)
        _save_perturbation_partial(perturb_path, attack, existing, len(baseline))

    source = f"{attack.attack_name}.{label}.json"
    ds = Dataset(samples=existing, source_file=source, attack=attack)
    validate_alignment(baseline, ds)
    logger.info("Perturbed dataset prepared and validated: %d samples", len(ds))
    return ds


def _validate_loaded_structure(baseline: Dataset, loaded: Dataset) -> None:
    """Statically validate a loaded dataset against the baseline.

    Checks that:
    - Sample IDs are identical.
    - Each sample has the same correct answer index.
    - Each sample has the same number of options.

    Does NOT perform any semantic/intelligent validation.
    """
    baseline_map = {s.id: s for s in baseline.samples}
    loaded_map = {s.id: s for s in loaded.samples}

    errors: list[str] = []

    missing = set(baseline_map) - set(loaded_map)
    extra = set(loaded_map) - set(baseline_map)

    if missing:
        preview = sorted(missing)[:10]
        errors.append(
            f"{len(missing)} IDs in baseline but missing from loaded dataset: "
            f"{preview}..."
        )
    if extra:
        preview = sorted(extra)[:10]
        errors.append(
            f"{len(extra)} IDs in loaded dataset but not in baseline: "
            f"{preview}..."
        )

    for sid in set(baseline_map) & set(loaded_map):
        b = baseline_map[sid]
        l = loaded_map[sid]
        if b.answer != l.answer:
            errors.append(
                f"Sample {sid}: answer mismatch (baseline={b.answer}, "
                f"loaded={l.answer})"
            )
        if len(b.options) != len(l.options):
            errors.append(
                f"Sample {sid}: option count mismatch "
                f"(baseline={len(b.options)}, loaded={len(l.options)})"
            )

    if errors:
        error_preview = "\n".join(errors[:10])
        suffix = f"\n... and {len(errors) - 10} more errors" if len(errors) > 10 else ""
        raise ValueError(
            f"Loaded dataset '{loaded.filename}' does not match baseline "
            f"'{baseline.filename}':\n{error_preview}{suffix}"
        )


def _load_perturbation_partial(path: Path) -> list[Sample]:
    """Load previously cached perturbation results.

    Returns an empty list if no cache file exists or if the file is corrupt.
    """
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    samples: list[Sample] = []
    raw_list = data.get("samples", data) if isinstance(data, dict) else data
    if not isinstance(raw_list, list):
        return []

    for item in raw_list:
        try:
            samples.append(
                Sample(
                    id=item["id"],
                    task=TaskType(item["task"]),
                    question=item["question"],
                    options=tuple(item["options"]),
                    answer=item["answer"],
                    rationale=item.get("rationale"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    return samples


def _save_perturbation_partial(
    path: Path,
    attack: AttackType,
    samples: list[Sample],
    total_samples: int,
) -> None:
    """Save perturbation progress atomically.

    Preserves the original ``started_at`` timestamp from any existing partial
    file to avoid resetting it on every incremental save.
    """
    os.makedirs(path.parent, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    started_at = now
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict):
                started_at = prev.get("started_at", now)
        except (json.JSONDecodeError, OSError):
            pass

    data = {
        "attack_type": attack.attack_name,
        "attack_label": attack.label,
        "started_at": started_at,
        "last_updated": now,
        "total_samples": total_samples,
        "completed_samples": len(samples),
        "samples": [
            {
                "id": s.id,
                "task": s.task.value,
                "question": s.question,
                "options": list(s.options),
                "answer": s.answer,
                "rationale": s.rationale,
            }
            for s in samples
        ],
    }

    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".perturb_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
