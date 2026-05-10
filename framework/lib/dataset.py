"""Dataset loading, validation, and management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .attacks import AttackType
from .types import Sample, TaskType


@dataclass
class Dataset:
    """A loaded dataset with its associated attack metadata.

    Attributes:
        samples: The list of evaluation samples.
        source_file: The original file path this dataset was loaded from.
        attack: The attack applied to this dataset, or None for baseline.
        filename: The base filename (used for partial results naming).
    """

    samples: list[Sample]
    source_file: str
    attack: AttackType | None = None

    @property
    def filename(self) -> str:
        """Base filename of the source file, e.g. 'dataset.json'."""
        return Path(self.source_file).name

    @property
    def sample_ids(self) -> set[int]:
        """Set of all sample IDs in this dataset."""
        return {s.id for s in self.samples}

    @property
    def is_baseline(self) -> bool:
        """Whether this dataset is the baseline (no attack applied)."""
        return self.attack is None

    def filter_by_task(self, task: TaskType) -> list[Sample]:
        """Return only samples of the specified task type."""
        return [s for s in self.samples if s.task == task]

    def get_sample_by_id(self, sample_id: int) -> Sample | None:
        """Look up a sample by its ID."""
        for s in self.samples:
            if s.id == sample_id:
                return s
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        attack_str = self.attack.attack_name if self.attack else "baseline"
        return (
            f"Dataset(file={self.filename!r}, attack={attack_str}, "
            f"samples={len(self.samples)})"
        )


def load_dataset(
    path: str | Path,
    attack: AttackType | None = None,
) -> Dataset:
    """Load a dataset from a JSON file.

    The JSON file must be an array of objects, each with at minimum:
    - id (int): unique sample identifier
    - task (str): one of the TaskType values
    - question (str): the question text
    - options (list[str]): answer choices
    - answer (int): 0-based correct answer index

    Optional fields:
    - rationale (str | None): explanation for the correct answer

    Args:
        path: Path to the JSON dataset file.
        attack: The attack type applied to this dataset, or None for baseline.

    Returns:
        A validated Dataset object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the dataset fails validation.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(raw_data).__name__}")

    if len(raw_data) == 0:
        raise ValueError("Dataset is empty")

    samples: list[Sample] = []
    seen_ids: set[int] = set()
    errors: list[str] = []

    for i, item in enumerate(raw_data):
        # Validate required fields
        for field_name in ("id", "task", "question", "options", "answer"):
            if field_name not in item:
                errors.append(f"Item {i}: missing required field '{field_name}'")
                continue

        if errors:
            continue

        sample_id = item["id"]
        if sample_id in seen_ids:
            errors.append(f"Item {i}: duplicate id {sample_id}")
            continue
        seen_ids.add(sample_id)

        # Validate task type
        try:
            task = TaskType(item["task"])
        except ValueError:
            errors.append(
                f"Item {i} (id={sample_id}): unknown task type '{item['task']}'. "
                f"Valid types: {[t.value for t in TaskType]}"
            )
            continue

        # Validate options
        options = item["options"]
        if not isinstance(options, list) or len(options) < 2:
            errors.append(
                f"Item {i} (id={sample_id}): 'options' must be a list with >= 2 items"
            )
            continue

        # Validate answer index
        answer = item["answer"]
        if not isinstance(answer, int) or not (0 <= answer < len(options)):
            errors.append(
                f"Item {i} (id={sample_id}): 'answer' {answer} out of range "
                f"for {len(options)} options"
            )
            continue

        samples.append(
            Sample(
                id=sample_id,
                task=task,
                question=item["question"],
                options=tuple(options),
                answer=answer,
                rationale=item.get("rationale"),
            )
        )

    if errors:
        error_preview = "\n".join(errors[:10])
        suffix = f"\n... and {len(errors) - 10} more errors" if len(errors) > 10 else ""
        raise ValueError(
            f"Dataset validation failed with {len(errors)} error(s):\n"
            f"{error_preview}{suffix}"
        )

    return Dataset(samples=samples, source_file=str(file_path), attack=attack)


def validate_alignment(baseline: Dataset, attacked: Dataset) -> None:
    """Validate that baseline and attacked datasets have matching sample IDs.

    Both datasets must contain exactly the same set of sample IDs for
    robustness metrics (flip rate, consistency) to be computed correctly.

    Args:
        baseline: The baseline (unperturbed) dataset.
        attacked: An attacked (perturbed) dataset.

    Raises:
        ValueError: If the ID sets don't match.
    """
    baseline_ids = baseline.sample_ids
    attacked_ids = attacked.sample_ids

    missing_in_attacked = baseline_ids - attacked_ids
    extra_in_attacked = attacked_ids - baseline_ids

    errors: list[str] = []
    if missing_in_attacked:
        preview = sorted(missing_in_attacked)[:10]
        errors.append(
            f"{len(missing_in_attacked)} IDs in baseline but not in attacked "
            f"({attacked.filename}): {preview}..."
        )
    if extra_in_attacked:
        preview = sorted(extra_in_attacked)[:10]
        errors.append(
            f"{len(extra_in_attacked)} IDs in attacked ({attacked.filename}) "
            f"but not in baseline: {preview}..."
        )

    if errors:
        raise ValueError(
            f"Dataset alignment failed between '{baseline.filename}' and "
            f"'{attacked.filename}':\n" + "\n".join(errors)
        )
