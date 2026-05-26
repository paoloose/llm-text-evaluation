"""Core type definitions for the LLM verbal reasoning evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskType(StrEnum):
    """Verbal reasoning task categories present in the dataset."""

    READING_COMPREHENSION = "reading_comprehension"
    SENTENCE_ELIMINATION = "sentence_elimination"
    VERBAL_SERIES = "verbal_series"
    SYNONYMS_AND_ANTONYMS = "synonyms_and_antonyms"
    SENTENCE_ORDERING = "sentence_ordering"
    ANALOGIES = "analogies"
    INCOMPLETE_SENTENCES = "incomplete_sentences"


class CrossLingualLanguage(StrEnum):
    """Supported target languages for cross-lingual perturbation."""

    ENGLISH = "english"
    FRENCH = "french"
    CHINESE = "chinese"
    ARABIC = "arabic"
    JAPANESE = "japanese"
    SWAHILI = "swahili"
    RUSSIAN = "russian"


@dataclass(frozen=True)
class Sample:
    """A single evaluation sample from the dataset.

    Attributes:
        id: Unique sample identifier, used to match across baseline/attacked datasets.
        task: The verbal reasoning task type.
        question: The question text (may include passage/context).
        options: List of answer choices.
        answer: 0-based index of the correct option.
        rationale: Optional explanation for the correct answer.
    """

    id: int
    task: TaskType
    question: str
    options: tuple[str, ...]
    answer: int
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.answer < len(self.options):
            raise ValueError(
                f"Sample {self.id}: answer index {self.answer} out of range "
                f"for {len(self.options)} options"
            )


@dataclass
class ChoiceLogprobs:
    """Per-answer-choice log probabilities.

    Maps `answer_index → logprob` for each answer choice the model
    assigned a probability to.
    """

    choice_logprobs: dict[int, float]  # answer_index → logprob


@dataclass
class EvaluatedSample:
    """Result of evaluating a single sample against a model.

    Attributes:
        sample_id: Links back to Sample.id for cross-dataset matching.
        task: Task type of the evaluated sample.
        expected: Ground-truth answer index.
        predicted: Model's predicted answer index (None if parsing failed).
        correct: Whether the prediction matches the expected answer.
        raw_response: Raw text returned by the model, for debugging.
        latency_ms: Wall-clock time for this prediction in milliseconds.
        batch_id: Identifier of the batch this sample was part of.
        timestamp: ISO 8601 timestamp of when the evaluation completed.
        logprobs: Optional per-choice log probabilities, when the provider
            supports logprob extraction.
    """

    sample_id: int
    task: TaskType
    expected: int
    predicted: int | None
    correct: bool
    raw_response: str
    latency_ms: float
    batch_id: int
    timestamp: str = ""
    logprobs: ChoiceLogprobs | None = None
