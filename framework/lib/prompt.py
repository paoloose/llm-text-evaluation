"""Prompt engineering for LLM verbal reasoning evaluation.

Handles:
- Building system and user messages per task type
- Constructing JSON schema for structured output (single and batch)
- Parsing model responses with fallback strategies
"""

from __future__ import annotations

import json
import re

from .types import CrossLingualLanguage, Sample, TaskType


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert evaluator for verbal reasoning tasks in Spanish.
You will receive questions with numbered answer options (0-based index).
Analyze each question carefully and select the BEST answer.

TASK-SPECIFIC GUIDELINES:
- Reading comprehension: Focus on what the text explicitly states or strongly implies.
- Sentence ordering (plan de redacción): Find the logical sequence that creates a coherent, well-structured text.
- Sentence elimination: Identify the sentence that does NOT belong thematically or logically.
- Verbal series: Identify the pattern connecting the words (synonyms, antonyms, categories, relationships).
- Analogies: Match the underlying relationship between the given pair of concepts.
- Synonyms and antonyms: Select the word with the closest or most opposite meaning in context.
- Incomplete sentences: Choose the option that best completes the sentence's meaning and grammar.

RULES:
- Consider the context, question, and ALL options before deciding.
- Your response must be valid JSON matching the required schema.
- Provide ONLY the answer index, no explanations."""


# ── JSON schemas ───────────────────────────────────────────────────────────────

SINGLE_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "single_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "integer",
                    "description": "0-based index of the correct option",
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}

BATCH_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "batch_answers",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Sample ID from the question",
                            },
                            "answer": {
                                "type": "integer",
                                "description": "0-based index of the correct option",
                            },
                        },
                        "required": ["id", "answer"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["answers"],
            "additionalProperties": False,
        },
    },
}


# ── Prompt building ────────────────────────────────────────────────────────────


def _format_options(options: tuple[str, ...]) -> str:
    """Format options as numbered list."""
    return "\n".join(f"{i}) {opt}" for i, opt in enumerate(options))


def build_single_prompt(sample: Sample) -> str:
    """Build user message for a single sample.

    Args:
        sample: The sample to format as a prompt.

    Returns:
        Formatted user message string.
    """
    return (
        f"Question (id: {sample.id}, type: {sample.task.value}):\n"
        f"{sample.question}\n\n"
        f"Options:\n{_format_options(sample.options)}"
    )


def build_batch_prompt(samples: list[Sample]) -> str:
    """Build user message for a batch of samples.

    Args:
        samples: List of samples to format as a single prompt.

    Returns:
        Formatted user message string with all questions.
    """
    parts: list[str] = ["Answer each of the following questions:\n"]

    for i, sample in enumerate(samples, 1):
        parts.append(
            f"---\nQuestion {i} (id: {sample.id}, type: {sample.task.value}):\n"
            f"{sample.question}\n\n"
            f"Options:\n{_format_options(sample.options)}\n"
        )

    return "\n".join(parts)


def build_messages(
    samples: list[Sample],
) -> tuple[list[dict[str, str]], dict]:
    """Build the full message list and response format for a batch of samples.

    Args:
        samples: One or more samples to include in this request.

    Returns:
        Tuple of (messages, response_format) ready for the provider.
    """
    if len(samples) == 1:
        user_msg = build_single_prompt(samples[0])
        response_format = SINGLE_ANSWER_SCHEMA
    else:
        user_msg = build_batch_prompt(samples)
        response_format = BATCH_ANSWER_SCHEMA

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    return messages, response_format


# ── Response parsing ───────────────────────────────────────────────────────────


def parse_single_response(raw: str) -> int | None:
    """Parse a single-answer JSON response.

    Tries JSON parsing first, then falls back to regex extraction.

    Args:
        raw: Raw model response text.

    Returns:
        The predicted answer index, or None if parsing fails.
    """
    # Try JSON parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict) and "answer" in data:
            answer = data["answer"]
            if isinstance(answer, int):
                return answer
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: extract first integer from response
    match = re.search(r"\b(\d+)\b", raw)
    if match:
        return int(match.group(1))

    return None


def parse_batch_response(
    raw: str,
    expected_ids: list[int],
) -> dict[int, int | None]:
    """Parse a batch-answer JSON response.

    Returns a mapping from sample ID to predicted answer index.

    Args:
        raw: Raw model response text.
        expected_ids: List of sample IDs we expect in the response.

    Returns:
        Dict mapping sample_id → predicted_answer (None if missing/failed).
    """
    results: dict[int, int | None] = {sid: None for sid in expected_ids}

    # Try JSON parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict) and "answers" in data:
            for item in data["answers"]:
                if isinstance(item, dict) and "id" in item and "answer" in item:
                    sid = item["id"]
                    answer = item["answer"]
                    if sid in results and isinstance(answer, int):
                        results[sid] = answer
    except (json.JSONDecodeError, TypeError):
        pass

    return results


# ── Cross-lingual translation ──────────────────────────────────────────────────

TRANSLATION_SYSTEM = """\
You are a professional translator. Your task is to translate verbal reasoning \
multiple-choice questions into {language_name}. You MUST translate EVERY word \
of the text.

CRITICAL RULES:
- Translate ALL text completely. Do NOT skip any sentence, paragraph, or \
labelled item (I, II, III, etc.). EVERYTHING must be translated.
- Translate ALL answer options completely. Every word in every option.
- Preserve the exact meaning, nuance, and difficulty level.
- Preserve all formatting: line breaks, roman numerals, option numbering.
- The correct answer index must be preserved.
- The number of options and their order must be preserved.
- Your response must be valid JSON matching the required schema."""

TRANSLATION_SINGLE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation_single",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Translated question text",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Translated answer options in original order",
                },
            },
            "required": ["question", "options"],
            "additionalProperties": False,
        },
    },
}

TRANSLATION_BATCH_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Sample ID from the question",
                            },
                            "question": {
                                "type": "string",
                                "description": "Translated question text",
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Translated answer options in original order",
                            },
                        },
                        "required": ["id", "question", "options"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["translations"],
            "additionalProperties": False,
        },
    },
}

_LANGUAGE_NAMES: dict[CrossLingualLanguage, str] = {
    CrossLingualLanguage.FRENCH: "French",
    CrossLingualLanguage.CHINESE: "Chinese",
}


def build_translation_messages(
    samples: list[Sample],
    language: CrossLingualLanguage,
) -> tuple[list[dict[str, str]], dict]:
    lang_name = _LANGUAGE_NAMES.get(language, language.value.capitalize())

    if len(samples) == 1:
        response_format = TRANSLATION_SINGLE_SCHEMA
        user_msg = _format_translation_user(samples, lang_name)
    else:
        response_format = TRANSLATION_BATCH_SCHEMA
        user_msg = _format_translation_batch_user(samples, lang_name)

    system_msg = TRANSLATION_SYSTEM.format(language_name=lang_name)

    return (
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        response_format,
    )


def _format_translation_user(samples: list[Sample], lang_name: str) -> str:
    return (
        f"Translate the following question and its answer options "
        f"into {lang_name}:\n\n"
        f"Question (id: {samples[0].id}):\n{samples[0].question}\n\n"
        f"Options:\n{_format_options(samples[0].options)}"
    )


def _format_translation_batch_user(samples: list[Sample], lang_name: str) -> str:
    parts: list[str] = [
        f"Translate each of the following questions and their answer options "
        f"into {lang_name}:\n"
    ]
    for i, s in enumerate(samples, 1):
        parts.append(
            f"---\nQuestion {i} (id: {s.id}):\n{s.question}\n\n"
            f"Options:\n{_format_options(s.options)}\n"
        )
    return "\n".join(parts)


def parse_translation_response(
    raw: str,
    expected_ids: list[int],
) -> dict[int, dict[str, object]]:
    results: dict[int, dict[str, object]] = {}
    expected_set = set(expected_ids)

    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return results

    if isinstance(data, dict):
        if "translations" in data:
            for item in data["translations"]:
                _ingest_translation_item(item, expected_set, results)
        elif "id" in data and "question" in data and "options" in data:
            _ingest_translation_item(data, expected_set, results)
        elif "question" in data and "options" in data and len(expected_ids) == 1:
            _ingest_translation_item(
                {"id": expected_ids[0], **data}, expected_set, results
            )

    return results


def _ingest_translation_item(
    item: object,
    expected_ids: set[int],
    results: dict[int, dict[str, object]],
) -> None:
    if not isinstance(item, dict):
        return
    sid = item.get("id")
    question = item.get("question")
    options = item.get("options")
    if (
        isinstance(sid, int)
        and sid in expected_ids
        and isinstance(question, str)
        and isinstance(options, list)
        and all(isinstance(o, str) for o in options)
    ):
        results[sid] = {"question": question, "options": options}
