"""Prompt engineering for LLM verbal reasoning evaluation.

Handles:
- Building system and user messages per task type
- Constructing JSON schema for structured output (single and batch)
- Parsing model responses with fallback strategies
"""

from __future__ import annotations

import json
import re

from .types import CrossLingualLanguage, Sample

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
        f"<question id={sample.id} type={sample.task.value}>\n"
        f"<content>"
        f"{sample.question}"
        f"</content>"
        f"<options n={len(sample.options)}>\n"
        f"{_format_options(sample.options)}"
        f"</options>\n"
        f"</question>\n"
    )


def build_batch_prompt(samples: list[Sample]) -> str:
    """Build user message for a batch of samples.

    Args:
        samples: List of samples to format as a single prompt.

    Returns:
        Formatted user message string with all questions.
    """
    parts: list[str] = ["<instructions>Answer each of the following questions</instructions>\n"]

    for i, sample in enumerate(samples, 1):
        parts.append(build_single_prompt(sample))

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

TRANSLATION_SYSTEM_PROMPT = """\
Translate the user's question & options into {language_name}.\nTags:\n\
- The "question" tag contains the full question. Do not stop until you find the closing tag: </question>. It has a length property, and your translated question should roughly have the same length.\n\
- The "options" tag contains a list of human readable possible answers to the question. Parse them until closing tag: </options>.\n\
\n\
OUTPUT FORMAT — return ONLY a JSON object with a single key:\n\
  "translations": an array of objects, each with:\n\
    "id":       the sample id (integer)\n\
    "question": translated question text (string)\n\
    "options":  translated answer options in original order (array of strings)\n\
\n\
Do NOT wrap the JSON in markdown code blocks. No extra text. Translate every word."""

_LANGUAGE_NAMES: dict[CrossLingualLanguage, str] = {
    CrossLingualLanguage.FRENCH: "French",
    CrossLingualLanguage.CHINESE: "Chinese",
}

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
            },
            "required": ["question"],
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
                        },
                        "required": ["id", "question"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["translations"],
            "additionalProperties": False,
        },
    },
}

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
    system_msg = TRANSLATION_SYSTEM_PROMPT.format(language_name=lang_name)
    return (
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        response_format,
    )

def _format_translation_user(samples: list[Sample], lang_name: str) -> str:
    q = samples[0].question
    return (
        f"<question length={len(q)}>\n"
        f"{q}\n"
        f"</question>\n"
        f"<options n={len(samples[0].options)}>\n"
        f"{_format_options(samples[0].options)}\n"
        f"</options>"
    )


def _format_translation_batch_user(samples: list[Sample], lang_name: str) -> str:
    parts: list[str] = []
    for i, s in enumerate(samples, 1):
        parts.append(
            f"<sample id={s.id}>\n"
            f"{_format_translation_user([s], lang_name)}\n"
            f"</sample>"
        )
    return "\n".join(parts)


def _repair_translation_json(raw: str) -> str:
    """Repair LLM-generated JSON that contains unescaped typographic double
    quotes inside string values (e.g. French ``d"interprétation``).

    Heuristic: a ``"`` preceded and followed by an alphanumeric character
    is treated as an inner quote and escaped with a backslash.
    """
    chars = list(raw)
    escape_next = False
    i = 0
    while i < len(chars):
        c = chars[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if c == "\\":
            escape_next = True
            i += 1
            continue
        if c == '"':
            prev_char = chars[i - 1] if i > 0 else ""
            next_char = chars[i + 1] if i + 1 < len(chars) else ""
            if prev_char.isalnum() and next_char.isalnum():
                chars.insert(i, "\\")
                i += 2
                continue
        i += 1
    return "".join(chars)


def parse_translation_response(
    raw: str,
    expected_ids: list[int],
) -> dict[int, str]:
    """Return ``{sample_id: translated_question}``.  Options are NOT translated —
    they are preserved from the baseline."""
    results: dict[int, str] = {}
    expected_set = set(expected_ids)

    raw_fixed = (
        raw.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        data = json.loads(raw_fixed)
    except (json.JSONDecodeError, TypeError):
        repaired = _repair_translation_json(raw_fixed)
        try:
            data = json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            return results

    if isinstance(data, dict):
        _parse_dict_response(data, expected_ids, expected_set, results)
    elif isinstance(data, list):
        _parse_list_response(data, expected_ids, expected_set, results)

    return results


def _parse_dict_response(
    data: dict,
    expected_ids: list[int],
    expected_set: set[int],
    results: dict[int, str],
) -> None:
    if "translations" in data:
        for item in data["translations"]:
            _ingest(item, expected_set, results)
    elif "id" in data and "question" in data:
        _ingest(data, expected_set, results)
    elif "question" in data and len(expected_ids) == 1:
        _ingest(
            {"id": expected_ids[0], "question": data["question"]},
            expected_set,
            results,
        )


def _parse_list_response(
    data: list,
    expected_ids: list[int],
    expected_set: set[int],
    results: dict[int, str],
) -> None:
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        if not isinstance(question, str):
            continue
        sid = item.get("id")
        if isinstance(sid, int) and sid in expected_set:
            results[sid] = question
        elif idx < len(expected_ids):
            sid = expected_ids[idx]
            if sid not in results:
                results[sid] = question


def _ingest(
    item: object,
    expected_ids: set[int],
    results: dict[int, str],
) -> None:
    if not isinstance(item, dict):
        return
    sid = item.get("id")
    question = item.get("question")
    if isinstance(sid, int) and sid in expected_ids and isinstance(question, str):
        results[sid] = question
