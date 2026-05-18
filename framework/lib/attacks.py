"""Attack type definitions for adversarial robustness evaluation.

Each AttackType subclass defines a perturbation strategy. Instances can either
load a pre-computed perturbed dataset via ``load_from``, or apply the perturbation
on-the-fly by implementing ``perturb()``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .prompt import build_translation_messages, parse_translation_response
from .types import CrossLingualLanguage, Sample

if TYPE_CHECKING:
    from .providers.base import BaseProvider

logger = logging.getLogger("llm_verbal_framework")


@dataclass(frozen=True)
class AttackType:
    """Base class for attack/perturbation strategies.

    Attributes:
        label: User-defined label for this specific attack variant, e.g.
            ``synonym_low_intensity``. When ``None`` the subclass is expected to
            provide a sensible default via ``__post_init__``.
        load_from: Optional path to a pre-computed perturbed dataset file.
    """

    label: str | None = field(default=None)
    load_from: str | Path | None = field(default=None)

    @property
    def attack_name(self) -> str:
        """Machine-readable attack type name, derived from the class name."""
        name = type(self).__name__
        result: list[str] = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.lower())
        return "".join(result)

    def perturb(self, samples: list[Sample]) -> list[Sample]:
        """Transform baseline samples into perturbed versions.

        Subclasses must override this method to implement the perturbation logic.
        When ``load_from`` is provided, this method is not called; the dataset is
        loaded directly from the given file instead.

        Args:
            samples: The baseline (unperturbed) samples.

        Returns:
            A new list of perturbed samples with the same IDs and structure.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{self.attack_name}.perturb() is not implemented. "
            f"Provide a 'load_from' path to use a pre-computed dataset, "
            f"or implement perturb() for this attack type."
        )


@dataclass(frozen=True)
class CrossLingual(AttackType):
    """Cross-lingual perturbation: translating questions and options to another language.

    Uses an LLM to translate the baseline dataset into the target language
    while preserving sample IDs, answer indices, and option ordering.

    When ``label`` is not provided it defaults to ``{language}_base``.

    Args:
        language: Target language for translation.
        label: Unique identifier for this perturbation variant.
            Defaults to ``{language.value}_base``.
        load_from: Path to a pre-computed translated dataset file.
            When set, ``perturb()`` is skipped.
        model: Provider instance to use for translation. Defaults to
            ``OpencodeGo("minimax-m2.5")`` with the API key read from the
            ``OPENCODEGO_APIKEY`` environment variable.
    """

    language: "CrossLingualLanguage | None" = field(default=None)
    model: "BaseProvider | None" = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.language is None:
            raise ValueError("CrossLingual requires a 'language' argument")
        if self.label is None:
            object.__setattr__(self, "label", f"{self.language.value}_base")

    @property
    def _translation_model(self) -> "BaseProvider":
        if self.model is not None:
            return self.model
        from .providers.opencode_go import OpencodeGo

        api_key = os.environ.get("OPENCODEGO_APIKEY")
        if not api_key:
            raise ValueError(
                "CrossLingual requires an API key: set the OPENCODEGO_APIKEY "
                "environment variable or pass an explicit 'model'."
            )
        return OpencodeGo(
            model="minimax-m2.5",
            api_key=api_key,
            batch=1,
            temperature=0.0,
            enforce_json=True,
            retry_times=1,
            max_errors=1,
        )

    async def perturb(self, samples: list[Sample]) -> list[Sample]:
        """Translate samples to the target language using an LLM.

        Args:
            samples: Baseline samples to translate.

        Returns:
            New list of Samples with translated ``question`` and ``options``
            fields. ``id``, ``task``, ``answer``, and ``rationale`` are
            preserved unchanged.
        """
        provider = self._translation_model
        translated: list[Sample] = []
        remaining = list(samples)

        batch_idx = 0

        while remaining:
            batch_samples = remaining[: provider.batch_size]
            remaining = remaining[provider.batch_size :]
            batch_idx += 1
            sample_ids = [s.id for s in batch_samples]
            logger.info(
                "Translate batch %d: %d samples [%s] → %s",
                batch_idx, len(batch_samples), sample_ids, self.language.value,
            )

            messages, response_format = build_translation_messages(
                batch_samples, self.language
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Translation batch %d messages:\n%s",
                    batch_idx,
                    json.dumps(messages, ensure_ascii=False, indent=2),
                )

            start = time.perf_counter()
            try:
                raw_response, _, _ = await provider.complete(
                    messages, response_format
                )
            except Exception as exc:
                logger.error(
                    "Translation batch %d failed: %s: %s",
                    batch_idx, type(exc).__name__, exc,
                )
                raise

            elapsed = time.perf_counter() - start
            logger.info(
                "Translation batch %d completed in %.1fs", batch_idx, elapsed,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Translation batch %d raw response:\n%s",
                    batch_idx, raw_response,
                )

            parsed = parse_translation_response(raw_response, sample_ids)

            for s in batch_samples:
                entry = parsed.get(s.id)
                if entry is None:
                    logger.warning(
                        "Sample %d missing from translation response — "
                        "using original text", s.id,
                    )
                    translated.append(s)
                    continue

                translated.append(
                    Sample(
                        id=s.id,
                        task=s.task,
                        question=entry["question"],
                        options=tuple(entry["options"]),
                        answer=s.answer,
                        rationale=s.rationale,
                    )
                )

        return translated


@dataclass(frozen=True)
class Synonym(AttackType):
    """Synonym substitution: replacing key words with synonyms."""
    pass


@dataclass(frozen=True)
class Paraphrasing(AttackType):
    """Paraphrasing: rewriting sentences without changing meaning."""
    pass


@dataclass(frozen=True)
class MinimalPairs(AttackType):
    """Minimal pairs: changing a single critical word (negation, quantifier, connector)."""
    pass


@dataclass(frozen=True)
class ShortcutRemoval(AttackType):
    """Shortcut removal: eliminating explicit reasoning cues.

    Removes connectors like 'because', 'therefore', 'first/then'.
    """
    pass
