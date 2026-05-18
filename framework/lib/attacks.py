"""Attack type definitions for adversarial robustness evaluation.

Each AttackType subclass defines a perturbation strategy. Instances can either
load a pre-computed perturbed dataset via ``load_from``, or apply the perturbation
on-the-fly by implementing ``perturb()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .types import Sample


@dataclass(frozen=True)
class AttackType:
    """Base class for attack/perturbation strategies.

    Attributes:
        label: User-defined label for this specific attack variant, e.g. synonym_low_intensity.
        load_from: Optional path to a pre-computed perturbed dataset file.
    """

    label: str
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
    """Cross-lingual perturbation: translating parts of the input to another language.

    Examples: translate prompt, context, question, or mix languages.
    """
    pass


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
