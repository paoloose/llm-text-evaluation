"""Perturbation pipeline stage.

Produces perturbed (attacked) datasets from a baseline by either loading a
pre-computed file (when ``load_from`` is set on the attack) or applying the
perturbation strategy on-the-fly.
"""

from __future__ import annotations

import logging

from .attacks import AttackType
from .dataset import Dataset, load_dataset, validate_alignment

logger = logging.getLogger("llm_verbal_framework")


def generate_perturbed_dataset(
    baseline: Dataset,
    attack: AttackType,
) -> Dataset:
    """Produce a perturbed dataset from a baseline.

    If ``attack.load_from`` is set, the dataset is loaded directly from the
    given file and tagged with the attack metadata. Otherwise, the attack's
    ``perturb()`` method is called to transform the baseline samples.

    The resulting dataset is always validated for alignment with the baseline
    (matching sample IDs) before being returned.

    Args:
        baseline: The baseline (unperturbed) dataset.
        attack: The attack/perturbation instance.

    Returns:
        A validated Dataset tagged with the attack metadata.

    Raises:
        FileNotFoundError: If ``load_from`` points to a non-existent file.
        NotImplementedError: If ``load_from`` is not set and the attack does
            not implement ``perturb()``.
        ValueError: If the perturbed dataset fails ID alignment validation.
    """
    if attack.load_from:
        logger.info(
            "Loading perturbed dataset from: %s (%s)", attack.load_from, attack.attack_name
        )
        ds = load_dataset(attack.load_from, attack=attack)
    else:
        logger.info(
            "Applying perturbation: %s (label=%s)", attack.attack_name, attack.label
        )
        perturbed_samples = attack.perturb(baseline.samples)
        source = f"{baseline.filename}.{attack.attack_name}.json"
        ds = Dataset(samples=perturbed_samples, source_file=source, attack=attack)

    validate_alignment(baseline, ds)
    logger.info("Perturbed dataset prepared and validated: %d samples", len(ds))
    return ds
