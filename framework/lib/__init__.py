"""LLM Verbal Reasoning Evaluation Framework.

A modular framework for evaluating LLM robustness on verbal reasoning tasks.

Usage::

    from pathlib import Path
    from llm_verbal_framework import Benchmark, providers, attacks, CrossLingualLanguage

    benchmark = Benchmark(
        baseline=Path("dataset.json"),
        attacks=[
            attacks.CrossLingual(
                language=CrossLingualLanguage.FRENCH,
                load_from=Path("dataset.french.json"),
            ),
        ],
        models=[
            providers.Ollama(model="qwen2.5:7b-instruct", batch=2),
        ],
        base_dir=".benchmark",
    )
    result = benchmark.run()
    result.save("stats.json")
"""

from .benchmark import Benchmark
from .types import CrossLingualLanguage
from . import attacks
from . import perturb
from . import providers

__all__ = ["Benchmark", "CrossLingualLanguage", "attacks", "perturb", "providers"]
