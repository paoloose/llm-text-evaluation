"""LLM Verbal Reasoning Evaluation Framework.

A modular framework for evaluating LLM robustness on verbal reasoning tasks

Usage::

    from pathlib import Path
    from lib import Benchmark, providers, attacks

    benchmark = Benchmark(
        baseline=Path("dataset.json"),
        attacked=[
            (Path("dataset.french.json"), attacks.CrossLingual("fr_mixed")),
        ],
        models=[
            providers.Ollama(model="qwen2.5:7b-instruct", batch=2),
        ],
    )
    result = benchmark.run()
    result.save("stats.json")
"""

from .benchmark import Benchmark
from . import attacks
from . import providers

__all__ = ["Benchmark", "attacks", "providers"]
