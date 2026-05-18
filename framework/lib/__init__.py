"""LLM Verbal Reasoning Evaluation Framework.

A modular framework for evaluating LLM robustness on verbal reasoning tasks

Usage::

    from pathlib import Path
    from .lib import Benchmark, providers, attacks

    benchmark = Benchmark(
        baseline=Path("dataset.json"),
        attacks=[
            attacks.CrossLingual("fr_mixed", load_from=Path("dataset.french.json")),
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
from . import attacks
from . import perturb
from . import providers

__all__ = ["Benchmark", "attacks", "perturb", "providers"]
