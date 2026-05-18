from pathlib import Path

from lib import Benchmark, providers, attacks

benchmark = Benchmark(
    baseline=Path("dataset.json"),
    attacks=[
        attacks.CrossLingual("fr_mixed", load_from=Path("dataset.french.json")),
        attacks.Synonym("synonym_1", load_from=Path("dataset.synonyms.json")),
        attacks.Paraphrasing("paraphrasing_1", load_from=Path("dataset.paraphrasing.json")),
    ],
    models=[
        providers.Ollama(model="qwen2.5:7b-instruct", batch=2),
        providers.OpenRouter(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            api_key="sk-or-v1-...",
            batch=2,
            logprobs=True,
            top_logprobs=5,
        ),
        providers.OpencodeGo(
            model="kimi-k2.6",
            api_key="oc-go-v1-...",
            batch=2,
            logprobs=True,
            top_logprobs=5,
        ),
        providers.OpencodeGo(model="minimax-m2.7", api_key="oc-go-v1-...", batch=1),
    ],
    concurrency=4,
    partial_results_dir="partial",
    base_dir=".benchmark",
)

result = benchmark.run()

if result.is_finished:
    for model in result:
        for ds in model.evaluated_datasets:
            print(ds.stats)
    result.save("stats.json")
    result.save("report.md")
