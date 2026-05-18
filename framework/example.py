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
        providers.Ollama(
            model="qwen2.5:7b-instruct",
            url="localhost:11434",
            batch=2,
        ),
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
            label="temp=0.0",
            logprobs=True,
            top_logprobs=5,
        ),
        providers.OpencodeGo(
            model="kimi-k2.6",
            api_key="oc-go-v1-...",
            batch=1,
            temperature=0.7,
            label="temp=0.7",
        ),
    ],
    concurrency=4,
    partial_results_dir="partial",
    base_dir=".benchmark",
)

result = benchmark.run()

for model in result:
    print(f"\nModel: {model.model_name}")
    for ds in model.evaluated_datasets:
        print(f"  {ds.dataset_file} — accuracy: {ds.metrics.accuracy:.2%}")
        if ds._robustness:
            r = ds._robustness
            print(f"    accuracy_drop={r.accuracy_drop:+.2%}  flip_rate={r.flip_rate:.2%}  pos_transfer={r.positive_transfer:.2%}")

if result.is_finished:
    result.save("stats.json")
    result.save("report.md")
else:
    print("\nWarning: benchmark did not finish — partial results were saved.")
    result.save("stats.json")
