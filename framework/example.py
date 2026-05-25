import os
from pathlib import Path
from lib import Benchmark, providers, attacks, CrossLingualLanguage

OPENCODEGO_APIKEY = 'sk-...'
os.environ['OPENCODEGO_APIKEY'] = OPENCODEGO_APIKEY

benchmark = Benchmark(
    baseline=Path("dataset.json"),
    attacks=[
        attacks.CrossLingual(language=CrossLingualLanguage.FRENCH),
        attacks.CrossLingual(language=CrossLingualLanguage.CHINESE),
        attacks.Synonym("synonym_1"),
        attacks.Paraphrasing("paraphrasing_1"),
    ],
    models=[
        providers.Ollama(
            model="qwen2.5:7b-instruct",
            url="localhost:11434",
            batch=2,
        ),
        providers.OpenRouter(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            api_key=OPENCODEGO_APIKEY,
            batch=2,
            logprobs=True,
            top_logprobs=5,
        ),
        providers.OpencodeGo(
            model="kimi-k2.6",
            api_key=OPENCODEGO_APIKEY,
            label="temp=0.0",
            logprobs=True,
            top_logprobs=5,
        ),
        providers.OpencodeGo(
            model="kimi-k2.6",
            api_key=OPENCODEGO_APIKEY,
            batch=1,
            temperature=0.7,
            label="temp=0.7",
        ),
    ],
    concurrency=4,
    base_dir="benchmark",
)

result = benchmark.run()

for model in result:
    print(f"\nModel: {model.model_name}")
    for ds in model.evaluated_datasets:
        print(f"  {ds.dataset_file} - accuracy: {ds.metrics.accuracy:.2%}")
        if ds._robustness:
            r = ds._robustness
            print(f"    accuracy_drop={r.accuracy_drop:+.2%}  flip_rate={r.flip_rate:.2%}  pos_transfer={r.positive_transfer:.2%}")

if result.is_finished:
    result.save("stats.json")
    result.save("report.md")
else:
    print("\nWarning: benchmark did not finish - partial results were saved.")
    result.save("stats.json")
