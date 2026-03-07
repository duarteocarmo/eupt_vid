# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers",
#     "torch",
#     "polars",
# ]
# ///

import polars
from transformers import pipeline

pipe = pipeline("token-classification", model="lisaterumi/postagger-portuguese")

df = polars.read_parquet("data/ptbrvid-data/journalistic/test-00000-of-00001.parquet")

samples = polars.concat(
    [
        df.filter(polars.col("label") == 0).sample(n=2, seed=42),
        df.filter(polars.col("label") == 1).sample(n=2, seed=42),
    ]
)

for row in samples.iter_rows(named=True):
    sentence = row["text"]
    label = row["label"]
    # truncate display if sentence is very long
    display = sentence[:120] + "..." if len(sentence) > 120 else sentence
    print(f"\n{'=' * 80}")
    print(f"Label: {label} | Text: {display}")
    print(f"{'=' * 80}")

    results = pipe(sentence)
    for token in results:
        print(
            f"  {token['word']:>20}  →  {token['entity']:>12}  (score: {token['score']:.4f})"
        )
