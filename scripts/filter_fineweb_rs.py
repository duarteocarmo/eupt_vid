# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "fasttext-rs>=0.2.1",
#     "polars",
#     "huggingface-hub",
# ]
# ///
"""Filter a FineWeb-2 Portuguese parquet for PT-PT using fasttext-rs."""

import time
from pathlib import Path

import fasttext_rs
import polars as pl
from huggingface_hub import hf_hub_download

MODEL_PATH = "models/20260305_171908_veracruz_6M_best.bin"
THRESHOLD = 0.7
OUTPUT_PATH = Path("data/fineweb2_ptpt_sample_rs.parquet")


def download_parquet() -> Path:
    path = hf_hub_download(
        repo_id="HuggingFaceFW/fineweb-2",
        filename="data/por_Latn/test/000_00000.parquet",
        repo_type="dataset",
    )
    return Path(path)


def get_ptpt_score(model, text: str) -> float:
    labels, probs = model.predict(text.replace("\n", " "), k=2)
    for label, prob in zip(labels, probs):
        if label == "__label__PT_PT":
            return prob
    return 0.0


def main():
    print("Loading model...")
    model = fasttext_rs.load_model(MODEL_PATH)

    print("Downloading parquet...")
    parquet_path = download_parquet()
    df = pl.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} rows")

    print(f"\nClassifying with threshold={THRESHOLD}...")
    start = time.perf_counter()

    scores = [get_ptpt_score(model, text) for text in df["text"].to_list()]
    elapsed = time.perf_counter() - start

    df = df.with_columns(pl.Series("ptpt_score", scores))
    speed = len(df) / elapsed

    print(f"  Classified {len(df):,} rows in {elapsed:.1f}s ({speed:,.0f} rows/sec)")
    print("  Score distribution:")
    print(
        f"    mean={df['ptpt_score'].mean():.3f}  median={df['ptpt_score'].median():.3f}"
    )

    # Filter
    df_ptpt = df.filter(pl.col("ptpt_score") >= THRESHOLD)
    kept_pct = len(df_ptpt) / len(df) * 100

    print(f"\n  Total: {len(df):,} → PT-PT: {len(df_ptpt):,} ({kept_pct:.1f}% kept)")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_ptpt.write_parquet(OUTPUT_PATH)
    print(f"  Saved to {OUTPUT_PATH}")

    # Sample URLs
    sample = df_ptpt.sample(n=min(25, len(df_ptpt)), seed=42)
    print("\n  25 random PT-PT URLs:")
    for url in sample["url"].to_list():
        print(f"  - {url}")


if __name__ == "__main__":
    main()
