"""Train a fastText classifier on the exact same data as the PtBrVId paper (all 6 domains).

Uses configurable fasttext params. Evaluates on DSL-TL and FRMT.
"""

import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import fasttext as ft
import polars as pl
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score

from eupt_vid.utils import record_experiment

SEED = 42
FT_MAP = {"__label__PT_PT": 0, "__label__PT_BR": 1}
DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "ptbrvid-data"
)
MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
ALL_DOMAINS = ["journalistic", "legal", "literature", "politics", "social_media", "web"]

CONFIG = {
    "name": "paper_data_default",
}


def load_all_train() -> pl.DataFrame:
    """Load all training splits from all 6 domains on disk."""
    frames = []
    for domain in ALL_DOMAINS:
        path = DATA_DIR / domain
        df = pl.read_parquet(str(path / "train-*.parquet"))
        df = df.with_columns(pl.lit(domain).alias("domain"))
        n_pt = df.filter(pl.col("label") == 0).height
        n_br = df.filter(pl.col("label") == 1).height
        print(f"  {domain:15s} {df.height:>10,}  (PT-PT: {n_pt:,}, PT-BR: {n_br:,})")
        frames.append(df)

    combined = pl.concat(frames).sample(fraction=1.0, seed=SEED)
    n_pt = combined.filter(pl.col("label") == 0).height
    n_br = combined.filter(pl.col("label") == 1).height
    print(f"  {'TOTAL':15s} {combined.height:>10,}  (PT-PT: {n_pt:,}, PT-BR: {n_br:,})")
    return combined


def load_all_valid() -> pl.DataFrame:
    """Load all validation splits."""
    frames = []
    for domain in ALL_DOMAINS:
        path = DATA_DIR / domain
        df = pl.read_parquet(str(path / "valid-*.parquet"))
        frames.append(df)
    return pl.concat(frames).sample(fraction=1.0, seed=SEED)


def prepare_fasttext(df: pl.DataFrame) -> pl.DataFrame:
    """Add fasttext-formatted line column."""
    return df.with_columns(
        pl.when(pl.col("label") == 0)
        .then(pl.lit("__label__PT_PT"))
        .otherwise(pl.lit("__label__PT_BR"))
        .alias("ft_label")
    ).with_columns(
        (
            pl.col("ft_label")
            + " "
            + pl.col("text").str.replace_all("\n", " ").str.strip_chars()
        ).alias("ft_line")
    )


def train_model(train_df: pl.DataFrame, tmpdir: str):
    """Train with CONFIG params."""
    train_path = Path(tmpdir) / "train.txt"
    lines = train_df["ft_line"].drop_nulls().to_list()
    lines = [line for line in lines if len(line.strip()) > 20]
    train_path.write_text("\n".join(lines))

    print(
        f"\nTraining '{CONFIG['name']}' on {len(lines):,} lines (default fasttext params)..."
    )

    start = time.perf_counter()
    model = ft.train_supervised(input=str(train_path))
    elapsed = time.perf_counter() - start

    model_path = Path(tmpdir) / "model.bin"
    model.save_model(str(model_path))
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  Done in {elapsed:.1f}s | {size_mb:.1f} MB")
    return model


def eval_benchmark(model, texts: list[str], true_labels: list[int], name: str) -> float:
    """Evaluate and print results. Returns PT-PT F1."""
    start = time.perf_counter()
    preds = model.predict(texts)
    elapsed = time.perf_counter() - start

    pred_labels = [FT_MAP[p[0]] for p in preds[0]]
    speed = len(texts) / elapsed
    pt_f1 = f1_score(true_labels, pred_labels, pos_label=0) * 100

    print(f"\n### {name} ({len(texts):,} samples, {speed:,.0f} samples/sec)")
    print(
        classification_report(true_labels, pred_labels, target_names=["PT-PT", "PT-BR"])
    )
    print(f"  PT-PT F1: {pt_f1:.2f}%")
    return pt_f1


def main():
    print("=" * 60)
    print("Loading paper data (all 6 domains)...")
    print("=" * 60)
    train_df = load_all_train()
    valid_df = load_all_valid()

    train_df = prepare_fasttext(train_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = train_model(train_df, tmpdir)

        # In-domain (validation set)
        valid_df_prep = prepare_fasttext(valid_df)
        valid_texts = valid_df_prep["text"].str.replace_all("\n", " ").to_list()
        valid_labels = valid_df_prep["label"].to_list()
        eval_benchmark(model, valid_texts, valid_labels, "In-domain (valid)")

        # DSL-TL
        print("\nLoading DSL-TL...")
        ds = load_dataset("LCA-PORVID/dsl_tl", split="test")
        ds = ds.filter(lambda x: x["label"] in [0, 1])
        dstl_texts = [t.replace("\n", " ") for t in ds["text"]]
        dstl_labels = list(ds["label"])
        dstl_f1 = eval_benchmark(model, dstl_texts, dstl_labels, "DSL-TL")

        # FRMT
        print("\nLoading FRMT...")
        ds2 = load_dataset("hugosousa/frmt", split="test")
        pt_texts = [t for t in ds2["pt"] if t]
        br_texts = [t for t in ds2["br"] if t]
        frmt_texts = [t.replace("\n", " ") for t in pt_texts + br_texts]
        frmt_labels = [0] * len(pt_texts) + [1] * len(br_texts)
        frmt_f1 = eval_benchmark(model, frmt_texts, frmt_labels, "FRMT")

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_name = f"{ts}_{CONFIG['name']}.bin"
        dest = MODELS_DIR / model_name
        shutil.copy2(Path(tmpdir) / "model.bin", dest)
        print(f"\n💾 Model saved to {dest}")

    print("\n" + "=" * 60)
    print(f"SUMMARY — {CONFIG['name']}")
    print("=" * 60)
    print(f"  DSL-TL PT-PT F1: {dstl_f1:.2f}%")
    print(f"  FRMT   PT-PT F1: {frmt_f1:.2f}%")

    record_experiment(
        experiment_name=CONFIG["name"],
        details={
            "DSL-TL PT-PT F1": f"{dstl_f1:.1f}%",
            "FRMT PT-PT F1": f"{frmt_f1:.1f}%",
            **CONFIG,
        },
    )


if __name__ == "__main__":
    main()
