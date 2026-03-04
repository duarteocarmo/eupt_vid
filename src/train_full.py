import tempfile
import time
from pathlib import Path

import fasttext as ft
import polars
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score

from eupt_vid.utils import record_experiment

SEED = 42
FT_MAP = {"__label__PT_PT": 0, "__label__PT_BR": 1}
ALL_SUBSETS = ["journalistic", "legal", "literature", "politics", "social_media", "web"]

CONFIG = {
    "name": "full_baseline",
    "max_per_class": None,
    "fasttext_lr": 0.05,
    "fasttext_epoch": 25,
    "fasttext_wordNgrams": 2,
    "fasttext_minn": 3,
    "fasttext_maxn": 5,
    "fasttext_dim": 250,
    "fasttext_bucket": 1_000_000,
    "fasttext_minCount": 1000,
    "fasttext_loss": "softmax",
    "fasttext_thread": 24,
}


def load_data() -> polars.DataFrame:
    print("Loading all training splits...")
    frames = []
    for subset in ALL_SUBSETS:
        ds = load_dataset("liaad/PtBrVId", subset, split="train")
        df: polars.DataFrame = polars.from_arrow(ds.data.table)  # type: ignore[assignment]
        df = df.with_columns(polars.lit(subset).alias("category"))
        print(f"  {subset}: {df.height:,}")
        frames.append(df)

    combined = polars.concat(frames).sample(fraction=1.0, seed=SEED)

    n_pt = combined.filter(polars.col("label") == 0).height
    n_br = combined.filter(polars.col("label") == 1).height
    print(f"  Total: {combined.height:,} (PT-PT: {n_pt:,}, PT-BR: {n_br:,})")

    per_class: int = min(n_pt, n_br)
    if CONFIG["max_per_class"] is not None:
        per_class = min(per_class, int(CONFIG["max_per_class"]))

    balanced = polars.concat(
        [
            combined.filter(polars.col("label") == 0).sample(n=per_class, seed=SEED),
            combined.filter(polars.col("label") == 1).sample(n=per_class, seed=SEED),
        ]
    ).sample(fraction=1.0, seed=SEED)

    balanced = balanced.with_columns(
        polars.when(polars.col("label") == 0)
        .then(polars.lit("__label__PT_PT"))
        .otherwise(polars.lit("__label__PT_BR"))
        .alias("ft_label")
    ).with_columns(
        (
            polars.col("ft_label")
            + " "
            + polars.col("text").str.replace_all("\n", " ").str.strip_chars()
        ).alias("ft_line")
    )

    print(f"  Balanced: {balanced.height:,} ({per_class:,} per class)")
    return balanced


def split_train_test(df: polars.DataFrame, test_fraction: float = 0.2):
    train_frames, test_frames = [], []
    for label in [0, 1]:
        subset = df.filter(polars.col("label") == label).sample(fraction=1.0, seed=SEED)
        n_test = int(subset.height * test_fraction)
        test_frames.append(subset.head(n_test))
        train_frames.append(subset.tail(subset.height - n_test))

    train = polars.concat(train_frames).sample(fraction=1.0, seed=SEED)
    test = polars.concat(test_frames).sample(fraction=1.0, seed=SEED)
    print(f"  Train: {train.height:,} | Test: {test.height:,}")
    return train, test


def train_model(train_df: polars.DataFrame, tmpdir: str):
    train_path = Path(tmpdir) / "train.txt"
    lines = train_df["ft_line"].drop_nulls().to_list()
    lines = [line for line in lines if len(line.strip()) > 20]
    train_path.write_text("\n".join(lines))
    print(f"  {len(lines):,} lines")

    start = time.perf_counter()
    model = ft.train_supervised(
        input=str(train_path),
        lr=CONFIG["fasttext_lr"],
        epoch=CONFIG["fasttext_epoch"],
        wordNgrams=CONFIG["fasttext_wordNgrams"],
        minn=CONFIG["fasttext_minn"],
        maxn=CONFIG["fasttext_maxn"],
        dim=CONFIG["fasttext_dim"],
        bucket=CONFIG["fasttext_bucket"],
        loss=CONFIG["fasttext_loss"],
        thread=CONFIG["fasttext_thread"],
        minCount=CONFIG["fasttext_minCount"],
    )
    elapsed = time.perf_counter() - start

    model_path = Path(tmpdir) / "pt_vid.bin"
    model.save_model(str(model_path))
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  {elapsed:.1f}s | {size_mb:.1f} MB")
    return model


def evaluate(model, test_df: polars.DataFrame) -> dict[str, float]:
    texts = test_df["text"].str.replace_all("\n", " ").to_list()
    true_labels = test_df["label"].to_list()

    start = time.perf_counter()
    preds = model.predict(texts)
    elapsed = time.perf_counter() - start

    pred_labels = [FT_MAP[p[0]] for p in preds[0]]
    speed = len(texts) / elapsed

    pt_f1 = f1_score(true_labels, pred_labels, pos_label=0) * 100
    br_f1 = f1_score(true_labels, pred_labels, pos_label=1) * 100
    macro_f1 = f1_score(true_labels, pred_labels, average="macro") * 100
    acc = accuracy_score(true_labels, pred_labels) * 100

    print(f"\nIn-domain Results ({len(texts):,} docs, {speed:,.0f} sent/s):")
    print(f"  PT-PT F1: {pt_f1:.1f}%")
    print(f"  PT-BR F1: {br_f1:.1f}%")
    print(f"  Macro F1: {macro_f1:.1f}%")
    print(f"  Accuracy: {acc:.1f}%")
    print(
        classification_report(true_labels, pred_labels, target_names=["PT-PT", "PT-BR"])
    )
    return {"In-domain PT-PT F1": pt_f1}


def test_dstl(model) -> dict[str, float]:
    print("\nLoading DSL-TL test...")
    ds = load_dataset("LCA-PORVID/dsl_tl", split="test")
    ds = ds.filter(lambda x: x["label"] in [0, 1])
    texts = [t.replace("\n", " ") for t in ds["text"]]
    true_labels = list(ds["label"])
    print(f"  {len(texts)} documents")

    start = time.perf_counter()
    preds = model.predict(texts)
    elapsed = time.perf_counter() - start

    pred_labels = [FT_MAP[p[0]] for p in preds[0]]
    speed = len(texts) / elapsed

    pt_f1 = f1_score(true_labels, pred_labels, pos_label=0) * 100
    br_f1 = f1_score(true_labels, pred_labels, pos_label=1) * 100
    macro_f1 = f1_score(true_labels, pred_labels, average="macro") * 100
    acc = accuracy_score(true_labels, pred_labels) * 100

    print(f"\nDSL-TL Results ({len(texts):,} docs, {speed:,.0f} sent/s):")
    print(f"  PT-PT F1: {pt_f1:.1f}%")
    print(f"  PT-BR F1: {br_f1:.1f}%")
    print(f"  Macro F1: {macro_f1:.1f}%")
    print(f"  Accuracy: {acc:.1f}%")
    print(
        classification_report(true_labels, pred_labels, target_names=["PT-PT", "PT-BR"])
    )
    return {"DSL-TL PT-PT F1": pt_f1}


if __name__ == "__main__":
    data = load_data()
    train, test = split_train_test(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = train_model(train_df=train, tmpdir=tmpdir)
        in_metrics = evaluate(model=model, test_df=test)
        dstl_metrics = test_dstl(model=model)

    all_metrics: dict[str, float | int | str] = {}
    for d in [in_metrics, dstl_metrics]:
        for k, v in d.items():
            all_metrics[k] = f"{v:.1f}%"

    record_experiment(
        script_name="train_full.py",
        experiment_name=str(CONFIG["name"]),
        metrics=all_metrics,
    )
