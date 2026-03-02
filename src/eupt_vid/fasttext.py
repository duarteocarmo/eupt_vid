import time
from pathlib import Path

import polars
import fasttext as ft
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

FT_MAP = {"__label__PT_PT": 0, "__label__PT_BR": 1}

DEFAULT_CONFIG = {
    "max_per_class": 100_000,
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


def load_and_balance_training_data(config: dict) -> polars.DataFrame:
    print("Loading journalistic training split...")
    ds = load_dataset("liaad/PtBrVId", "journalistic", split="train")
    df: polars.DataFrame = polars.from_arrow(ds.data.table)  # type: ignore[assignment]

    n_pt = df.filter(polars.col("label") == 0).height
    n_br = df.filter(polars.col("label") == 1).height
    print(f"  Raw: {df.height:,} (PT-PT: {n_pt:,}, PT-BR: {n_br:,})")

    per_class = min(n_pt, n_br)
    if config["max_per_class"] is not None:
        per_class = min(per_class, config["max_per_class"])

    balanced = polars.concat(
        [
            df.filter(polars.col("label") == 0).sample(n=per_class, seed=42),
            df.filter(polars.col("label") == 1).sample(n=per_class, seed=42),
        ]
    ).sample(fraction=1.0, seed=42)

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


def split_train_test(
    df: polars.DataFrame, test_fraction: float = 0.1, seed: int = 42
) -> tuple[polars.DataFrame, polars.DataFrame]:
    train_frames, test_frames = [], []
    for label in [0, 1]:
        subset = df.filter(polars.col("label") == label).sample(fraction=1.0, seed=seed)
        n_test = int(subset.height * test_fraction)
        test_frames.append(subset.head(n_test))
        train_frames.append(subset.tail(subset.height - n_test))

    train = polars.concat(train_frames).sample(fraction=1.0, seed=seed)
    test = polars.concat(test_frames).sample(fraction=1.0, seed=seed)
    print(f"  Train: {train.height:,} | Test: {test.height:,}")
    return train, test


def train_model(
    train_df: polars.DataFrame, config: dict, tmpdir: str
) -> ft.FastText._FastText:
    train_path = Path(tmpdir) / "train.txt"
    lines = train_df["ft_line"].drop_nulls().to_list()
    lines = [line for line in lines if len(line.strip()) > 20]
    train_path.write_text("\n".join(lines))
    print(f"  {len(lines):,} lines")

    start = time.perf_counter()
    model = ft.train_supervised(
        input=str(train_path),
        lr=config["fasttext_lr"],
        epoch=config["fasttext_epoch"],
        wordNgrams=config["fasttext_wordNgrams"],
        minn=config["fasttext_minn"],
        maxn=config["fasttext_maxn"],
        dim=config["fasttext_dim"],
        bucket=config["fasttext_bucket"],
        loss=config["fasttext_loss"],
        thread=config["fasttext_thread"],
        minCount=config["fasttext_minCount"],
    )
    elapsed = time.perf_counter() - start

    model_path = Path(tmpdir) / "pt_vid.bin"
    model.save_model(str(model_path))
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  {elapsed:.1f}s | {size_mb:.1f} MB")
    return model


def evaluate(model: ft.FastText._FastText, test_df: polars.DataFrame):
    texts = test_df["text"].str.replace_all("\n", " ").to_list()
    true_labels = test_df["label"].to_list()

    start = time.perf_counter()
    preds = model.predict(texts)
    elapsed = time.perf_counter() - start

    pred_labels = [FT_MAP[p[0]] for p in preds[0]]
    speed = len(texts) / elapsed

    print(f"\nIn-domain Results ({len(texts):,} docs, {speed:,.0f} sent/s):")
    print(f"  PT-PT F1: {f1_score(true_labels, pred_labels, pos_label=0) * 100:.1f}%")
    print(f"  PT-BR F1: {f1_score(true_labels, pred_labels, pos_label=1) * 100:.1f}%")
    print(
        f"  Macro F1: {f1_score(true_labels, pred_labels, average='macro') * 100:.1f}%"
    )
    print(f"  Accuracy: {accuracy_score(true_labels, pred_labels) * 100:.1f}%")
    print(
        classification_report(true_labels, pred_labels, target_names=["PT-PT", "PT-BR"])
    )


def test_dstl(model: ft.FastText._FastText):
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

    print(f"\nDSL-TL Results ({len(texts):,} docs, {speed:,.0f} sent/s):")
    print(f"  PT-PT F1: {f1_score(true_labels, pred_labels, pos_label=0) * 100:.1f}%")
    print(f"  PT-BR F1: {f1_score(true_labels, pred_labels, pos_label=1) * 100:.1f}%")
    print(
        f"  Macro F1: {f1_score(true_labels, pred_labels, average='macro') * 100:.1f}%"
    )
    print(f"  Accuracy: {accuracy_score(true_labels, pred_labels) * 100:.1f}%")
    print(
        classification_report(true_labels, pred_labels, target_names=["PT-PT", "PT-BR"])
    )


def main(config: dict | None = None):
    import tempfile

    config = config or DEFAULT_CONFIG
    data = load_and_balance_training_data(config=config)
    train, test = split_train_test(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = train_model(train_df=train, config=config, tmpdir=tmpdir)
        evaluate(model=model, test_df=test)
        test_dstl(model=model)
