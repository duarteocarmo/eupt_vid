import tempfile
import time
from pathlib import Path

import polars
import fasttext
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

MAX_PER_CLASS: int | None = 100_000  # set to None for full dataset


def load_and_balance_training_data() -> polars.DataFrame:
    print("Loading journalistic training split...")
    ds = load_dataset("liaad/PtBrVId", "journalistic", split="train")
    df = polars.from_arrow(ds.data.table)

    n_pt = df.filter(polars.col("label") == 0).height
    n_br = df.filter(polars.col("label") == 1).height
    print(f"  Raw: {df.height:,} (PT-PT: {n_pt:,}, PT-BR: {n_br:,})")

    per_class = min(n_pt, n_br)
    if MAX_PER_CLASS is not None:
        per_class = min(per_class, MAX_PER_CLASS)

    balanced = polars.concat(
        [
            df.filter(polars.col("label") == 0).sample(n=per_class, seed=42),
            df.filter(polars.col("label") == 1).sample(n=per_class, seed=42),
        ]
    ).sample(fraction=1.0, seed=42)

    # format for fasttext: __label__PT_PT text on one line
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
    """Stratified split keeping label distribution equal in both sets."""
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


def train_model(train_df: polars.DataFrame, tmpdir: str) -> fasttext.FastText._FastText:
    train_path = Path(tmpdir) / "train.txt"
    lines = train_df["ft_line"].drop_nulls().to_list()
    lines = [line for line in lines if len(line.strip()) > 15]  # skip empty/tiny docs
    train_path.write_text("\n".join(lines))

    print("Training fastText...")
    start = time.perf_counter()
    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        minn=3,
        maxn=5,
        dim=50,
        bucket=100_000,
        loss="softmax",
        thread=8,
    )
    elapsed = time.perf_counter() - start

    model_path = Path(tmpdir) / "pt_vid.bin"
    model.save_model(str(model_path))
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  {elapsed:.1f}s | {size_mb:.1f} MB")
    return model


def evaluate(model: fasttext.FastText._FastText, test_df: polars.DataFrame):
    ft_map = {"__label__PT_PT": 0, "__label__PT_BR": 1}
    texts = test_df["text"].str.replace_all("\n", " ").to_list()
    true_labels = test_df["label"].to_list()

    start = time.perf_counter()
    preds = model.predict(texts)
    elapsed = time.perf_counter() - start

    pred_labels = [ft_map[p[0]] for p in preds[0]]
    speed = len(texts) / elapsed

    print(f"\nResults ({len(texts):,} docs, {speed:,.0f} sent/s):")
    print(f"  PT-PT F1: {f1_score(true_labels, pred_labels, pos_label=0) * 100:.1f}%")
    print(f"  PT-BR F1: {f1_score(true_labels, pred_labels, pos_label=1) * 100:.1f}%")
    print(
        f"  Macro F1: {f1_score(true_labels, pred_labels, average='macro') * 100:.1f}%"
    )
    print(f"  Accuracy: {accuracy_score(true_labels, pred_labels) * 100:.1f}%")
    print(
        classification_report(true_labels, pred_labels, target_names=["PT-PT", "PT-BR"])
    )


if __name__ == "__main__":
    data = load_and_balance_training_data()
    train, test = split_train_test(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = train_model(train_df=train, tmpdir=tmpdir)
        evaluate(model=model, test_df=test)
