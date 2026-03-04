import tempfile
import time
from pathlib import Path

import fasttext as ft
import polars
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score

from eupt_vid.utils import record_experiment

SEED = 42
FT_MAP = {"__label__PT_PT": 0, "__label__PT_BR": 1}

CONFIG = {
    "name": "veracruz_autotune",
    "autotune_duration": 600,
    "autotune_metric": "f1:__label__PT_PT",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "veracruz_large"


def load_data() -> polars.DataFrame:
    print("Loading veracruz data...")
    pt = polars.read_parquet(DATA_DIR / "pt_split.parquet")
    br = polars.read_parquet(DATA_DIR / "br_split.parquet")

    pt = pt.with_columns(polars.lit(0).alias("label"))
    br = br.with_columns(polars.lit(1).alias("label"))

    df = polars.concat([pt, br]).sample(fraction=1.0, seed=SEED)

    n_pt = df.filter(polars.col("label") == 0).height
    n_br = df.filter(polars.col("label") == 1).height
    print(f"  Total: {df.height:,} (PT-PT: {n_pt:,}, PT-BR: {n_br:,})")

    df = df.with_columns(
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

    return df


def split_train_test(df: polars.DataFrame, test_fraction: float = 0.1):
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


def write_fasttext_file(df: polars.DataFrame, path: Path) -> int:
    lines = df["ft_line"].drop_nulls().to_list()
    lines = [line for line in lines if len(line.strip()) > 20]
    path.write_text("\n".join(lines))
    return len(lines)


def train_model(train_df: polars.DataFrame, valid_df: polars.DataFrame, tmpdir: str):
    train_path = Path(tmpdir) / "train.txt"
    valid_path = Path(tmpdir) / "valid.txt"

    n_train = write_fasttext_file(train_df, train_path)
    n_valid = write_fasttext_file(valid_df, valid_path)
    print(f"  Train: {n_train:,} lines | Valid: {n_valid:,} lines")

    print(
        f"  Autotuning for {CONFIG['autotune_duration']}s, "
        f"metric: {CONFIG['autotune_metric']}"
    )

    start = time.perf_counter()
    model = ft.train_supervised(
        input=str(train_path),
        autotuneValidationFile=str(valid_path),
        autotuneDuration=CONFIG["autotune_duration"],
        autotuneMetric=CONFIG["autotune_metric"],
    )
    elapsed = time.perf_counter() - start

    model_path = Path(tmpdir) / "pt_vid.bin"
    model.save_model(str(model_path))
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  {elapsed:.1f}s | {size_mb:.1f} MB")

    # Log the params autotune selected
    args = model.f.getArgs()
    selected = {
        "lr": args.lr,
        "epoch": args.epoch,
        "wordNgrams": args.wordNgrams,
        "minn": args.minn,
        "maxn": args.maxn,
        "dim": args.dim,
        "bucket": args.bucket,
        "minCount": args.minCount,
        "loss": str(args.loss).split(".")[-1],
    }
    print("  Autotune selected params:")
    for k, v in selected.items():
        print(f"    {k}: {v}")

    return model, selected


def print_results(
    true_labels: list[int],
    pred_labels: list[int],
    label: str,
    best_pt_pt_f1: float | None = None,
) -> float:
    pt_pt_f1 = f1_score(true_labels, pred_labels, pos_label=0) * 100

    print(f"\n{label}:")
    print(
        classification_report(true_labels, pred_labels, target_names=["PT-PT", "PT-BR"])
    )
    if best_pt_pt_f1 is not None:
        print(f"  PT-PT F1: {pt_pt_f1:.1f}% (vs. {best_pt_pt_f1:.0f}% best)")
    else:
        print(f"  PT-PT F1: {pt_pt_f1:.1f}%")
    return pt_pt_f1


def evaluate(model, test_df: polars.DataFrame) -> float:
    texts = test_df["text"].str.replace_all("\n", " ").to_list()
    true_labels = test_df["label"].to_list()

    preds = model.predict(texts)
    pred_labels = [FT_MAP[p[0]] for p in preds[0]]
    return print_results(
        true_labels=true_labels, pred_labels=pred_labels, label="In-domain"
    )


def test_dstl(model) -> float:
    print("\nLoading DSL-TL test...")
    ds = load_dataset("LCA-PORVID/dsl_tl", split="test")
    ds = ds.filter(lambda x: x["label"] in [0, 1])
    texts = [t.replace("\n", " ") for t in ds["text"]]
    true_labels = list(ds["label"])

    preds = model.predict(texts)
    pred_labels = [FT_MAP[p[0]] for p in preds[0]]
    return print_results(
        true_labels=true_labels,
        pred_labels=pred_labels,
        label="DSL-TL",
        best_pt_pt_f1=75,
    )


def test_frmt(model) -> float:
    print("\nLoading FRMT test...")
    ds = load_dataset("hugosousa/frmt", split="test")
    pt_texts = [t for t in ds["pt"] if t]
    br_texts = [t for t in ds["br"] if t]
    texts = pt_texts + br_texts
    true_labels = [0] * len(pt_texts) + [1] * len(br_texts)

    preds = model.predict(texts)
    pred_labels = [FT_MAP[p[0]] for p in preds[0]]
    return print_results(
        true_labels=true_labels, pred_labels=pred_labels, label="FRMT", best_pt_pt_f1=76
    )


if __name__ == "__main__":
    data = load_data()
    train, test = split_train_test(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model, selected_params = train_model(
            train_df=train, valid_df=test, tmpdir=tmpdir
        )
        in_domain_f1 = evaluate(model=model, test_df=test)
        dstl_f1 = test_dstl(model=model)
        frmt_f1 = test_frmt(model=model)

    record_experiment(
        experiment_name=str(CONFIG["name"]),
        details={
            "In-domain PT-PT F1": f"{in_domain_f1:.1f}%",
            "DSL-TL PT-PT F1": f"{dstl_f1:.1f}%",
            "FRMT PT-PT F1": f"{frmt_f1:.1f}%",
            **CONFIG,
            "selected_params": selected_params,
        },
    )
