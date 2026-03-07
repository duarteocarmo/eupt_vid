"""Generate precision-recall curves for PT-PT detection on DSL-TL and FRMT."""

import os
import sys
from pathlib import Path

import fasttext as ft
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, precision_recall_curve

MODEL_PATH = "models/20260305_171908_veracruz_6M_best.bin"
OUTPUT_DIR = Path("reports")


def get_ptpt_scores(model, texts: list[str]) -> np.ndarray:
    """Get P(PT-PT) for each text."""
    scores = []
    for t in texts:
        result = model.f.predict(t.replace("\n", " "), 2, 0.0, "")
        # result is [(prob, label), (prob, label)]
        for prob, label in result:
            if label == "__label__PT_PT":
                scores.append(prob)
                break
    return np.array(scores)


def load_dstl() -> tuple[list[str], np.ndarray]:
    ds = load_dataset("LCA-PORVID/dsl_tl", split="test")
    ds = ds.filter(lambda x: x["label"] in [0, 1])
    texts = list(ds["text"])
    # label 0 = PT-PT → 1 (positive), label 1 = PT-BR → 0
    labels = np.array([1 if la == 0 else 0 for la in ds["label"]])
    return texts, labels


def load_frmt() -> tuple[list[str], np.ndarray]:
    ds = load_dataset("hugosousa/frmt", split="test")
    pt_texts = [t for t in ds["pt"] if t]
    br_texts = [t for t in ds["br"] if t]
    texts = pt_texts + br_texts
    labels = np.array([1] * len(pt_texts) + [0] * len(br_texts))
    return texts, labels


def plot_pr_curve(
    curves: list[tuple[str, np.ndarray, np.ndarray, float]],
    output_path: Path,
):
    """Plot precision-recall curves for multiple datasets."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, precision, recall, ap in curves:
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.1%})")

    ax.set_xlabel("Recall (PT-PT)", fontsize=12)
    ax.set_ylabel("Precision (PT-PT)", fontsize=12)
    ax.set_title("Precision-Recall Curve — PT-PT Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim((0, 1.02))
    ax.set_ylim((0, 1.02))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("Loading model...")
    model = ft.load_model(MODEL_PATH)

    print("Loading datasets...")
    dstl_texts, dstl_labels = load_dstl()
    frmt_texts, frmt_labels = load_frmt()
    print(f"  DSL-TL: {len(dstl_texts)} samples | FRMT: {len(frmt_texts)} samples")

    print("Scoring...")
    dstl_scores = get_ptpt_scores(model, dstl_texts)
    frmt_scores = get_ptpt_scores(model, frmt_texts)

    # Mix both datasets
    mixed_texts = dstl_texts + frmt_texts
    mixed_labels = np.concatenate([dstl_labels, frmt_labels])
    mixed_scores = np.concatenate([dstl_scores, frmt_scores])
    print(f"  Mixed: {len(mixed_texts)} samples")

    curves = []
    for name, labels, scores in [
        ("DSL-TL", dstl_labels, dstl_scores),
        ("FRMT", frmt_labels, frmt_scores),
        ("DSL-TL + FRMT", mixed_labels, mixed_scores),
    ]:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        curves.append((name, precision, recall, ap))

        # Print some useful operating points
        print(f"\n{name} — operating points:")
        print(f"  {'Threshold':>10s}  {'Precision':>10s}  {'Recall':>10s}")
        for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
            mask = thresholds >= t
            if mask.any():
                idx = np.where(mask)[0][0]
                print(f"  {t:>10.2f}  {precision[idx]:>10.1%}  {recall[idx]:>10.1%}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_pr_curve(curves, OUTPUT_DIR / "pr_curve_ptpt.png")
