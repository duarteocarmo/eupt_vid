"""Test quantization variants of a fasttext model on DSL-TL and FRMT."""

import os
import sys

import fasttext as ft
from datasets import load_dataset
from sklearn.metrics import f1_score

MODEL_PATH = "models/20260305_171908_veracruz_6M_best.bin"
FT_MAP = {"__label__PT_PT": 0, "__label__PT_BR": 1}


def batch_predict(model, texts: list[str]) -> list[int]:
    preds = []
    for t in texts:
        r = model.f.predict(t, 1, 0.0, "")
        preds.append(FT_MAP[r[0][1]])
    return preds


def eval_model(model, name: str, dstl_texts, dstl_labels, frmt_texts, frmt_labels):
    d = f1_score(dstl_labels, batch_predict(model, dstl_texts), pos_label=0) * 100
    f = f1_score(frmt_labels, batch_predict(model, frmt_texts), pos_label=0) * 100
    print(f"| {name:45s} | {d:5.1f}% | {f:5.1f}% |")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("Loading eval data...")
    ds = load_dataset("LCA-PORVID/dsl_tl", split="test")
    ds = ds.filter(lambda x: x["label"] in [0, 1])
    dstl_texts = [t.replace("\n", " ") for t in ds["text"]]
    dstl_labels = list(ds["label"])

    ds2 = load_dataset("hugosousa/frmt", split="test")
    pt_texts = [t for t in ds2["pt"] if t]
    br_texts = [t for t in ds2["br"] if t]
    frmt_texts = [t.replace("\n", " ") for t in pt_texts + br_texts]
    frmt_labels = [0] * len(pt_texts) + [1] * len(br_texts)

    orig_mb = os.path.getsize(MODEL_PATH) / 1e6

    print()
    print(f"| {'Model':45s} | DSL-TL | FRMT   |")
    print(f"|{'-' * 47}|--------|--------|")

    # Original
    model = ft.load_model(MODEL_PATH)
    eval_model(
        model,
        f"Original ({orig_mb:.0f} MB)",
        dstl_texts,
        dstl_labels,
        frmt_texts,
        frmt_labels,
    )
    del model

    # Quantization variants
    configs = [
        ("default", {}),
        ("qnorm", {"qnorm": True}),
        ("dsub=4", {"dsub": 4}),
        ("cutoff=100k", {"cutoff": 100000}),
        ("qnorm+cutoff=100k", {"qnorm": True, "cutoff": 100000}),
    ]

    for label, kwargs in configs:
        m = ft.load_model(MODEL_PATH)
        m.quantize(retrain=False, **kwargs)
        path = f"/tmp/q_{label.replace('=', '_').replace('+', '_')}.ftz"
        m.save_model(path)
        mb = os.path.getsize(path) / 1e6
        eval_model(
            m,
            f"Quant {label} ({mb:.0f} MB)",
            dstl_texts,
            dstl_labels,
            frmt_texts,
            frmt_labels,
        )
        del m

    print()
