# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "datasets",
#     "fasttext",
#     "scikit-learn",
#     "torch",
#     "transformers",
# ]
# ///
"""Compare our best FastText model vs paper BERT-large (PtVId) and PeroVaz on DSL-TL and FRMT."""

import time

import fasttext as ft
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FT_MAP = {"__label__PT_PT": 0, "__label__PT_BR": 1}
FASTTEXT_MODEL = "models/20260305_171908_veracruz_6M_best.bin"
FASTTEXT_MODEL_Q = "models/20260305_171908_veracruz_6M_best_dsub4.ftz"
BERT_LARGE_MODEL = "liaad/PtVId"  # Paper model (BERTimbau-large, 334M params)
BERT_BASE_MODEL = (
    "liaad/LVI_bert-base-portuguese-cased"  # Unpublished variant (bert-base)
)
PEROVAZ_MODEL = "bastao/PeroVaz_PT-BR_Classifier"
BATCH_SIZE = 256


# --- Data loading ---


def load_dstl() -> tuple[list[str], list[int]]:
    ds = load_dataset("LCA-PORVID/dsl_tl", split="test")
    ds = ds.filter(lambda x: x["label"] in [0, 1])
    texts = list(ds["text"])
    labels = list(ds["label"])
    return texts, labels


def load_frmt() -> tuple[list[str], list[int]]:
    ds = load_dataset("hugosousa/frmt", split="test")
    pt_texts = [t for t in ds["pt"] if t]
    br_texts = [t for t in ds["br"] if t]
    texts = pt_texts + br_texts
    labels = [0] * len(pt_texts) + [1] * len(br_texts)
    return texts, labels


# --- Predictors ---


def predict_fasttext(model, texts: list[str]) -> list[int]:
    preds = []
    for t in texts:
        r = model.f.predict(t.replace("\n", " "), 1, 0.0, "")
        preds.append(FT_MAP[r[0][1]])
    return preds


def predict_transformer(
    model,
    tokenizer,
    device: torch.device,
    texts: list[str],
    label_remap: dict[int, int],
) -> list[int]:
    all_preds: list[int] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(label_remap[p] for p in preds)
    return all_preds


# --- Evaluation ---


def evaluate(
    model_name: str,
    dataset_name: str,
    true_labels: list[int],
    pred_labels: list[int],
    elapsed: float,
):
    pt_pt_f1 = f1_score(true_labels, pred_labels, pos_label=0) * 100
    speed = len(true_labels) / elapsed

    print(f"\n### {model_name} — {dataset_name} ({speed:,.0f} samples/sec)")
    print(
        classification_report(
            true_labels, pred_labels, target_names=["PT-PT", "PT-BR"], digits=2
        )
    )
    print(f"  PT-PT F1: {pt_pt_f1:.2f}%")


if __name__ == "__main__":
    # Load data
    print("Loading datasets...")
    dstl_texts, dstl_labels = load_dstl()
    frmt_texts, frmt_labels = load_frmt()
    print(f"  DSL-TL: {len(dstl_texts)} samples | FRMT: {len(frmt_texts)} samples\n")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- FastText (full) ---
    print("=" * 70)
    print("FASTTEXT (our best, full)")
    print("=" * 70)
    ft_model = ft.load_model(FASTTEXT_MODEL)

    start = time.perf_counter()
    ft_dstl_preds = predict_fasttext(ft_model, dstl_texts)
    evaluate(
        "FastText (full)",
        "DSL-TL",
        dstl_labels,
        ft_dstl_preds,
        time.perf_counter() - start,
    )

    start = time.perf_counter()
    ft_frmt_preds = predict_fasttext(ft_model, frmt_texts)
    evaluate(
        "FastText (full)",
        "FRMT",
        frmt_labels,
        ft_frmt_preds,
        time.perf_counter() - start,
    )
    del ft_model

    # --- FastText (quantized) ---
    print("\n" + "=" * 70)
    print("FASTTEXT (quantized dsub=4)")
    print("=" * 70)
    ft_model_q = ft.load_model(FASTTEXT_MODEL_Q)

    start = time.perf_counter()
    ftq_dstl_preds = predict_fasttext(ft_model_q, dstl_texts)
    evaluate(
        "FastText (quantized)",
        "DSL-TL",
        dstl_labels,
        ftq_dstl_preds,
        time.perf_counter() - start,
    )

    start = time.perf_counter()
    ftq_frmt_preds = predict_fasttext(ft_model_q, frmt_texts)
    evaluate(
        "FastText (quantized)",
        "FRMT",
        frmt_labels,
        ftq_frmt_preds,
        time.perf_counter() - start,
    )
    del ft_model_q

    # --- BERT-large ---
    print("\n" + "=" * 70)
    print("PtVId paper (liaad/PtVId) — BERTimbau-large, 334M params")
    print("=" * 70)
    bert_tok = AutoTokenizer.from_pretrained(BERT_LARGE_MODEL)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        BERT_LARGE_MODEL
    ).to(device)
    bert_model.eval()
    # PtVId: label 0 = PT-PT, label 1 = PT-BR (same as ours)
    bert_remap = {0: 0, 1: 1}

    start = time.perf_counter()
    bert_dstl_preds = predict_transformer(
        bert_model, bert_tok, device, dstl_texts, bert_remap
    )
    evaluate(
        "PtVId paper (liaad/PtVId)",
        "DSL-TL",
        dstl_labels,
        bert_dstl_preds,
        time.perf_counter() - start,
    )

    start = time.perf_counter()
    bert_frmt_preds = predict_transformer(
        bert_model, bert_tok, device, frmt_texts, bert_remap
    )
    evaluate(
        "PtVId paper (liaad/PtVId)",
        "FRMT",
        frmt_labels,
        bert_frmt_preds,
        time.perf_counter() - start,
    )
    del bert_model, bert_tok

    # --- LVI (bert-base, unpublished) ---
    print("\n" + "=" * 70)
    print("LVI unpublished (liaad/LVI) — BERTimbau-base, unpublished variant")
    print("=" * 70)
    lvi_tok = AutoTokenizer.from_pretrained(BERT_BASE_MODEL)
    lvi_model = AutoModelForSequenceClassification.from_pretrained(BERT_BASE_MODEL).to(
        device
    )
    lvi_model.eval()
    lvi_remap = {0: 0, 1: 1}

    start = time.perf_counter()
    lvi_dstl_preds = predict_transformer(
        lvi_model, lvi_tok, device, dstl_texts, lvi_remap
    )
    evaluate(
        "LVI unpublished (liaad/LVI)",
        "DSL-TL",
        dstl_labels,
        lvi_dstl_preds,
        time.perf_counter() - start,
    )

    start = time.perf_counter()
    lvi_frmt_preds = predict_transformer(
        lvi_model, lvi_tok, device, frmt_texts, lvi_remap
    )
    evaluate(
        "LVI unpublished (liaad/LVI)",
        "FRMT",
        frmt_labels,
        lvi_frmt_preds,
        time.perf_counter() - start,
    )
    del lvi_model, lvi_tok

    # --- PeroVaz ---
    print("\n" + "=" * 70)
    print("PeroVaz (bastao/PeroVaz)")
    print("=" * 70)
    pv_tok = AutoTokenizer.from_pretrained(PEROVAZ_MODEL)
    pv_model = AutoModelForSequenceClassification.from_pretrained(PEROVAZ_MODEL).to(
        device
    )
    pv_model.eval()
    # PeroVaz: label 0 = BR, label 1 = PT → remap
    pv_remap = {0: 1, 1: 0}

    start = time.perf_counter()
    pv_dstl_preds = predict_transformer(pv_model, pv_tok, device, dstl_texts, pv_remap)
    evaluate(
        "PeroVaz", "DSL-TL", dstl_labels, pv_dstl_preds, time.perf_counter() - start
    )

    start = time.perf_counter()
    pv_frmt_preds = predict_transformer(pv_model, pv_tok, device, frmt_texts, pv_remap)
    evaluate("PeroVaz", "FRMT", frmt_labels, pv_frmt_preds, time.perf_counter() - start)

    # --- Summary ---
    import os

    ft_size = os.path.getsize(FASTTEXT_MODEL) / 1e6
    ftq_size = os.path.getsize(FASTTEXT_MODEL_Q) / 1e6

    print("\n" + "=" * 70)
    print("SUMMARY (PT-PT F1)")
    print("=" * 70)
    models = [
        f"FastText full ({ft_size:.0f}MB)",
        f"FastText quantized ({ftq_size:.0f}MB)",
        "PtVId paper (liaad/PtVId)",
        "LVI unpublished (liaad/LVI)",
        "PeroVaz (bastao/PeroVaz)",
    ]
    dstl_f1s = [
        f1_score(dstl_labels, ft_dstl_preds, pos_label=0) * 100,
        f1_score(dstl_labels, ftq_dstl_preds, pos_label=0) * 100,
        f1_score(dstl_labels, bert_dstl_preds, pos_label=0) * 100,
        f1_score(dstl_labels, lvi_dstl_preds, pos_label=0) * 100,
        f1_score(dstl_labels, pv_dstl_preds, pos_label=0) * 100,
    ]
    frmt_f1s = [
        f1_score(frmt_labels, ft_frmt_preds, pos_label=0) * 100,
        f1_score(frmt_labels, ftq_frmt_preds, pos_label=0) * 100,
        f1_score(frmt_labels, bert_frmt_preds, pos_label=0) * 100,
        f1_score(frmt_labels, lvi_frmt_preds, pos_label=0) * 100,
        f1_score(frmt_labels, pv_frmt_preds, pos_label=0) * 100,
    ]
    print(f"\n| {'Model':30s} | DSL-TL  | FRMT    |")
    print(f"|{'-' * 32}|---------|---------|")
    for name, d, f in zip(models, dstl_f1s, frmt_f1s):
        print(f"| {name:30s} | {d:5.2f}%  | {f:5.2f}%  |")
    print()
