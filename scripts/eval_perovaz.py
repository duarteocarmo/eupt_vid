# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "datasets",
#     "scikit-learn",
#     "torch",
#     "transformers",
# ]
# ///
"""Evaluate bastao/PeroVaz_PT-BR_Classifier on DSL-TL and FRMT benchmarks."""

import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "bastao/PeroVaz_PT-BR_Classifier"
# Model labels: 0=BR, 1=PT → remap to our convention: PT-PT=0, PT-BR=1
LABEL_REMAP = {0: 1, 1: 0}  # model_label → our_label
BATCH_SIZE = 256


def predict_batch(
    texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
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
        all_preds.extend(LABEL_REMAP[p] for p in preds)
    return all_preds


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


def test_dstl(model, tokenizer, device) -> float:
    print("Loading DSL-TL test...")
    ds = load_dataset("LCA-PORVID/dsl_tl", split="test")
    ds = ds.filter(lambda x: x["label"] in [0, 1])
    texts = [t.replace("\n", " ") for t in ds["text"]]
    true_labels = list(ds["label"])

    pred_labels = predict_batch(texts, model, tokenizer, device)
    return print_results(
        true_labels=true_labels,
        pred_labels=pred_labels,
        label="DSL-TL",
        best_pt_pt_f1=75,
    )


def test_frmt(model, tokenizer, device) -> float:
    print("\nLoading FRMT test...")
    ds = load_dataset("hugosousa/frmt", split="test")
    pt_texts = [t for t in ds["pt"] if t]
    br_texts = [t for t in ds["br"] if t]
    texts = pt_texts + br_texts
    true_labels = [0] * len(pt_texts) + [1] * len(br_texts)

    pred_labels = predict_batch(texts, model, tokenizer, device)
    return print_results(
        true_labels=true_labels,
        pred_labels=pred_labels,
        label="FRMT",
        best_pt_pt_f1=76,
    )


if __name__ == "__main__":
    print(f"Loading {MODEL_ID}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print(f"  Device: {device}")

    dstl_f1 = test_dstl(model, tokenizer, device)
    frmt_f1 = test_frmt(model, tokenizer, device)

    print("\n--- Summary ---")
    print(f"  DSL-TL PT-PT F1: {dstl_f1:.1f}%")
    print(f"  FRMT PT-PT F1:   {frmt_f1:.1f}%")
