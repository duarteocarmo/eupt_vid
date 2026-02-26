# /// script
# dependencies = [
#     "transformers==5.2.0",
#     "torch==2.10.0",
#     "scikit-learn",
#     "datasets",
# ]
# requires-python = ">=3.12"
# ///

from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

LABEL2ID = {"PT-PT": 0, "PT-BR": 1}

MODELS = [
    ("liaad/PtVId", "PtVId (bert-large)"),
    ("liaad/LVI_bert-base-portuguese-cased", "LVI (bert-base, paper)"),
]

PAPER_BERT = {"dsl_tl": 81.26, "frmt": 68.81}


def evaluate(pipe, texts: list[str], true_ids: list[int]) -> dict:
    raw = pipe(texts, batch_size=32)
    pred_ids = [LABEL2ID[p["label"]] for p in raw]
    result = {
        "f1_binary": f1_score(true_ids, pred_ids) * 100,
        "f1_macro": f1_score(true_ids, pred_ids, average="macro") * 100,
        "accuracy": accuracy_score(true_ids, pred_ids) * 100,
    }
    print(f"  F1 (binary): {result['f1_binary']:.2f}%")
    print(f"  F1 (macro):  {result['f1_macro']:.2f}%")
    print(f"  Accuracy:    {result['accuracy']:.2f}%")
    print(classification_report(true_ids, pred_ids, target_names=["PT-PT", "PT-BR"]))
    return result


# --- Load datasets once ---
print("Loading DSL-TL...")
dsl_ds = load_dataset("LCA-PORVID/dsl_tl", split="test")
dsl_ds = dsl_ds.filter(lambda x: x["label"] in [0, 1])
dsl_texts = list(dsl_ds["text"])
dsl_true = list(dsl_ds["label"])
print(f"  {len(dsl_texts)} documents (paper expects 857)\n")

print("Loading FRMT...")
frmt_raw = load_dataset("hugosousa/frmt", split="test")
pt_texts = [t for t in frmt_raw["pt"] if t]
br_texts = [t for t in frmt_raw["br"] if t]
frmt_texts = pt_texts + br_texts
frmt_true = [0] * len(pt_texts) + [1] * len(br_texts)
print(f"  {len(frmt_texts)} documents ({len(pt_texts)} PT-PT, {len(br_texts)} PT-BR)")
print("  Paper expects: 5,226 (2,614 PT-PT, 2,612 PT-BR)\n")

# --- Evaluate each model ---
results = {}
for model_name, label in MODELS:
    print(f"{'=' * 60}")
    print(f"Model: {label} ({model_name})")
    print(f"{'=' * 60}")
    pipe = pipeline("text-classification", model=model_name, device="mps")

    print("\n  DSL-TL:")
    dsl_result = evaluate(pipe=pipe, texts=dsl_texts, true_ids=dsl_true)

    print("  FRMT:")
    frmt_result = evaluate(pipe=pipe, texts=frmt_texts, true_ids=frmt_true)

    results[label] = {"dsl_tl": dsl_result, "frmt": frmt_result}
    del pipe

# --- Summary ---
labels = list(results.keys())
paper_label = labels[-1]  # LVI is the paper's model

print(f"\n{'=' * 60}")
print("Summary (binary F1)")
print(f"{'=' * 60}")
header = f"{'Dataset':<10} {'Paper BERT':>12}"
for label in labels:
    header += f" {label:>22}"
header += f" {'Delta':>10}"
print(header)

for dataset, key in [("DSL-TL", "dsl_tl"), ("FRMT", "frmt")]:
    row = f"{dataset:<10} {PAPER_BERT[key]:>11.2f}%"
    for label in labels:
        row += f" {results[label][key]['f1_binary']:>21.2f}%"
    delta = results[paper_label][key]["f1_binary"] - PAPER_BERT[key]
    row += f" {delta:>+9.2f}pp"
    print(row)
