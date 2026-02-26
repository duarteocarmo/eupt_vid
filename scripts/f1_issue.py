# /// script
# dependencies = [
#     "transformers==5.2.0",
#     "torch==2.10.0",
#     "scikit-learn",
#     "datasets",
# ]
# requires-python = ">=3.12"
# ///

"""
The paper's evaluate.py (acl branch) calls f1_score(dataset["label"], pred)
without average="macro". sklearn defaults to average="binary" (pos_label=1).
Label 1 = PT-BR. So the paper reports PT-BR F1 only, not macro F1.

We show this using liaad/PtVId (bert-large), which matches the paper's
BERT_F numbers exactly: 84.97% on DSL-TL and 77.25% on FRMT.
"""

from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report

LABEL2ID = {"PT-PT": 0, "PT-BR": 1}

pipe = pipeline("text-classification", model="liaad/PtVId", device="mps")


def predict(texts: list[str]) -> list[int]:
    raw = pipe(texts, batch_size=32)
    return [LABEL2ID[p["label"]] for p in raw]


# --- Load datasets ---
dsl_ds = load_dataset("LCA-PORVID/dsl_tl", split="test")
dsl_ds = dsl_ds.filter(lambda x: x["label"] in [0, 1])
dsl_true = list(dsl_ds["label"])
dsl_preds = predict(list(dsl_ds["text"]))

frmt_raw = load_dataset("hugosousa/frmt", split="test")
pt_texts = [t for t in frmt_raw["pt"] if t]
br_texts = [t for t in frmt_raw["br"] if t]
frmt_true = [0] * len(pt_texts) + [1] * len(br_texts)
frmt_preds = predict(pt_texts + br_texts)

# --- Show the issue ---
print("=== The F1 metric issue ===\n")
print("Paper's evaluate.py (acl branch):")
print('  "f1": f1_score(dataset["label"], pred)')
print(
    "  No average= → sklearn default: average='binary', pos_label=1 → PT-BR F1 only\n"
)

for name, true, preds in [
    ("DSL-TL", dsl_true, dsl_preds),
    ("FRMT", frmt_true, frmt_preds),
]:
    binary = f1_score(true, preds) * 100
    macro = f1_score(true, preds, average="macro") * 100
    print(f"--- {name} ---")
    print(
        f"  f1_score(true, pred)                  = {binary:.2f}%  ← paper reports this"
    )
    print(
        f"  f1_score(true, pred, average='macro')  = {macro:.2f}%  ← what you'd expect"
    )
    print(
        classification_report(
            true, preds, target_names=["PT-PT (0)", "PT-BR (1)"], digits=4
        )
    )
