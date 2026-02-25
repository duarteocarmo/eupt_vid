# /// script
# dependencies = [
#     "transformers==5.2.0",
#     "torch==2.10.0",
#     "scikit-learn",
#     "polars",
#     "httpx",
# ]
# requires-python = ">=3.12"
# ///

import io
import httpx
import polars
from transformers import pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report

pipe = pipeline("text-classification", model="liaad/PtVId", top_k=None)


def predict(texts):
    raw = pipe(texts, batch_size=32)
    return [max(p, key=lambda x: x["score"])["label"] for p in raw]


# --- DSL-TL (from original repo: PT_dev.tsv) ---
print("=== DSL-TL ===")
dsl_tl_url = "https://raw.githubusercontent.com/LanguageTechnologyLab/DSL-TL/main/DSL-TL-Corpus/PT-DSL-TL/PT_dev.tsv"
raw_text = httpx.get(dsl_tl_url).text
df = polars.read_csv(
    io.StringIO(raw_text),
    separator="\t",
    has_header=False,
    new_columns=["id", "text", "label"],
)

print(f"Total documents: {len(df)}")
print(f"Label distribution:")
print(df.group_by("label").agg(polars.len()).sort("label"))

# Exclude "PT" (Both/Neither) as the paper did
df = df.filter(polars.col("label") != "PT")
bp_count = df.filter(polars.col("label") == "PT-BR").height
ep_count = df.filter(polars.col("label") == "PT-PT").height
print(f"\nAfter excluding 'PT': {len(df)} documents ({bp_count} BP, {ep_count} EP)")
print("Paper expects: 857 documents (588 BP, 269 EP)")
assert len(df) == 857, f"Expected 857, got {len(df)}"
assert bp_count == 588, f"Expected 588 BP, got {bp_count}"
assert ep_count == 269, f"Expected 269 EP, got {ep_count}"
print("Numbers match!\n")

true_labels = df["label"].to_list()
preds = predict(df["text"].to_list())

f1 = f1_score(true_labels, preds, average="macro")
acc = accuracy_score(true_labels, preds)
print(f"F1 (macro): {f1 * 100:.2f}% (paper BERTd: 84.97%, BERT: 81.26%)")
print(f"Accuracy:   {acc * 100:.2f}%")
print(classification_report(true_labels, preds))


# --- FRMT (from google-research/google-research) ---
print("\n=== FRMT ===")
frmt_base = "https://raw.githubusercontent.com/google-research/google-research/master/frmt/dataset"
frmt_rows = []
for bucket in ["lexical", "entity", "random"]:
    for region, label in [("pt-BR", "PT-BR"), ("pt-PT", "PT-PT")]:
        url = f"{frmt_base}/{bucket}_bucket/pt_{bucket}_test_en_{region}.tsv"
        raw_text = httpx.get(url).text
        tsv = polars.read_csv(
            io.StringIO(raw_text),
            separator="\t",
            has_header=False,
            new_columns=["en", "text"],
            quote_char=None,
        )
        frmt_rows.append(
            tsv.select("text").with_columns(polars.lit(label).alias("label"))
        )
        print(f"  {bucket}/{region}: {len(tsv)} documents")

frmt_df = polars.concat(frmt_rows)
frmt_bp = frmt_df.filter(polars.col("label") == "PT-BR").height
frmt_ep = frmt_df.filter(polars.col("label") == "PT-PT").height
print(f"\nTotal: {len(frmt_df)} documents ({frmt_bp} BP, {frmt_ep} EP)")
print(f"Paper expects: 5,226 documents (2,612 BP, 2,614 EP)\n")

frmt_true = frmt_df["label"].to_list()
frmt_preds = predict(frmt_df["text"].to_list())

frmt_f1 = f1_score(frmt_true, frmt_preds, average="macro")
frmt_acc = accuracy_score(frmt_true, frmt_preds)
print(f"F1 (macro): {frmt_f1 * 100:.2f}% (paper BERTd: 77.25%, BERT: 68.81%)")
print(f"Accuracy:   {frmt_acc * 100:.2f}%")
print(classification_report(frmt_true, frmt_preds))

# --- Summary ---
print("\n=== Summary ===")
print(f"{'Dataset':<10} {'Paper BERTd':>12} {'Paper BERT':>12} {'Us (raw)':>12}")
print(f"{'DSL-TL':<10} {'84.97%':>12} {'81.26%':>12} {f'{f1 * 100:.2f}%':>12}")
print(f"{'FRMT':<10} {'77.25%':>12} {'68.81%':>12} {f'{frmt_f1 * 100:.2f}%':>12}")
