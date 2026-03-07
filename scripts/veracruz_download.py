# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "huggingface-hub",
#     "polars",
# ]
# ///
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars
from huggingface_hub import hf_hub_url

REPO_ID = "bastao/VeraCruz_PT-BR"
TARGET_ROWS = 10_000_000
SENTENCES_PER_CHUNK = 4
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "veracruz_fatty"
MAX_FILE_INDEX = 1500
WORKERS = 5
SEED = 42

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_chunks(text: str) -> list[str]:
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    chunks = []
    for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
        chunk = " ".join(sentences[i : i + SENTENCES_PER_CHUNK]).strip()
        if len(chunk) > 20:
            chunks.append(chunk)
    return chunks


def fetch_and_split(lang: str, idx: int) -> list[str]:
    filename = f"{lang}/{lang}_{idx}.parquet"
    url = hf_hub_url(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    df = polars.read_parquet(url)
    texts = df["text"].drop_nulls().to_list()
    chunks = []
    for text in texts:
        chunks.extend(split_into_chunks(text))
    return chunks


def download_and_split(lang: str) -> Path:
    print(f"\n{'=' * 50}")
    print(f"Processing {lang.upper()} (target: {TARGET_ROWS:,} rows)")
    print(f"{'=' * 50}")

    indices = list(range(1, MAX_FILE_INDEX + 1))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    all_chunks: list[str] = []
    lock = threading.Lock()
    start = time.perf_counter()
    done = threading.Event()
    files_processed = 0
    files_failed = 0

    def process_file(idx: int):
        nonlocal files_processed, files_failed
        if done.is_set():
            return
        try:
            chunks = fetch_and_split(lang=lang, idx=idx)
        except Exception:
            with lock:
                files_failed += 1
            return

        with lock:
            if done.is_set():
                return
            all_chunks.extend(chunks)
            files_processed += 1

            elapsed = time.perf_counter() - start
            pct = min(len(all_chunks) / TARGET_ROWS * 100, 100)
            rate = len(all_chunks) / elapsed if elapsed > 0 else 0
            eta = (TARGET_ROWS - len(all_chunks)) / rate if rate > 0 else 0

            print(
                f"  {lang}_{idx}.parquet | +{len(chunks):,} | {len(all_chunks):,}/{TARGET_ROWS:,} ({pct:.1f}%) | {files_processed} ok, {files_failed} skip | ETA: {eta:.0f}s",  # noqa: E501
                flush=True,
            )

            if len(all_chunks) >= TARGET_ROWS:
                done.set()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pool.map(process_file, indices)

    all_chunks = all_chunks[:TARGET_ROWS]
    result = polars.DataFrame({"text": all_chunks})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{lang}_split.parquet"
    result.write_parquet(output_path)

    elapsed = time.perf_counter() - start
    print(f"\n  Saved {result.height:,} rows to {output_path} ({elapsed:.0f}s)")
    return output_path


if __name__ == "__main__":
    for lang in ["pt", "br"]:
        download_and_split(lang=lang)
    print("\nDone!")
