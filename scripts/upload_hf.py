# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "huggingface-hub",
# ]
# ///
"""Upload fasttext-euptvid folder to Hugging Face Hub."""

from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "duarteocarmo/fasttext-euptvid"
FOLDER = Path(__file__).resolve().parent.parent / "fasttext-euptvid"


def main():
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # Upload using large folder API (resumable, multi-threaded, resilient)
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(FOLDER),
    )

    print(f"\nUploaded to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
