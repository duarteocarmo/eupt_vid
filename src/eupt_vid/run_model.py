"""Interactive CLI for PT-PT vs PT-BR classification."""

import argparse
import sys

import fasttext as ft


def predict_safe(model, text: str, k: int = 2) -> list[tuple[float, str]]:
    """Wrapper around model.predict that handles numpy 2.x compat."""
    return model.f.predict(text, k, 0.0, "")


def main():
    parser = argparse.ArgumentParser(description="PT-PT vs PT-BR classifier")
    parser.add_argument("--model", required=True, help="Path to fasttext .bin model")
    args = parser.parse_args()

    model = ft.load_model(args.model)
    print(f"Loaded {args.model}")
    print("Type a sentence and press Enter. Ctrl+C to quit.\n")

    try:
        while True:
            text = input("> ").strip()
            if not text:
                continue
            results = predict_safe(model, text, k=2)
            for prob, label in results:
                tag = label.replace("__label__", "")
                print(f"  {tag}: {prob:.3f}")
            print()
    except KeyboardInterrupt, EOFError:
        print("\nBye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
