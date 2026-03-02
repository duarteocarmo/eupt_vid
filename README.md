# eupt-vid

A fast detector for Portuguese language varieties (PT-PT vs PT-BR). Classifies text into European or Brazilian Portuguese with minimal latency.

## Scripts

| Command | Description |
|---|---|
| `uv run python src/train_journalistic.py` | fastText on journalistic subset only |
| `uv run python src/train_full.py` | fastText on all subsets combined |
| `uv run scripts/run_repro.py` | BERT baseline evaluation (PtVId, LVI) |
