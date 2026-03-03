# eupt-vid

A fast detector for Portuguese language varieties (PT-PT vs PT-BR). Classifies text into European or Brazilian Portuguese with minimal latency.

## Scripts

| Command | Description |
|---|---|
| `uv run python src/train_journalistic.py` | fastText on journalistic subset only |
| `uv run python src/train_full.py` | fastText on all subsets, no balancing |
| `uv run python src/train_balanced.py` | fastText balanced per category |
| `uv run python src/train_autotune.py` | fastText balanced + autotune (10min) |
| `uv run python src/train_journalistic_autotune.py` | fastText journalistic 200K/class + autotune (10min) |
| `uv run scripts/run_repro.py` | BERT baseline evaluation (PtVId, LVI) |
