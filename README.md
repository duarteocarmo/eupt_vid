# eupt-vid

A fast detector for Portuguese language varieties (PT-PT vs PT-BR). Classifies text into European or Brazilian Portuguese with minimal latency.

## Scripts

| Command | Description |
|---|---|
| `uv run python -m eupt_vid` | Train and evaluate fastText classifier |
| `uv run scripts/run_repro.py` | BERT baseline evaluation (PtVId, LVI) |
