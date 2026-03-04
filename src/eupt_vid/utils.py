from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def record_experiment(
    script_name: str,
    experiment_name: str,
    metrics: dict[str, float | int | str],
) -> None:
    """Append an experiment result to reports/experiment_logger.md."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "experiment_logger.md"

    if not log_path.exists():
        log_path.write_text(
            "# Experiment Log\n\n"
            "| Date | Experiment | Script | Metrics |\n"
            "|------|------------|--------|---------|\n"
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
    line = f"| {now} | {experiment_name} | `{script_name}` | {metrics_str} |\n"

    with open(log_path, "a") as f:
        f.write(line)

    print(f"\n📝 Recorded to {log_path}")
