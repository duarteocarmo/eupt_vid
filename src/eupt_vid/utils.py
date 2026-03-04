import inspect
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def record_experiment(
    experiment_name: str,
    details: dict[str, object],
) -> None:
    """Append an experiment result to reports/experiment_logger.md."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "experiment_logger.md"

    if not log_path.exists():
        log_path.write_text("# Experiment Log\n")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    caller = Path(inspect.stack()[1].filename).name

    lines = [
        f"\n## Experiment: {experiment_name}\n",
        f"- date: {now}\n",
        f"- script: {caller}\n",
    ]
    for key, value in details.items():
        lines.append(f"- {key}: {value}\n")

    with open(log_path, "a") as f:
        f.writelines(lines)

    print(f"\n📝 Recorded to {log_path}")
