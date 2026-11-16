"""Append-only registry of model results at reports/model_comparison.csv."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from src.utils.config import BASELINE2_METRICS, REPORTS

COMPARISON_CSV = REPORTS / "model_comparison.csv"
COLUMNS = [
    "timestamp", "model", "train_set", "eval_set",
    "team_acc", "team_macro_f1", "issue_macro_f1", "issue_top3",
    "joint_exact", "north_star_coverage_95", "precision_at_95_threshold",
    "per_team_thr_coverage_95", "teams_below_floor", "ece",
    "n_eval", "git_commit",
    # Added 2026-11-15 when the north star moved to 90% precision.
    "north_star_coverage_90", "precision_at_90_threshold",
]


def north_star_columns(ns: dict) -> dict:
    """Registry columns for both the 95% and 90% operating points of a
    north-star summary. Missing points are left blank."""
    out = {}
    for pct in (95, 90):
        pt = (ns.get("operating_points") or {}).get(f"at_{pct}p") or {}
        out[f"north_star_coverage_{pct}"] = pt.get("coverage")
        out[f"precision_at_{pct}_threshold"] = pt.get("precision")
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPORTS.parent, text=True).strip()
    except Exception:
        return ""


def log_result(model: str, eval_set: str, metrics: dict) -> None:
    row = {c: metrics.get(c) for c in COLUMNS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["model"] = model
    row["eval_set"] = eval_set
    if row.get("git_commit") is None:
        row["git_commit"] = _git_commit()
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = not COMPARISON_CSV.exists()
    pd.DataFrame([row]).to_csv(COMPARISON_CSV, mode="a", header=header,
                               index=False, columns=COLUMNS)


def seed_from_baseline_2() -> None:
    m = json.loads((BASELINE2_METRICS / "metrics.json").read_text())
    ns = json.loads((BASELINE2_METRICS / "north_star.json").read_text())
    op = ns["operating_points"]["at_95p"]
    log_result("baseline_model_2", "test", {
        "train_set": "3k_team",
        "team_acc": m["model"]["test"]["accuracy"],
        "team_macro_f1": m["model"]["test"]["macro_f1"],
        "north_star_coverage_95": op["coverage"],
        "precision_at_95_threshold": op["precision"],
        "teams_below_floor": ",".join(ns["guardrail"]["teams_below_floor"]),
        "ece": ns["ece_test"],
        "n_eval": ns["n_test"],
    })
