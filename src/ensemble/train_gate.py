"""Fit the routing gate on validation, freeze it, and score it on test.

Protocol (same spirit as improved_model_1):
  1. Out-of-fold gate scores on validation (grouped by leakage_group_id).
  2. The north-star threshold (90% precision) is chosen on those out-of-fold
     scores.
  3. The gate is refitted on all of validation and applied once to test.
The primary model alone is scored on the same joined rows for comparison.

Usage:
    python -m src.ensemble.train_gate \
        --primary improved_model_1 --secondary improved_model_2
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.ensemble import gate as G
from src.ensemble.data import assemble, load_groups, read_predictions
from src.utils.config import METRICS_ROOT, NORTH_STAR_KEY
from src.utils.routing import north_star_summary

ARTIFACT_DIR = METRICS_ROOT / "ensemble_gate"


def frame_features(df: pd.DataFrame, secondary_names: list[str],
                   team_classes: list[str]) -> np.ndarray:
    return G.build_features(
        df["primary_team"].to_numpy(), df["primary_conf"].to_numpy(),
        [df[f"{s}__team"].to_numpy() for s in secondary_names],
        [df[f"{s}__conf"].to_numpy() for s in secondary_names],
        df["n_words"].to_numpy(), team_classes,
    )


def fit_and_evaluate(fit_df: pd.DataFrame, eval_df: pd.DataFrame,
                     primary_name: str, secondary_names: list[str],
                     kind: str = "logreg", n_splits: int = 5,
                     team_names: dict[str, str] | None = None) -> dict:
    """Fit the gate on `fit_df`, pick the threshold there, score on `eval_df`.

    Returns {"bundle", "gate", "primary_alone", "curves", "teams"} where the
    two summaries come from the shared north-star harness.
    """
    team_names = team_names or {}
    team_classes = sorted(set(fit_df["actual_team_id"].astype(str))
                          | set(fit_df["primary_team"].astype(str)))
    X_fit = frame_features(fit_df, secondary_names, team_classes)
    X_eval = frame_features(eval_df, secondary_names, team_classes)
    fit_correct = (fit_df["primary_team"].to_numpy()
                   == fit_df["actual_team_id"].to_numpy())
    eval_correct = (eval_df["primary_team"].to_numpy()
                    == eval_df["actual_team_id"].to_numpy())
    groups = fit_df["group"].to_numpy() if "group" in fit_df.columns else None

    oof = G.cross_fit_scores(X_fit, fit_correct, groups=groups, kind=kind,
                             n_splits=n_splits)
    estimator = G.fit_gate(X_fit, fit_correct, kind=kind)
    eval_scores = estimator.predict_proba(X_eval)[:, 1]

    y_eval = eval_df["actual_team_id"].to_numpy()
    p_eval = eval_df["primary_team"].to_numpy()
    gate_ns, curves, teams = north_star_summary(
        oof, fit_correct, eval_scores, eval_correct, y_eval, p_eval,
        team_names)
    base_ns, _, _ = north_star_summary(
        fit_df["primary_conf"].to_numpy(), fit_correct,
        eval_df["primary_conf"].to_numpy(), eval_correct, y_eval, p_eval,
        team_names)

    main = gate_ns["operating_points"].get(NORTH_STAR_KEY)
    bundle = G.make_bundle(estimator, kind, team_classes, primary_name,
                           secondary_names,
                           None if main is None else main["threshold"])
    return {"bundle": bundle, "gate": gate_ns, "primary_alone": base_ns,
            "curves": curves, "teams": teams, "eval_scores": eval_scores}


def _point(ns: dict, key: str = NORTH_STAR_KEY) -> dict | None:
    return ns["operating_points"].get(key)


def comparison(result: dict) -> dict:
    out = {}
    for key in ("at_95p", "at_90p"):
        g, b = _point(result["gate"], key), _point(result["primary_alone"], key)
        out[key] = {
            "gate": None if g is None else
            {k: g[k] for k in ("threshold", "coverage", "precision")},
            "primary_alone": None if b is None else
            {k: b[k] for k in ("threshold", "coverage", "precision")},
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", default="improved_model_1")
    p.add_argument("--secondary", nargs="+", default=["improved_model_2"])
    p.add_argument("--kind", default="both", choices=["logreg", "hgb", "both"])
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    from src.utils.labels import team_names
    from src.utils.registry import log_result, north_star_columns

    frames = {}
    for split in ("val", "test"):
        frames[split] = assemble(
            read_predictions(args.primary, split),
            {s: read_predictions(s, split) for s in args.secondary},
            groups=load_groups(split),
        )
        print(f"[gate] {split}: {len(frames[split]):,} joined rows")

    kinds = ["logreg", "hgb"] if args.kind == "both" else [args.kind]
    results = {k: fit_and_evaluate(frames["val"], frames["test"], args.primary,
                                   args.secondary, kind=k,
                                   n_splits=args.n_splits,
                                   team_names=team_names) for k in kinds}

    # Model choice uses validation only: highest out-of-fold coverage at the
    # north-star threshold. Test numbers are reported for every kind but never used
    # to choose.
    def _val_cov(r):
        pt = _point(r["gate"])
        return -1.0 if pt is None else pt["validation_coverage"]
    best_kind = max(kinds, key=lambda k: _val_cov(results[k]))
    best = results[best_kind]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["bundle"], ARTIFACT_DIR / "gate.joblib", compress=3)
    (ARTIFACT_DIR / "north_star.json").write_text(
        json.dumps(best["gate"], indent=2))
    (ARTIFACT_DIR / "comparison.json").write_text(json.dumps({
        "primary": args.primary, "secondary": args.secondary,
        "chosen_kind": best_kind,
        "by_kind": {k: comparison(r) for k, r in results.items()},
    }, indent=2))
    best["curves"].to_csv(ARTIFACT_DIR / "coverage_curve.csv", index=False)
    if best["teams"] is not None:
        best["teams"].to_csv(ARTIFACT_DIR / "per_team_routing.csv",
                             index=False)

    main_pt = _point(best["gate"])
    guard = best["gate"].get("guardrail") or {}
    test = frames["test"]
    log_result(G.MODEL_NAME, "test", {
        "train_set": f"gate({args.primary}+{'+'.join(args.secondary)})",
        "team_acc": float((test["primary_team"]
                           == test["actual_team_id"]).mean()),
        **north_star_columns(best["gate"]),
        "teams_below_floor": ",".join(guard.get("teams_below_floor", [])),
        "ece": best["gate"]["ece_test"],
        "n_eval": len(test),
    })

    print(json.dumps(comparison(best), indent=2))
    print(f"[gate] chose {best_kind}; wrote {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
