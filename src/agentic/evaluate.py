"""Score the LLM stage and fit the rescue rule on validation only.

Rescue rule (team task): an asked complaint is auto-routed after all when
the LLM agrees with the stacker AND the gate score is at least `t`, with
`t` the lowest value whose 95% lower-bound precision on the validation
sample reaches target + margin. Disagreements always stay in review.
The LLM's pick replaces the stacker's team for every asked row.

Overall numbers combine the exact non-asked rows with the asked rows. When
the run was a sample of the pool, the asked rows' contribution is scaled by
pool / sample and marked as estimated.

Usage:
    python -m src.agentic.evaluate --fit val --score test
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.agentic import config as C
from src.agentic.tasks import load_split_data
from src.ensemble.thresholds import choose_threshold, wilson_lower
from src.utils.config import NORTH_STAR_TARGET


def _load(split: str) -> tuple[pd.DataFrame, dict]:
    p = pd.read_parquet(C.ARTIFACT_DIR / f"predictions_{split}.parquet")
    info = json.loads((C.ARTIFACT_DIR / f"run_{split}.json").read_text())
    return p, info


def fit_rescue(val: pd.DataFrame, target: float = NORTH_STAR_TARGET,
               margin: float = 0.0025, min_routed: int = 50) -> dict:
    m = val[val.team_asked & (val.llm_team == val.stack_team)]
    if len(m) < min_routed:
        return {"threshold": None, "reason": "too few agreeing rows"}
    correct = (m.llm_team == m.actual_team_id).to_numpy()
    t = choose_threshold(m.gate_score.to_numpy(), correct, target, margin,
                         min_routed=min_routed, wilson=True)
    out = {"threshold": t, "n_agree": int(len(m)),
           "agree_share": round(float(val.team_asked.sum() and
                                      len(m) / val.team_asked.sum()), 4)}
    if t is not None:
        r = m.gate_score >= t
        out.update({"val_rescued_of_asked": round(float(r.mean() * out["agree_share"]), 4),
                    "val_rescued_precision": round(float(correct[r].mean()), 4)})
    return out


def team_report(p: pd.DataFrame, info: dict, d, rule: dict) -> dict:
    m = p[p.team_asked]
    pool, n = info.get("team_pool", 0), len(m)
    scale = pool / n if n else 0.0
    before_acc = float((m.stack_team == m.actual_team_id).mean()) if n else None
    after_acc = float((m.llm_team == m.actual_team_id).mean()) if n else None
    # Rescue on the sample
    t = rule.get("threshold")
    resc = m[(m.llm_team == m.stack_team) & (m.gate_score >= (t if t is not None else np.inf))]
    resc_n, resc_ok = len(resc), int((resc.llm_team == resc.actual_team_id).sum())
    # Exact base: rows the gate already auto-routed
    auto = d.auto_route
    base_n, base_ok = int(auto.sum()), int((d.stack_team == d.y_team)[auto].sum())
    N = len(d)
    new_n = base_n + resc_n * scale
    new_ok = base_ok + resc_ok * scale
    acc_overall_before = float((d.stack_team == d.y_team).mean())
    acc_overall_after = (acc_overall_before * N
                         + ((after_acc or 0) - (before_acc or 0)) * n * scale) / N
    return {
        "asked_rows": n, "asked_pool": pool, "sample_scale": round(scale, 3),
        "asked_accuracy_stacker": None if n == 0 else round(before_acc, 4),
        "asked_accuracy_llm": None if n == 0 else round(after_acc, 4),
        "rescue_threshold": t,
        "rescued_rows_sample": resc_n,
        "rescued_precision": None if resc_n == 0 else round(resc_ok / resc_n, 4),
        "rescued_precision_lower95": None if resc_n == 0 else
            round(float(wilson_lower(resc_ok, resc_n)), 4),
        "coverage_before": round(base_n / N, 4),
        "precision_before": round(base_ok / base_n, 4) if base_n else None,
        "coverage_after": round(new_n / N, 4),
        "precision_after": round(new_ok / new_n, 4) if new_n else None,
        "team_accuracy_overall_before": round(acc_overall_before, 4),
        "team_accuracy_overall_after": round(acc_overall_after, 4),
        "estimated_from_sample": scale > 1.0001,
    }


def issue_report(p: pd.DataFrame, info: dict, d) -> dict:
    m = p[p.issue_asked]
    pool, n = info.get("issue_pool", 0), len(m)
    scale = pool / n if n else 0.0
    y = m.actual_issue_id
    before = float((m.issue_cand1 == y).mean()) if n else None
    after = float((m.llm_issue == y).mean()) if n else None
    top3 = float(((m.issue_cand1 == y) | (m.issue_cand2 == y)
                  | (m.issue_cand3 == y)).mean()) if n else None
    # Overall top-1 before: ML top-1 inside the stacker's team on every row.
    from src.agentic.tasks import issue_candidates
    names, _ = issue_candidates(d.Q, d.stack_team)
    overall_before = float((names[:, 0] == d.y_issue).mean())
    N = len(d)
    overall_after = overall_before + ((after or 0) - (before or 0)) * n * scale / N
    return {
        "asked_rows": n, "asked_pool": pool, "sample_scale": round(scale, 3),
        "asked_top1_ml": None if n == 0 else round(before, 4),
        "asked_top1_llm": None if n == 0 else round(after, 4),
        "asked_top3_ml": None if n == 0 else round(top3, 4),
        "overall_top1_before": round(overall_before, 4),
        "overall_top1_after": round(overall_after, 4),
        "estimated_from_sample": scale > 1.0001,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fit", default="val")
    ap.add_argument("--score", nargs="+", default=["test"])
    ap.add_argument("--margin", type=float, default=0.0025)
    args = ap.parse_args()

    val, _ = _load(args.fit)
    rule = fit_rescue(val, NORTH_STAR_TARGET, args.margin)
    print(f"[agentic] rescue rule fitted on {args.fit}: {rule}")
    report = {"fit_split": args.fit, "rescue_rule": rule, "splits": {}}
    for split in [args.fit, *args.score]:
        p, info = _load(split)
        d = load_split_data(split)
        rep = {"team": team_report(p, info, d, rule),
               "issue": issue_report(p, info, d), "model": info.get("model")}
        report["splits"][split] = rep
        t, i = rep["team"], rep["issue"]
        print(f"\n== {split} ({info.get('model')})")
        print(f"  team  asked {t['asked_rows']:,}/{t['asked_pool']:,}: accuracy "
              f"{t['asked_accuracy_stacker']} -> {t['asked_accuracy_llm']}")
        print(f"  routing: {t['coverage_before']:.4f} @ {t['precision_before']} "
              f"-> {t['coverage_after']:.4f} @ {t['precision_after']} "
              f"(rescued {t['rescued_rows_sample']} sample rows at "
              f"{t['rescued_precision']}, lower95 {t['rescued_precision_lower95']})"
              + ("  [estimated from sample]" if t["estimated_from_sample"] else ""))
        print(f"  team accuracy overall {t['team_accuracy_overall_before']} -> "
              f"{t['team_accuracy_overall_after']}")
        print(f"  issue asked {i['asked_rows']:,}/{i['asked_pool']:,}: top-1 "
              f"{i['asked_top1_ml']} -> {i['asked_top1_llm']} (top-3 ceiling "
              f"{i['asked_top3_ml']}); overall top-1 {i['overall_top1_before']} -> "
              f"{i['overall_top1_after']}")
    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (C.ARTIFACT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"\n[agentic] wrote {C.ARTIFACT_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
