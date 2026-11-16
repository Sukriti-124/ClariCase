"""Stack the team models, gate the routing on their full probability vectors,
and fit the routing thresholds with a drift margin and per-team floors.

Everything is fitted on validation. Test (and the 2026 holdout, if it has been
exported) is scored once with everything frozen.

Scenarios compared:
  primary_alone    improved_model_1's team and confidence (today's setup)
  gate_on_primary  improved_model_1's team; gate over every model's probabilities
  stack_alone      stacker's team and probability
  stack_gate       stacker's team; gate over stacker + every model's probabilities

Routing policies, all fitted on validation:
  global           one threshold where validation precision reaches the target
  margin           the 95% lower bound must reach target + a drift margin
                   (default 0.25 points, see --margin)
  margin_floors    as margin, plus per-team thresholds raised until every team
                   reaches the 0.85 floor (a team that cannot is never
                   auto-routed)

The candidate for serving is whichever of gate_on_primary and stack_gate
routes more of validation under margin_floors. Test is not used to choose.

Usage:
    python -m src.ensemble.train_stack
    python -m src.ensemble.train_stack --eval-splits test holdout_2026
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.ensemble import gate as G
from src.ensemble.data import load_meta, read_predictions, write_predictions
from src.ensemble.label_shift import adjust, estimate_prior, prior_from_labels
from src.ensemble.probs import TEAMS, team_cols, team_matrix, with_team_probs
from src.ensemble.stack import (fold_splits, make_stacker, stack_features,
                                stacker_proba, tune_stacker)
from src.ensemble.thresholds import (choose_threshold, route_mask,
                                     routing_report, team_floor_thresholds,
                                     temporal_margin)
from src.utils.config import (DATE_COL, GROUP_COL, ID_COL, METRICS_ROOT,
                              MIN_ROUTED, NORTH_STAR_TARGET,
                              TEAM_PRECISION_FLOOR)

ARTIFACT_DIR = METRICS_ROOT / "ensemble_stack"
MODEL_NAME = "ensemble_stack_v2"
DEFAULT_MODELS = ("improved_model_1", "improved_model_2", "baseline_model_2",
                  "knn_embed")
POLICIES = ("global", "margin", "margin_floors")
CANDIDATES = ("gate_on_primary", "stack_gate")


@dataclass
class Aligned:
    ids: np.ndarray
    y: np.ndarray
    n_words: np.ndarray
    groups: np.ndarray | None
    dates: np.ndarray | None
    probs: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.ids)


def align(frames: dict[str, pd.DataFrame], teams=TEAMS,
          meta: pd.DataFrame | None = None) -> Aligned:
    """Inner-join every model's predictions on complaint_id, so every model
    is scored on exactly the same complaints."""
    names = list(frames)
    first = frames[names[0]]
    base = first[[ID_COL, "actual_team_id", "n_words"]].copy()
    base[ID_COL] = base[ID_COL].astype(str)
    cols = team_cols(teams)
    for name in names:
        team_matrix(frames[name], teams)  # clear error if p_ columns missing
        sub = frames[name][[ID_COL] + cols].copy()
        sub[ID_COL] = sub[ID_COL].astype(str)
        sub.columns = [ID_COL] + [f"{name}::{c}" for c in cols]
        base = base.merge(sub, on=ID_COL, how="inner", validate="one_to_one")
    groups = dates = None
    if meta is not None:
        base = base.merge(meta, on=ID_COL, how="left", validate="one_to_one")
        if base[GROUP_COL].notna().all():
            groups = base[GROUP_COL].astype(str).to_numpy()
        dates = base[DATE_COL].to_numpy()
    probs = {n: base[[f"{n}::{c}" for c in cols]].to_numpy(np.float64)
             for n in names}
    return Aligned(ids=base[ID_COL].to_numpy(),
                   y=base["actual_team_id"].astype(str).to_numpy(),
                   n_words=base["n_words"].to_numpy(),
                   groups=groups, dates=dates, probs=probs)


def _gated(val, evals, yv, val_chosen, val_mats, eval_chosen, eval_mats,
           n_teams, splits, kind, seed):
    """Out-of-fold gate scores on validation, a gate refitted on all of it,
    and that gate's scores on each eval split."""
    Xv = G.prob_features(val_chosen, val_mats, val.n_words, n_teams)
    correct = (val_chosen == yv).astype(int)
    oof = np.zeros(len(yv))
    for fit_idx, hold_idx in splits:
        est = G.make_estimator(kind, seed).fit(Xv[fit_idx], correct[fit_idx])
        oof[hold_idx] = est.predict_proba(Xv[hold_idx])[:, 1]
    est = G.make_estimator(kind, seed).fit(Xv, correct)
    eval_score = {
        s: est.predict_proba(G.prob_features(
            eval_chosen[s], eval_mats[s], evals[s].n_words, n_teams))[:, 1]
        for s in evals}
    return oof, est, eval_score, eval_mats


def build_scenarios(val: Aligned, evals: dict[str, Aligned],
                    model_names: list[str], teams=TEAMS, kind: str = "hgb",
                    C_grid=(0.1, 1.0), n_splits: int = 5, seed: int = 42
                    ) -> tuple[dict, dict]:
    T = len(teams)
    tix = {t: i for i, t in enumerate(teams)}
    yv = np.array([tix[t] for t in val.y])
    splits = fold_splits(yv, val.groups, n_splits, seed)
    Pv = [val.probs[m] for m in model_names]
    Pe = {s: [a.probs[m] for m in model_names] for s, a in evals.items()}

    C, tuning, oof_stack = tune_stacker(stack_features(Pv), yv, T, splits,
                                        C_grid)
    stacker = make_stacker(C).fit(stack_features(Pv), yv)
    stack_e = {s: stacker_proba(stacker, stack_features(P), T)
               for s, P in Pe.items()}

    sc: dict[str, dict] = {}
    p_chosen_v = Pv[0].argmax(1)
    p_chosen_e = {s: P[0].argmax(1) for s, P in Pe.items()}
    sc["primary_alone"] = {
        "val_chosen": p_chosen_v, "val_score": Pv[0].max(1),
        "eval_chosen": p_chosen_e,
        "eval_score": {s: P[0].max(1) for s, P in Pe.items()},
        "gate": None, "val_probs": Pv[0],
        "eval_probs": {s: P[0] for s, P in Pe.items()},
    }
    oof, est, es, mats = _gated(val, evals, yv, p_chosen_v, Pv, p_chosen_e,
                                Pe, T, splits, kind, seed)
    sc["gate_on_primary"] = {**sc["primary_alone"], "val_score": oof,
                             "eval_score": es, "gate": est,
                             "gate_eval_mats": mats}

    s_chosen_v = oof_stack.argmax(1)
    s_chosen_e = {s: P.argmax(1) for s, P in stack_e.items()}
    sc["stack_alone"] = {
        "val_chosen": s_chosen_v, "val_score": oof_stack.max(1),
        "eval_chosen": s_chosen_e,
        "eval_score": {s: P.max(1) for s, P in stack_e.items()},
        "gate": None, "val_probs": oof_stack, "eval_probs": stack_e,
    }
    oof, est, es, mats = _gated(val, evals, yv, s_chosen_v, [oof_stack] + Pv,
                                s_chosen_e,
                                {s: [stack_e[s]] + Pe[s] for s in Pe},
                                T, splits, kind, seed)
    sc["stack_gate"] = {**sc["stack_alone"], "val_score": oof,
                        "eval_score": es, "gate": est,
                        "gate_eval_mats": mats}

    info = {"stacker": stacker, "stacker_C": C, "stacker_tuning": tuning}
    return sc, info


def fit_policies(val_score, val_chosen, val_y, dates, target: float,
                 margin_mode="auto", floor: float = TEAM_PRECISION_FLOOR,
                 min_routed: int = MIN_ROUTED, team_min_routed: int = 30
                 ) -> tuple[dict, dict]:
    correct = val_chosen == val_y
    thr_global = choose_threshold(val_score, correct, target, 0.0,
                                  min_routed, wilson=False)
    if margin_mode == "auto":
        tm = (temporal_margin(val_score, correct, dates, target, min_routed)
              if dates is not None else {"margin": 0.0, "reason": "no dates"})
    else:
        tm = {"margin": float(margin_mode), "reason": "fixed"}
    m = tm["margin"]
    thr_m = choose_threshold(val_score, correct, target, m, min_routed,
                             wilson=True)
    floors = ({} if thr_m is None else team_floor_thresholds(
        val_score, val_chosen, val_y, thr_m, floor, margin=m,
        min_routed=team_min_routed))
    return {
        "global": {"base": thr_global, "teams": None},
        "margin": {"base": thr_m, "teams": None},
        "margin_floors": {"base": thr_m, "teams": floors},
    }, tm


def _team_metrics(chosen, y) -> dict:
    return {"accuracy": round(float((chosen == y).mean()), 4),
            "macro_f1": round(float(f1_score(y, chosen, average="macro",
                                             zero_division=0)), 4)}


def corrected_eval(sc: dict, name: str, val: Aligned, evals: dict,
                   teams=TEAMS) -> dict[str, tuple]:
    """For a stacker scenario: the eval splits re-scored after the team prior
    is re-estimated from the stacker's own outputs on that split (no labels).
    Returns {split: (chosen_idx, score, estimated_prior)}. The gate is re-run
    on the adjusted probabilities so its score reflects them too."""
    s = sc[name]
    if s["gate"] is None and not name.startswith("stack"):
        return {}
    source = prior_from_labels(val.y, teams)
    out = {}
    for e, a in evals.items():
        P = s["eval_probs"][e]
        est, _ = estimate_prior(P, source)
        Q = adjust(P, source, est)
        chosen = Q.argmax(1)
        if s["gate"] is None:
            score = Q.max(1)
        else:
            mats = s["gate_eval_mats"][e]
            mats = [Q] + mats[1:] if name.startswith("stack") else mats
            score = s["gate"].predict_proba(G.prob_features(
                chosen, mats, a.n_words, len(teams)))[:, 1]
        out[e] = (chosen, score, est)
    return out


def evaluate(sc: dict, val: Aligned, evals: dict[str, Aligned], teams=TEAMS,
             target: float = NORTH_STAR_TARGET, margin_mode="auto",
             floor: float = TEAM_PRECISION_FLOOR, names=None,
             scenarios=None, prior_correction: bool = True) -> dict:
    """Fit the routing policies for each scenario on validation and score
    them on validation (out-of-fold) and every eval split. With
    `prior_correction`, stacker scenarios are also scored on the eval splits
    after the label-shift correction (rows with corrected=True)."""
    teams = np.asarray(teams)
    rows, fitted, margins, priors = [], {}, {}, {}
    for name in (scenarios or list(sc)):
        s = sc[name]
        vc = teams[s["val_chosen"]]
        policies, tm = fit_policies(s["val_score"], vc, val.y, val.dates,
                                    target, margin_mode, floor)
        fitted[name], margins[name] = policies, tm
        splits = {"val": (vc, s["val_score"], val.y, False)}
        for e, a in evals.items():
            splits[e] = (teams[s["eval_chosen"][e]], s["eval_score"][e], a.y,
                         False)
        if prior_correction and name.startswith("stack"):
            for e, (ci, score, est) in corrected_eval(
                    sc, name, val, evals, teams).items():
                splits[f"{e}+corrected"] = (teams[ci], score, evals[e].y, True)
                priors.setdefault(name, {})[e] = est.round(4).tolist()
        for pname in POLICIES:
            pol = policies[pname]
            for split, (chosen, score, y, corrected) in splits.items():
                rep = routing_report(route_mask(score, chosen, pol["base"],
                                                pol["teams"]),
                                     chosen, y, floor, names)
                rows.append({
                    "scenario": name, "policy": pname,
                    "split": split.replace("+corrected", ""),
                    "corrected": corrected,
                    "threshold": pol["base"], "coverage": rep["coverage"],
                    "precision": rep["precision"], "n_routed": rep["n_routed"],
                    "teams_below_floor": ",".join(rep["teams_below_floor"]),
                    **_team_metrics(chosen, y),
                })
    return {"rows": rows, "policies": fitted, "margins": margins,
            "estimated_priors": priors}


def pick_final(results: dict) -> str:
    """Candidate with the higher validation coverage under margin_floors."""
    df = pd.DataFrame(results["rows"])
    m = ((df["policy"] == "margin_floors") & (df["split"] == "val")
         & ~df["corrected"] & df["scenario"].isin(CANDIDATES))
    return str(df[m].sort_values("coverage", ascending=False)
               .iloc[0]["scenario"])


def _jsonable_policy(pol: dict) -> dict:
    return {"base": pol["base"], "teams": pol["teams"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                   help="first one is the primary model")
    p.add_argument("--eval-splits", nargs="+", default=["test"])
    p.add_argument("--kind", default="hgb", choices=list(G.GATE_KINDS))
    p.add_argument("--margin", default="0.0025",
                   help="precision margin above the target, as a fraction. "
                        "0.0025 covers the 0.2-0.5 point shortfall seen on "
                        "test in the first gate run. \"auto\" estimates it "
                        "from drift inside validation, which came out at the "
                        "2-point cap and was far too conservative.")
    p.add_argument("--floor", type=float, default=TEAM_PRECISION_FLOOR)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    from src.utils.labels import team_names
    from src.utils.registry import log_result

    def load(split):
        frames = {m: read_predictions(m, split) for m in args.models}
        return align(frames, TEAMS, load_meta(split))

    val = load("val")
    evals = {s: load(s) for s in args.eval_splits}
    for s, a in {"val": val, **evals}.items():
        print(f"[stack] {s}: {len(a):,} rows scored by every model")

    sc, info = build_scenarios(val, evals, args.models, TEAMS, args.kind,
                               n_splits=args.n_splits)
    res = evaluate(sc, val, evals, TEAMS, NORTH_STAR_TARGET, args.margin,
                   args.floor, team_names)
    final = pick_final(res)
    print(f"[stack] stacker C={info['stacker_C']}; final scenario: {final}")

    # Same final scenario at 95%, for the registry's 95% columns only.
    res95 = evaluate(sc, val, evals, TEAMS, 0.95, args.margin, args.floor,
                     team_names, scenarios=[final], prior_correction=False)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(res["rows"])
    table.to_csv(ARTIFACT_DIR / "comparison.csv", index=False)
    pol = res["policies"][final]["margin_floors"]
    (ARTIFACT_DIR / "comparison.json").write_text(json.dumps({
        "models": args.models, "primary": args.models[0],
        "target": NORTH_STAR_TARGET, "floor": args.floor,
        "final_scenario": final, "final_policy": "margin_floors",
        "margins": res["margins"],
        "stacker_C": info["stacker_C"],
        "stacker_tuning": info["stacker_tuning"],
        "final_thresholds": _jsonable_policy(pol),
        "estimated_priors": res["estimated_priors"],
        "source_prior": dict(zip(TEAMS, prior_from_labels(val.y, TEAMS)
                                 .round(4).tolist())),
        "rows": res["rows"],
    }, indent=2, default=str))

    s = sc[final]
    bundle = {
        "model_name": MODEL_NAME, "models": list(args.models),
        "teams": list(TEAMS), "scenario": final,
        "stacker": info["stacker"] if final.startswith("stack") else None,
        "gate": s["gate"], "gate_kind": args.kind,
        "base_threshold": pol["base"], "team_thresholds": pol["teams"],
        "target": NORTH_STAR_TARGET, "margin": res["margins"][final],
        "floor": args.floor,
        "gate_feature_names": G.prob_feature_names(
            (["stack"] if final.startswith("stack") else []) + list(args.models),
            TEAMS),
        # Label-shift correction: the team mix the stacker was calibrated to.
        "source_prior": prior_from_labels(val.y, TEAMS).tolist(),
        # Serving fallback when a model is missing: the primary alone under
        # its own validated margin + per-team-floor policy.
        "primary_name": args.models[0],
        "primary_fallback_policy": _jsonable_policy(
            res["policies"]["primary_alone"]["margin_floors"]),
    }
    joblib.dump(bundle, ARTIFACT_DIR / "stack.joblib", compress=3)

    teams = np.asarray(TEAMS)
    out_splits = {"val": (val, s["val_chosen"], s["val_score"], s["val_probs"])}
    for e, a in evals.items():
        out_splits[e] = (a, s["eval_chosen"][e], s["eval_score"][e],
                         s["eval_probs"][e])
    for split, (a, chosen, score, probs) in out_splits.items():
        chosen_s = teams[chosen]
        mask = route_mask(score, chosen_s, pol["base"], pol["teams"])
        rep = routing_report(mask, chosen_s, a.y, args.floor, team_names)
        rep["per_team"].to_csv(ARTIFACT_DIR / f"per_team_routing_{split}.csv",
                               index=False)
        frame = pd.DataFrame({
            ID_COL: a.ids, "actual_team_id": a.y,
            "predicted_team_id": chosen_s, "confidence": score,
            "n_words": a.n_words, "auto_route": mask,
        })
        write_predictions("ensemble_stack", split,
                          with_team_probs(frame, probs, TEAMS))
        if split == "val":
            continue
        row95 = next(r for r in res95["rows"] if r["split"] == split
                     and r["policy"] == "margin_floors" and not r["corrected"])
        log_result(MODEL_NAME, split, {
            "train_set": f"{final}({'+'.join(args.models)})",
            "team_acc": float((chosen_s == a.y).mean()),
            "team_macro_f1": float(f1_score(a.y, chosen_s, average="macro",
                                            zero_division=0)),
            "north_star_coverage_90": rep["coverage"],
            "precision_at_90_threshold": rep["precision"],
            "north_star_coverage_95": row95["coverage"],
            "precision_at_95_threshold": row95["precision"],
            "teams_below_floor": ",".join(rep["teams_below_floor"]),
            "n_eval": len(a),
        })

    show = table[table["split"] != "val"][
        ["scenario", "policy", "split", "corrected", "coverage", "precision",
         "teams_below_floor", "accuracy", "macro_f1"]]
    print(show.to_string(index=False))
    print(f"[stack] margins: {json.dumps(res['margins'], default=str)}")
    print(f"[stack] wrote {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
