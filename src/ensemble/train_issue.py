"""Issue classification on top of the chosen team.

Combines the issue models by weighted log-linear pooling, then ranks the
issues inside the chosen team.

Components (issue files written by export_predictions):
  improved_model_1   SVM issue heads, one per team
  embed_issue_head   logistic regression per team on sentence embeddings
  knn_embed          issue vote among the 25 nearest training complaints

Pooled score of issue i, which belongs to team t:
    sum over models m of  w_m * log P_m(i | t)   +   beta * log prior(i | t)
The first component's weight is fixed at 1. The other weights and beta are
chosen on validation by issue macro-F1 *with the true team*, so team errors
do not drive the choice; prior(i | t) is the issue mix inside each team on
validation. Test is scored once.

The team comes from --team-source: ensemble_stack (written by train_stack,
the default) or improved_model_1 (today's team).

Usage:
    python -m src.ensemble.train_issue
    python -m src.ensemble.train_issue --team-source improved_model_1
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.ensemble.data import predictions_path, read_predictions
from src.ensemble.probs import (ISSUE_TEAM, ISSUES, block_normalize,
                                issue_cols, read_issue_probs)
from src.utils.config import ID_COL, METRICS_ROOT

ARTIFACT_DIR = METRICS_ROOT / "ensemble_issue"
MODEL_NAME = "ensemble_issue_v1"
DEFAULT_COMPONENTS = ("improved_model_1", "embed_issue_head", "knn_embed")
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0)
BETA_GRID = (-0.5, -0.25, 0.0, 0.25)
ISSUE_ARR = np.asarray(ISSUES)


@dataclass
class IssueData:
    ids: np.ndarray
    y_team: np.ndarray
    y_issue: np.ndarray
    logQ: dict[str, np.ndarray]
    teams: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.ids)


def load_issue_data(split: str, components, team_sources) -> IssueData:
    cols = issue_cols()
    first = read_issue_probs(components[0], split)
    base = first[[ID_COL, "actual_team_id", "actual_issue_id"]].copy()
    for c in components:
        df = read_issue_probs(c, split)[[ID_COL] + cols]
        df.columns = [ID_COL] + [f"{c}::{x}" for x in cols]
        base = base.merge(df, on=ID_COL, how="inner", validate="one_to_one")
    for src in team_sources:
        df = read_predictions(src, split)[[ID_COL, "predicted_team_id"]]
        df = df.rename(columns={"predicted_team_id": f"team::{src}"})
        base = base.merge(df, on=ID_COL, how="inner", validate="one_to_one")
    logQ = {c: np.log(block_normalize(
        base[[f"{c}::{x}" for x in cols]].to_numpy(np.float64), ISSUE_TEAM))
        for c in components}
    return IssueData(
        ids=base[ID_COL].to_numpy(),
        y_team=base["actual_team_id"].astype(str).to_numpy(),
        y_issue=base["actual_issue_id"].astype(str).to_numpy(),
        logQ=logQ,
        teams={s: base[f"team::{s}"].astype(str).to_numpy()
               for s in team_sources})


def prior_log(y_issue, issues=ISSUES, issue_team=ISSUE_TEAM) -> np.ndarray:
    counts = pd.Series(y_issue).value_counts()
    c = np.array([counts.get(i, 0) for i in issues], dtype=np.float64) + 1.0
    return np.log(block_normalize(c[None, :], issue_team, eps=0.0)[0])


def pooled(logQ: dict, weights: dict, prior: np.ndarray | None = None,
           beta: float = 0.0) -> np.ndarray:
    L = sum(w * logQ[c] for c, w in weights.items() if w)
    if not isinstance(L, np.ndarray):  # every weight zero
        L = np.zeros_like(next(iter(logQ.values())))
    if prior is not None and beta:
        L = L + beta * prior[None, :]
    return L


def topk_in_team(L: np.ndarray, chosen_team, k: int = 3,
                 issue_team: np.ndarray = ISSUE_TEAM) -> np.ndarray:
    """Indices of the k best issues inside each row's chosen team; -1 where
    the team has fewer than k issues (or is unknown)."""
    outside = issue_team[None, :] != np.asarray(chosen_team).astype(str)[:, None]
    L2 = np.where(outside, -np.inf, L)
    if k == 1:
        top = L2.argmax(axis=1)[:, None]
    else:
        top = np.argsort(-L2, axis=1, kind="stable")[:, :k]
    bad = ~np.isfinite(np.take_along_axis(L2, top, axis=1))
    return np.where(bad, -1, top)


def issue_metrics(top: np.ndarray, y_issue) -> dict:
    y = np.asarray(y_issue).astype(str)
    names = np.where(top >= 0, ISSUE_ARR[np.clip(top, 0, None)], "")
    pred = names[:, 0]
    return {
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro",
                                         zero_division=0)), 4),
        "top3_accuracy": round(float((names == y[:, None]).any(axis=1)
                                     .mean()), 4)
        if top.shape[1] >= 3 else None,
    }


def tune(data: IssueData, components, prior: np.ndarray,
         weight_grid=WEIGHT_GRID, beta_grid=BETA_GRID) -> tuple[dict, list]:
    """Weights and beta by validation macro-F1 with the true team."""
    head, others = components[0], list(components[1:])
    table, best = [], None
    for ws in itertools.product(weight_grid, repeat=len(others)):
        weights = {head: 1.0, **dict(zip(others, ws))}
        for beta in beta_grid:
            top = topk_in_team(pooled(data.logQ, weights, prior, beta),
                               data.y_team, k=1)
            m = issue_metrics(top, data.y_issue)
            table.append({"weights": weights, "beta": beta, **m})
            key = (m["macro_f1"], m["accuracy"])
            if best is None or key > best[0]:
                best = (key, {"weights": weights, "beta": beta})
    return best[1], table


def report_rows(data: IssueData, split: str, components, final: dict,
                prior: np.ndarray, team_source: str) -> list[dict]:
    rows = []

    def add(label, weights, beta, team_label, teams):
        top = topk_in_team(pooled(data.logQ, weights, prior, beta), teams)
        rows.append({"split": split, "issue_model": label, "team": team_label,
                     **issue_metrics(top, data.y_issue)})

    if "improved_model_1" in data.teams and "improved_model_1" in components:
        add("improved_model_1 (today)", {"improved_model_1": 1.0}, 0.0,
            "improved_model_1", data.teams["improved_model_1"])
    for c in components:
        add(c, {c: 1.0}, 0.0, team_source, data.teams[team_source])
        add(c, {c: 1.0}, 0.0, "true team", data.y_team)
    add("pooled", final["weights"], final["beta"], team_source,
        data.teams[team_source])
    if "improved_model_1" in data.teams and team_source != "improved_model_1":
        add("pooled", final["weights"], final["beta"], "improved_model_1",
            data.teams["improved_model_1"])
    add("pooled", final["weights"], final["beta"], "true team", data.y_team)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--components", nargs="+", default=list(DEFAULT_COMPONENTS))
    p.add_argument("--team-source", default="ensemble_stack")
    p.add_argument("--eval-splits", nargs="+", default=["test"])
    args = p.parse_args()

    from src.utils.registry import log_result

    sources = list(dict.fromkeys([args.team_source, "improved_model_1"]))
    data = {s: load_issue_data(s, args.components, sources)
            for s in ["val", *args.eval_splits]}
    for s, d in data.items():
        print(f"[issue] {s}: {len(d):,} rows")

    prior = prior_log(data["val"].y_issue)
    final, table = tune(data["val"], args.components, prior)
    print(f"[issue] chosen on validation: {final}")

    rows = []
    for s, d in data.items():
        rows += report_rows(d, s, args.components, final, prior,
                            args.team_source)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "comparison.csv", index=False)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps({
        "components": args.components, "team_source": args.team_source,
        "chosen": final, "tuning": table, "rows": rows}, indent=2))

    for s, d in data.items():
        teams = d.teams[args.team_source]
        top = topk_in_team(pooled(d.logQ, final["weights"], prior,
                                  final["beta"]), teams)
        names = np.where(top >= 0, ISSUE_ARR[np.clip(top, 0, None)], "")
        out = pd.DataFrame({
            ID_COL: d.ids, "actual_team_id": d.y_team,
            "actual_issue_id": d.y_issue, "predicted_team_id": teams,
            "issue_top1": names[:, 0], "issue_top2": names[:, 1],
            "issue_top3": names[:, 2]})
        path = predictions_path(MODEL_NAME, s)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        if s == "val":
            continue
        labels = sorted(set(d.y_issue) | set(names[:, 0]))
        (ARTIFACT_DIR / f"classification_report_issue_{s}.json").write_text(
            json.dumps(classification_report(
                d.y_issue, names[:, 0], labels=labels, output_dict=True,
                zero_division=0), indent=2))
        m = issue_metrics(top, d.y_issue)
        log_result(MODEL_NAME, s, {
            "train_set": f"pooled({'+'.join(args.components)}), "
                         f"team={args.team_source}",
            "team_acc": float((teams == d.y_team).mean()),
            "issue_macro_f1": m["macro_f1"], "issue_top3": m["top3_accuracy"],
            "joint_exact": m["accuracy"], "n_eval": len(d),
        })

    print(pd.DataFrame(rows)[lambda x: x["split"] != "val"]
          .to_string(index=False))
    print(f"[issue] wrote {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
