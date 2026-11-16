"""Routing thresholds: one global threshold, an optional safety margin, and
per-team floors.

Everything here is fitted on validation and frozen for test. The functions
take a routing score per complaint (higher = more likely the chosen team is
right), whether the chosen team was right, and the chosen/true teams.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import MIN_ROUTED, TEAM_PRECISION_FLOOR

GRID = np.round(np.linspace(0.0, 1.0, 1001), 3)


def wilson_lower(k, n, z: float = 1.645) -> np.ndarray:
    """One-sided 95% lower bound on a precision of k correct out of n."""
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1.0 + z * z / n
        centre = p + z * z / (2 * n)
        spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        low = (centre - spread) / denom
    return np.where(n > 0, low, 0.0)


def routed_counts(scores, correct, grid=GRID) -> tuple[np.ndarray, np.ndarray]:
    """For each threshold t in grid: how many rows have score >= t, and how
    many of those are correct."""
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    cum_correct = np.concatenate([[0], np.cumsum(correct[order])])
    n = len(scores) - np.searchsorted(np.sort(scores), grid, side="left")
    return n, cum_correct[n]


def choose_threshold(scores, correct, target: float, margin: float = 0.0,
                     min_routed: int = MIN_ROUTED, wilson: bool = False,
                     grid=GRID) -> float | None:
    """Lowest threshold whose routed precision reaches target + margin.
    With `wilson=True` the 95% lower bound has to reach it, not the point
    estimate. None if no threshold gets there."""
    n, k = routed_counts(scores, correct, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = wilson_lower(k, n) if wilson else np.where(n > 0, k / n, 0.0)
    ok = (n >= min_routed) & (stat >= target + margin)
    idx = np.flatnonzero(ok)
    return None if idx.size == 0 else float(grid[idx[0]])


def team_floor_thresholds(scores, chosen, true, base: float,
                          floor: float = TEAM_PRECISION_FLOOR,
                          margin: float = 0.0, min_routed: int = 30,
                          wilson: bool = True, grid=GRID,
                          teams=None) -> dict[str, dict]:
    """Per predicted team, the lowest threshold >= `base` at which complaints
    routed to that team are at least `floor` (+ margin) precise.

    Thresholds are only ever raised above `base`, never lowered. Status:
      ok          the global threshold already meets the floor
      raised      a higher threshold meets it
      never       no threshold meets it: this team is never auto-routed
      unverified  fewer than `min_routed` rows reach `base`; left at base
    """
    scores = np.asarray(scores, dtype=np.float64)
    chosen = np.asarray(chosen).astype(str)
    true = np.asarray(true).astype(str)
    g = grid[grid >= base]
    teams = sorted(set(chosen)) if teams is None else list(teams)
    out: dict[str, dict] = {}
    for team in teams:
        m = chosen == team
        n, k = routed_counts(scores[m], true[m] == team, g)
        if n.size == 0 or n[0] < min_routed:
            out[team] = {"threshold": base, "status": "unverified",
                         "n_val_routed": int(n[0]) if n.size else 0}
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = wilson_lower(k, n) if wilson else k / n
        idx = np.flatnonzero((n >= min_routed) & (stat >= floor + margin))
        if idx.size == 0:
            out[team] = {"threshold": None, "status": "never",
                         "n_val_routed": int(n[0])}
        else:
            out[team] = {"threshold": float(g[idx[0]]),
                         "status": "ok" if idx[0] == 0 else "raised",
                         "n_val_routed": int(n[idx[0]])}
    return out


def route_mask(scores, chosen, base: float | None,
               team_thresholds: dict[str, dict] | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if base is None:
        return np.zeros(len(scores), dtype=bool)
    chosen = np.asarray(chosen).astype(str)
    thr = np.full(len(scores), base, dtype=np.float64)
    for team, info in (team_thresholds or {}).items():
        t = info["threshold"]
        thr[chosen == team] = np.inf if t is None else t
    return scores >= thr


def routing_report(mask, chosen, true, floor: float = TEAM_PRECISION_FLOOR,
                   names: dict[str, str] | None = None) -> dict:
    mask = np.asarray(mask, dtype=bool)
    chosen = np.asarray(chosen).astype(str)
    true = np.asarray(true).astype(str)
    names = names or {}
    correct = chosen == true
    rows = []
    for team in sorted(set(true) | set(chosen)):
        to_team = mask & (chosen == team)
        own = true == team
        n_to = int(to_team.sum())
        prec = float(correct[to_team].mean()) if n_to else np.nan
        rows.append({
            "team_id": team, "team_name": names.get(team, ""),
            "n_eval": int(own.sum()), "n_routed_to": n_to,
            "precision_routed": round(prec, 4),
            "auto_routed_share": round(float(mask[own].mean()), 4)
            if own.any() else np.nan,
            "meets_floor": bool(prec >= floor) if n_to else None,
        })
    per_team = pd.DataFrame(rows)
    n_routed = int(mask.sum())
    return {
        "coverage": round(float(mask.mean()), 4),
        "precision": round(float(correct[mask].mean()), 4) if n_routed else None,
        "n_routed": n_routed,
        "teams_below_floor": per_team.loc[per_team["meets_floor"] == False,  # noqa: E712
                                          "team_id"].tolist(),
        "per_team": per_team,
    }


def temporal_margin(scores, correct, dates, target: float,
                    min_routed: int = MIN_ROUTED, late_frac: float = 1 / 3,
                    cap: float = 0.02) -> dict:
    """How much precision a threshold loses when it is used on later data.

    Splits validation by date: fits the threshold on the earlier part, applies
    it to the latest `late_frac`, and returns the shortfall below target as the
    margin (0 if none, capped at `cap`). Uses validation only.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    d = pd.to_datetime(pd.Series(dates), errors="coerce").to_numpy()
    valid = ~pd.isna(d)
    if valid.sum() < 4 * min_routed:
        return {"margin": 0.0, "reason": "not enough dated rows"}
    cut = np.sort(d[valid])[int(valid.sum() * (1 - late_frac))]
    early = valid & (d < cut)
    late = valid & (d >= cut)
    thr = choose_threshold(scores[early], correct[early], target,
                           min_routed=min_routed)
    if thr is None or late.sum() < min_routed:
        return {"margin": 0.0, "reason": "threshold not reachable on early part"}
    routed_late = late & (scores >= thr)
    if routed_late.sum() < min_routed:
        return {"margin": 0.0, "reason": "too few late rows routed"}
    late_prec = float(correct[routed_late].mean())
    return {
        "margin": round(float(np.clip(target - late_prec, 0.0, cap)), 4),
        "threshold_early": thr,
        "late_precision": round(late_prec, 4),
        "cut_date": str(pd.Timestamp(cut).date()),
    }
