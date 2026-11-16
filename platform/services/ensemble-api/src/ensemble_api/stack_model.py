"""The served ensemble: stacker + gate + thresholds + label-shift correction.

Loads the bundle written by `python -m src.ensemble.train_stack` and makes one
routing decision per request from the probability vectors the model services
return. The feature code is imported from src.ensemble so the service builds
exactly the features the gate was trained on.

Decision, given every model's probabilities:
  1. stacker  -> P(team) (or the primary's own vector, if the bundle says so)
  2. label-shift correction of P using the rolling prior estimate
  3. chosen team = argmax of the corrected P
  4. gate(chosen, [P, each model's probs], n_words) -> P(chosen is right)
  5. auto-route iff score >= the chosen team's threshold
  6. scenario: auto / assisted (models agree, below threshold) / contested

If any model the bundle needs is missing, `fallback` applies the validated
primary-alone policy instead (its own margin threshold and floors) and the
scenario is `degraded`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.ensemble import gate as G
from src.ensemble.label_shift import RollingPrior
from src.ensemble.stack import stack_features, stacker_proba
from src.ensemble.thresholds import route_mask


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


@dataclass
class Decision:
    team: str
    score: float
    auto_route: bool
    scenario: str
    threshold: float | None
    stack_probs: np.ndarray | None
    model_teams: dict[str, str]
    prior_active: bool
    candidates: list[str] | None = None   # top teams, chosen first


class StackBundle:
    def __init__(self, raw: dict, version: str, prior: RollingPrior):
        self.raw = raw
        self.version = version
        self.prior = prior
        self.teams: list[str] = list(raw["teams"])
        self.models: list[str] = list(raw["models"])
        self.primary: str = raw.get("primary_name", self.models[0])
        self.scenario_kind: str = raw["scenario"]
        self.uses_stacker = self.scenario_kind.startswith("stack")
        self.model_name: str = raw["model_name"]
        self.base_threshold = raw["base_threshold"]
        self.team_thresholds: dict = raw.get("team_thresholds") or {}
        fb = raw.get("primary_fallback_policy") or {
            "base": raw.get("primary_fallback_threshold"), "teams": None}
        self.fallback_policy = fb
        if self.uses_stacker and raw.get("stacker") is None:
            raise RuntimeError("bundle scenario needs a stacker but has none")

    # ---- helpers ----
    def _vec(self, probs: dict[str, float]) -> np.ndarray:
        return np.array([float(probs.get(t, 0.0)) for t in self.teams],
                        dtype=np.float64)

    def _routed(self, score: float, team: str, policy: dict) -> bool:
        return bool(route_mask(np.array([score]), np.array([team]),
                               policy["base"], policy.get("teams"))[0])

    def threshold_for(self, team: str) -> float | None:
        info = self.team_thresholds.get(team)
        if info is None:
            return self.base_threshold
        return info["threshold"]

    # ---- the decision ----
    def decide(self, model_probs: dict[str, dict[str, float]],
               n_words: int) -> Decision:
        missing = [m for m in self.models if m not in model_probs]
        if missing:
            raise KeyError(f"missing model probabilities: {missing}")
        mats = [self._vec(model_probs[m])[None, :] for m in self.models]
        model_teams = {m: self.teams[int(P.argmax())]
                       for m, P in zip(self.models, mats)}

        if self.uses_stacker:
            P_raw = stacker_proba(self.raw["stacker"], stack_features(mats),
                                  len(self.teams))[0]
            self.prior.observe(P_raw)
            P = self.prior.adjust(P_raw)
            gate_mats = [P[None, :]] + mats
            stack_probs = P
        else:
            P = mats[0][0]
            gate_mats = mats
            stack_probs = None
        j = int(P.argmax())
        team = self.teams[j]
        X = G.prob_features([j], gate_mats, [n_words], len(self.teams))
        score = float(self.raw["gate"].predict_proba(X)[0, 1])
        auto = self._routed(score, team, {"base": self.base_threshold,
                                          "teams": self.team_thresholds})
        if auto:
            scenario = "auto"
        elif all(t == team for t in model_teams.values()):
            scenario = "assisted"
        else:
            scenario = "contested"
        order = np.argsort(-P, kind="stable")[:3]
        return Decision(team=team, score=score, auto_route=auto,
                        scenario=scenario, threshold=self.threshold_for(team),
                        stack_probs=stack_probs, model_teams=model_teams,
                        prior_active=self.prior.active,
                        candidates=[self.teams[int(k)] for k in order])

    def fallback(self, primary_probs: dict[str, float]) -> Decision:
        P = self._vec(primary_probs)
        j = int(P.argmax())
        team, score = self.teams[j], float(P[j])
        auto = (self.fallback_policy["base"] is not None
                and self._routed(score, team, self.fallback_policy))
        return Decision(team=team, score=score, auto_route=auto,
                        scenario="degraded",
                        threshold=self.fallback_policy["base"],
                        stack_probs=None, model_teams={self.primary: team},
                        prior_active=False)


def load_stack(path: Path, window: int = 5000, min_rows: int = 500,
               refresh: int = 100, label_shift: bool = True) -> StackBundle:
    raw = joblib.load(path)
    source = np.asarray(raw["source_prior"], dtype=np.float64)
    prior = RollingPrior(source, window=window, min_rows=min_rows,
                         refresh=refresh)
    if not label_shift:
        prior.min_rows = 10 ** 12  # never activates
    return StackBundle(raw, _fingerprint(path), prior)
