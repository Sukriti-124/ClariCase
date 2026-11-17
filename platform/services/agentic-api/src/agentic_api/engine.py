"""Encoder + example pool + LLM chooser, loaded once per process."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.agentic.examples import ExamplePool
from src.agentic.llm import Choice, Chooser, PromptCache, get_backend
from src.agentic.prompts import issue_prompt, team_prompt
from src.ensemble.probs import ISSUE_TEAM, ISSUES, TEAMS


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


def encoder_for(raw: dict, device: str | None = None):
    from src.improved_model_2.predict import encoder_for as _enc
    return _enc(raw, device)


@dataclass
class Engine:
    pool: ExamplePool
    encoder: Any
    chooser: Chooser
    version: str

    @property
    def model_name(self) -> str:
        return f"agentic:{self.chooser.backend.model}"

    def choose(self, text: str, task: str, candidates: list[str],
               team: str | None) -> Choice:
        if not text or not text.strip():
            raise ValueError("text is empty")
        if task == "team":
            bad = [c for c in candidates if c not in TEAMS]
            if bad:
                raise KeyError(f"unknown team candidates {bad}")
        else:
            if team is None:
                raise KeyError("task 'issue' needs the team")
            iix = {i: j for j, i in enumerate(ISSUES)}
            bad = [c for c in candidates
                   if c not in iix or ISSUE_TEAM[iix[c]] != team]
            if bad:
                raise KeyError(f"issue candidates not in team {team}: {bad}")
        if len(candidates) == 1:
            return Choice(candidate=candidates[0], parsed=True, raw="",
                          cached=True, latency_ms=0)
        q = self.encoder.encode([text])[0]
        if task == "team":
            prompt = team_prompt(text, candidates, self.pool.for_teams(q, candidates))
        else:
            prompt = issue_prompt(text, team, candidates,
                                  self.pool.for_issues(q, candidates))
        return self.chooser.choose(prompt, candidates)


def load_engine(pool_path: Path, embed_model_path: Path,
                llm_cache_path: Path | None, device: str | None = None,
                backend=None) -> Engine:
    pool = ExamplePool.load(pool_path)
    raw = joblib.load(embed_model_path)
    chooser = Chooser(backend if backend is not None else get_backend(),
                      PromptCache(llm_cache_path))
    return Engine(pool=pool, encoder=encoder_for(raw, device), chooser=chooser,
                  version=_fingerprint(pool_path.with_suffix(".parquet")))
