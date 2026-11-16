"""Run the LLM candidate-choice stage on a split and save predictions.

Usage (dry run first, then the real thing):
    AGENTIC_BACKEND=mock python -m src.agentic.run --split val --rows 200
    python -m src.agentic.run --split val --rows 3000
    python -m src.agentic.run --split test --rows 3000
    python -m src.agentic.run --split val --rows 0      # every eligible row

--rows samples that many rows from EACH task's pool (seeded), so cost is
bounded while both tasks are measured. Outputs under docs/metrics/agentic/:
    predictions_<split>.parquet   per asked row: candidates, LLM pick, truth
    traces_<split>.jsonl          prompts' raw answers and parse status
    run_<split>.json              counts, cache hits, timing
Then:  python -m src.agentic.evaluate --fit val --score test
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.agentic import config as C
from src.agentic.examples import ExamplePool
from src.agentic.llm import Chooser, PromptCache, get_backend
from src.agentic.prompts import issue_prompt, team_prompt
from src.agentic.tasks import (issue_candidates, issue_task_rows,
                               load_split_data, team_candidates,
                               team_task_rows)
from src.ensemble.export_predictions import load_frame
from src.utils.config import ID_COL, RANDOM_SEED, TEXT_COL


def _sample(rows: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n and len(rows) > n:
        rows = np.random.default_rng(seed).choice(rows, n, replace=False)
    return np.sort(rows)


def run_split(split: str, rows: int, pool: ExamplePool, chooser: Chooser,
              texts: dict[str, str], Xq: np.ndarray, d, tasks=("team", "issue"),
              rerank: bool = False, seed: int = RANDOM_SEED,
              progress: bool = True) -> tuple[pd.DataFrame, list[dict], dict]:
    """Core loop, split from main() so tests can drive it with a mock."""
    traces: list[dict] = []
    info: dict = {"split": split, "rows_requested": rows, "tasks": list(tasks)}
    t0 = time.time()

    final_team = d.stack_team.copy()
    team_pick = np.full(len(d), "", dtype=object)
    team_rows = np.array([], dtype=int)
    team_cands = team_candidates(d.P)
    if "team" in tasks:
        team_rows = _sample(team_task_rows(d), rows, seed)
        prompts, cands = [], []
        for i in team_rows:
            c = [t for t in team_cands[i] if t]
            ex = pool.for_teams(Xq[i], c, query_text=texts[d.ids[i]] if rerank else None)
            prompts.append(team_prompt(texts[d.ids[i]], c, ex))
            cands.append(c)
        choices = chooser.choose_many(prompts, cands, progress=progress)
        for i, ch, c in zip(team_rows, choices, cands):
            team_pick[i] = ch.candidate
            final_team[i] = ch.candidate
            traces.append({"task": "team", ID_COL: d.ids[i], "candidates": c,
                           "pick": ch.candidate, "parsed": ch.parsed,
                           "cached": ch.cached, "raw": ch.raw,
                           "truth": d.y_team[i]})
        info["team_rows"] = int(len(team_rows))
        info["team_pool"] = int((~d.auto_route).sum())

    issue_names, gap = issue_candidates(d.Q, final_team)
    issue_pick = np.full(len(d), "", dtype=object)
    issue_rows = np.array([], dtype=int)
    if "issue" in tasks:
        issue_rows = _sample(issue_task_rows(gap, issue_names), rows, seed + 1)
        prompts, cands = [], []
        for i in issue_rows:
            c = [x for x in issue_names[i] if x]
            ex = pool.for_issues(Xq[i], c, query_text=texts[d.ids[i]] if rerank else None)
            prompts.append(issue_prompt(texts[d.ids[i]], final_team[i], c, ex))
            cands.append(c)
        choices = chooser.choose_many(prompts, cands, progress=progress)
        for i, ch, c in zip(issue_rows, choices, cands):
            issue_pick[i] = ch.candidate
            traces.append({"task": "issue", ID_COL: d.ids[i], "team": final_team[i],
                           "candidates": c, "pick": ch.candidate,
                           "parsed": ch.parsed, "cached": ch.cached,
                           "raw": ch.raw, "truth": d.y_issue[i]})
        info["issue_rows"] = int(len(issue_rows))
        info["issue_pool"] = int(len(issue_task_rows(gap, issue_names)))

    asked = np.union1d(team_rows, issue_rows)
    out = pd.DataFrame({
        ID_COL: d.ids[asked],
        "actual_team_id": d.y_team[asked],
        "actual_issue_id": d.y_issue[asked],
        "stack_team": d.stack_team[asked],
        "gate_score": d.gate_score[asked],
        "auto_route": d.auto_route[asked],
        "team_asked": np.isin(asked, team_rows),
        "team_cand1": team_cands[asked, 0],
        "team_cand2": team_cands[asked, 1] if team_cands.shape[1] > 1 else "",
        "team_cand3": team_cands[asked, 2] if team_cands.shape[1] > 2 else "",
        "llm_team": team_pick[asked],
        "final_team": final_team[asked],
        "issue_asked": np.isin(asked, issue_rows),
        "issue_cand1": issue_names[asked, 0],
        "issue_cand2": issue_names[asked, 1],
        "issue_cand3": issue_names[asked, 2],
        "issue_gap": gap[asked],
        "llm_issue": issue_pick[asked],
    })
    info.update({
        "n_asked_rows": int(len(asked)), "split_rows": int(len(d)),
        "cache_hits": int(sum(t["cached"] for t in traces)),
        "unparsed": int(sum(not t["parsed"] for t in traces)),
        "wall_clock_sec": round(time.time() - t0, 1),
        "model": chooser.backend.model, "rerank": rerank,
        "issue_gap_threshold": C.ISSUE_GAP_THRESHOLD,
    })
    return out, traces, info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--split", default="val", choices=["val", "test", "holdout_2026"])
    p.add_argument("--rows", type=int, default=3000,
                   help="rows per task (0 = all eligible)")
    p.add_argument("--tasks", nargs="+", default=["team", "issue"],
                   choices=["team", "issue"])
    p.add_argument("--rerank", action="store_true",
                   help="cross-encoder reranking of retrieved examples")
    p.add_argument("--device", default=None)
    p.add_argument("--rebuild-pool", action="store_true")
    args = p.parse_args()

    from src.agentic.examples import build_pool_from_cache
    from src.ensemble.embeddings import load_embed_bundle, split_embeddings

    stem = C.EXAMPLE_POOL
    if args.rebuild_pool or not stem.with_suffix(".parquet").exists():
        pool = build_pool_from_cache(device=args.device)
    else:
        pool = ExamplePool.load(stem)
    if args.rerank:
        pool.enable_reranker()

    d = load_split_data(args.split)
    frame = load_frame(args.split)
    frame = frame[frame[ID_COL].isin(set(d.ids))].reset_index(drop=True)
    # Same row order as d, so the cached embeddings line up with the ids.
    frame = frame.set_index(ID_COL).loc[d.ids].reset_index()
    texts = dict(zip(frame[ID_COL], frame[TEXT_COL]))
    Xq = split_embeddings(args.split, frame, load_embed_bundle(), args.device)

    chooser = Chooser(get_backend(), PromptCache(C.LLM_CACHE))
    print(f"[agentic] {args.split}: {len(d):,} rows, backend={chooser.backend.model}, "
          f"cache={len(chooser.cache):,} answers", flush=True)
    out, traces, info = run_split(args.split, args.rows, pool, chooser, texts,
                                  Xq, d, tuple(args.tasks), args.rerank)

    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(C.ARTIFACT_DIR / f"predictions_{args.split}.parquet", index=False)
    with (C.ARTIFACT_DIR / f"traces_{args.split}.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, default=str) + "\n")
    (C.ARTIFACT_DIR / f"run_{args.split}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    if "team" in args.tasks and info.get("team_rows"):
        m = out[out.team_asked]
        print(f"  team  : stacker {(m.stack_team == m.actual_team_id).mean():.3f} "
              f"-> LLM {(m.llm_team == m.actual_team_id).mean():.3f} on {len(m):,} rows")
    if "issue" in args.tasks and info.get("issue_rows"):
        m = out[out.issue_asked]
        print(f"  issue : ML top-1 {(m.issue_cand1 == m.actual_issue_id).mean():.3f} "
              f"-> LLM {(m.llm_issue == m.actual_issue_id).mean():.3f} on {len(m):,} rows")
    print(f"[agentic] wrote {C.ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
