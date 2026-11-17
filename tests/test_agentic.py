"""Agentic candidate-choice stage, with the mock backend and synthetic data."""
import json

import numpy as np
import pandas as pd
import pytest

from src.agentic import llm as L
from src.agentic.examples import ExamplePool, clip
from src.agentic.prompts import issue_prompt, team_card, team_prompt
from src.agentic.run import run_split
from src.agentic.tasks import (SplitData, issue_candidates, issue_task_rows,
                               team_candidates, team_task_rows)
from src.ensemble.probs import ISSUE_TEAM, ISSUES, TEAMS, block_normalize
from src.utils.config import ID_COL, ISSUE_COL, TEAM_COL, TEXT_COL

T = np.asarray(TEAMS)
I = np.asarray(ISSUES)


# ---- llm ----------------------------------------------------------------------

def test_parse_choice_takes_first_candidate_mentioned_and_falls_back():
    assert L.parse_choice("T02", ["T01", "T02"]) == ("T02", True)
    assert L.parse_choice("The answer is T02, not T01.", ["T01", "T02"]) == ("T02", True)
    assert L.parse_choice("T02_I001", ["T02_I001", "T02_I004"]) == ("T02_I001", True)
    assert L.parse_choice("T02_I0010", ["T02_I001"]) == ("T02_I001", False)
    assert L.parse_choice("", ["T05", "T04"]) == ("T05", False)


def test_prompt_cache_roundtrip_and_chooser_hits(tmp_path):
    cache = L.PromptCache(tmp_path / "c.sqlite")
    assert cache.get("m", "p") is None
    cache.put("m", "p", "T02")
    assert cache.get("m", "p") == "T02" and len(cache) == 1

    class Counting:
        model = "m"
        calls = 0

        def complete(self, prompt):
            self.calls += 1
            return "T02"

    b = Counting()
    ch = L.Chooser(b, cache)
    first = ch.choose("p", ["T01", "T02"])
    assert first.candidate == "T02" and first.cached and b.calls == 0
    second = ch.choose("q", ["T01", "T02"])
    assert not second.cached and b.calls == 1
    out = ch.choose_many(["p", "q", "r"], [["T01", "T02"]] * 3, concurrency=2)
    assert [o.candidate for o in out] == ["T02"] * 3 and b.calls == 2


def test_mock_backend_prefers_label_with_overlapping_words():
    prompt = ("Candidate teams:\n- T01: credit report bureau\n"
              "- T06: mortgage escrow foreclosure\n\n"
              "Complaint: my mortgage servicer mishandled the escrow\n")
    assert L.MockBackend().complete(prompt) == "T06"


def test_openai_backend_requires_key(monkeypatch):
    with pytest.raises(RuntimeError, match="API key"):
        L.OpenAICompatBackend(api_key="")


# ---- examples and prompts ---------------------------------------------------------

def _pool(n_per_issue=4, seed=0):
    rng = np.random.default_rng(seed)
    rows, vecs = [], []
    for j, iss in enumerate(ISSUES):
        for k in range(n_per_issue):
            rows.append({ID_COL: f"{j}-{k}", TEAM_COL: ISSUE_TEAM[j],
                         ISSUE_COL: iss, TEXT_COL: f"example {iss} number {k} " * 3})
            v = np.zeros(len(ISSUES) + 2)
            v[j] = 1.0
            v[-2:] = rng.random(2) * 0.1
            vecs.append(v / np.linalg.norm(v))
    return ExamplePool(pd.DataFrame(rows), np.array(vecs, dtype=np.float32))


def test_pool_build_caps_per_issue_and_retrieval_respects_candidate(tmp_path):
    train = pd.DataFrame({
        ID_COL: [str(i) for i in range(30)],
        TEAM_COL: ["T01"] * 20 + ["T02"] * 10,
        ISSUE_COL: ["T01_I005"] * 20 + ["T02_I001"] * 10,
        TEXT_COL: ["word " * 200] * 30,
    })
    X = np.random.default_rng(1).normal(size=(30, 8)).astype(np.float32)
    pool = ExamplePool.build(train, X, cap_per_issue=5)
    assert len(pool) == 10
    assert pool.text[0].endswith("…") and len(pool.text[0]) <= 350
    pool.save(tmp_path / "pool")
    back = ExamplePool.load(tmp_path / "pool")
    assert len(back) == 10 and back.X.shape == (10, 8)

    p = _pool()
    q = p.X[ISSUES.index("T02_I001") * 4]
    ex = p.for_teams(q, ["T01", "T02"], k=2)
    assert all(e.team == "T01" for e in ex["T01"]) and len(ex["T01"]) == 2
    assert ex["T02"][0].issue == "T02_I001"
    ex_i = p.for_issues(q, ["T02_I001", "T02_I004", "T99_I999"], k=2)
    assert all(e.issue == "T02_I004" for e in ex_i["T02_I004"])
    assert ex_i["T99_I999"] == []


def test_prompts_list_candidates_with_cards_and_examples():
    p = _pool()
    q = p.X[0]
    tp = team_prompt("My bank XXXX closed my checking account", ["T04", "T05"],
                     p.for_teams(q, ["T04", "T05"]))
    assert "- T04:" in tp and "- T05:" in tp and "Not for:" in tp
    assert tp.rstrip().endswith("(T04, T05).")
    assert "Similar complaints handled by this team" in tp
    ip = issue_prompt("text " * 400, "T02", ["T02_I001", "T02_I004"],
                      p.for_issues(q, ["T02_I001", "T02_I004"]))
    assert "- T02_I001:" in ip and "Debt Collection Team" in ip
    assert "…" in ip  # complaint clipped
    assert "T11" in team_card("T11") or "Credit repair" in team_card("T11")
    assert clip("a  b\nc XXXXXX", 100) == "a b c XXXX"


# ---- tasks ----------------------------------------------------------------------

def _split(n=400, seed=3):
    rng = np.random.default_rng(seed)
    yi = rng.integers(0, len(ISSUES), n)
    y_issue, y_team = I[yi], ISSUE_TEAM[yi]
    P = rng.dirichlet(np.ones(len(TEAMS)) * 0.3, n)
    # Stacker right 70% of the time; gate score higher when right.
    right = rng.random(n) < 0.7
    for i in range(n):
        if right[i]:
            P[i, list(TEAMS).index(y_team[i])] += 2.0
    P /= P.sum(1, keepdims=True)
    stack = T[P.argmax(1)]
    gate = np.clip(np.where(stack == y_team, 0.75, 0.45) + rng.normal(0, 0.15, n), 0, 1)
    Q = rng.random((n, len(ISSUES)))
    Q[np.arange(n), yi] += rng.choice([0.0, 2.0], n)
    Q = block_normalize(Q, ISSUE_TEAM)
    return SplitData(ids=np.array([f"c{i}" for i in range(n)]), y_team=y_team,
                     y_issue=y_issue, stack_team=stack, gate_score=gate,
                     auto_route=gate >= 0.7, P=P, Q=Q)


def test_candidates_put_ml_choice_first_and_stay_in_team():
    d = _split()
    tc = team_candidates(d.P)
    assert (tc[:, 0] == d.stack_team).all() and tc.shape == (len(d), 3)
    names, gap = issue_candidates(d.Q, d.stack_team)
    for r in range(len(d)):
        valid = [x for x in names[r] if x]
        assert all(ISSUE_TEAM[list(ISSUES).index(x)] == d.stack_team[r] for x in valid)
        assert valid[0] == I[np.where(ISSUE_TEAM == d.stack_team[r], d.Q[r], -1).argmax()]
    assert ((gap >= 0) & (gap <= 1)).all()
    rows = issue_task_rows(gap, names, threshold=0.3)
    assert (gap[rows] < 0.3).all()
    assert set(team_task_rows(d)) == set(np.flatnonzero(~d.auto_route))


def test_run_split_end_to_end_with_mock(tmp_path):
    d = _split()
    pool = _pool()
    # Query embeddings: the true issue's direction, so retrieval finds it.
    Xq = np.zeros((len(d), pool.X.shape[1]), dtype=np.float32)
    Xq[np.arange(len(d)), [list(ISSUES).index(x) for x in d.y_issue]] = 1.0
    # The mock backend scores candidates by word overlap with the complaint,
    # so a complaint written from the true team's and issue's cards is one
    # the mock can get right more often than the synthetic stacker.
    from src.agentic.prompts import issue_card
    texts = {cid: (team_card(d.y_team[i]).split(" Not for:")[0] + " "
                   + issue_card(d.y_issue[i]))
             for i, cid in enumerate(d.ids)}
    chooser = L.Chooser(L.MockBackend(), L.PromptCache(tmp_path / "c.sqlite"))
    out, traces, info = run_split("val", 50, pool, chooser, texts, Xq, d,
                                  progress=False)
    assert info["team_rows"] == 50 and info["issue_rows"] <= 50
    assert out.team_asked.sum() == 50
    assert set(out.loc[out.team_asked, "llm_team"]).issubset(set(TEAMS))
    m = out[out.team_asked]
    # The mock sees the truth in the complaint text, so it does better than
    # the stacker on asked rows.
    assert (m.llm_team == m.actual_team_id).mean() > (m.stack_team == m.actual_team_id).mean()
    assert (m.final_team == m.llm_team).all()
    mi = out[out.issue_asked]
    assert mi.llm_issue.isin(set(ISSUES)).all()
    assert all(t["parsed"] for t in traces)
    # Second run is served from the cache.
    out2, traces2, info2 = run_split("val", 50, pool, chooser, texts, Xq, d,
                                     progress=False)
    assert info2["cache_hits"] == len(traces2) and out2.equals(out)


def test_evaluate_fits_rescue_on_val_only(tmp_path, monkeypatch):
    from src.agentic import evaluate as E
    rng = np.random.default_rng(7)
    n = 600
    agree = rng.random(n) < 0.7
    gate = rng.random(n)
    truth = np.where(rng.random(n) < np.where(agree, 0.6 + 0.4 * gate, 0.4), "T01", "T02")
    val = pd.DataFrame({
        "team_asked": True, "stack_team": "T01",
        "llm_team": np.where(agree, "T01", "T02"), "actual_team_id": truth,
        "gate_score": gate,
    })
    rule = E.fit_rescue(val, 0.9, 0.0)
    assert rule["threshold"] is not None and 0 < rule["threshold"] < 1
    assert rule["val_rescued_precision"] >= 0.9
    assert rule["agree_share"] == pytest.approx(agree.mean(), abs=1e-4)
    assert E.fit_rescue(val.head(10), 0.9)["threshold"] is None
