"""Tests for scripts/trace_prediction.py (synthetic data only)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "trace_prediction.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("trace_prediction", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trace_prediction"] = mod
    spec.loader.exec_module(mod)
    return mod


tp = _load_script()

from src.improved_model_1.model import HierarchicalSVM, _ConstantIssue  # noqa: E402
from src.utils.text import clean_text  # noqa: E402

SEEDS = range(100)

VOCAB = ["card", "charged", "twice", "loan", "payment", "late", "fee",
         "account", "closed", "credit", "report", "error", "mortgage",
         "escrow", "debt", "collector", "called", "fraud", "Bank", "XXXX",
         "$100", "refund", "denied", "interest"]

CORPUS = [
    ("my card was charged twice for one purchase", "cards"),
    ("unauthorized charge on my credit card refund denied", "cards"),
    ("card fee charged again and again", "cards"),
    ("fraud on card the bank refused refund", "cards"),
    ("credit card interest rate raised without notice", "cards"),
    ("the card payment was declined twice", "cards"),
    ("credit report shows an error i disputed", "report"),
    ("wrong account on my credit report", "report"),
    ("report lists a closed account as open", "report"),
    ("credit bureau did not fix the report error", "report"),
    ("identity theft account on my report", "report"),
    ("dispute about report error ignored", "report"),
    ("mortgage escrow payment was miscalculated", "mortgage"),
    ("late fee on my mortgage payment", "mortgage"),
    ("loan servicer lost my mortgage payment", "mortgage"),
    ("escrow shortage raised my mortgage", "mortgage"),
    ("mortgage loan modification denied", "mortgage"),
    ("late payment reported on home loan", "mortgage"),
    ("debt collector called about a loan i paid", "mortgage"),
    ("interest on loan payment is wrong", "mortgage"),
]
TEXTS = [t for t, _ in CORPUS]
LABELS = [y for _, y in CORPUS]


def _fit(vectorizer):
    X = vectorizer.fit_transform(TEXTS)
    clf = LinearSVC(dual="auto", max_iter=5000).fit(X, LABELS)
    return vectorizer, clf


VECTORIZERS = {
    "tfidf": _fit(TfidfVectorizer(ngram_range=(1, 2))),
    "union": _fit(FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2))),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))),
    ])),
}


def _random_text(rng, min_words=1, max_words=12):
    n = int(rng.integers(min_words, max_words + 1))
    return " ".join(rng.choice(VOCAB, size=n))


# ---- Property 1 (input stage) -----------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_property1_input_stage_reflects_text(seed):
    """Feature: prediction-trace-script, Property 1: Input stage reflects the text.

    **Validates: Requirements 2.1**
    """
    rng = np.random.default_rng(seed)
    seps = [" ", "  ", "\t", "\n"]
    words = rng.choice(VOCAB, size=int(rng.integers(1, 15)))
    text = "".join(w + str(rng.choice(seps)) for w in words).strip()
    if rng.random() < 0.5:
        text = str(rng.choice(seps)) + text
    out = tp.trace_input(text)
    assert out["status"] == "ok"
    assert out["raw"] == text
    assert out["n_words"] == len(text.split()) == len(words)
    assert out["cleaned"] == clean_text(text)


# ---- Property 2 (n-gram contributions) --------------------------------------

@pytest.mark.parametrize("kind", sorted(VECTORIZERS))
@pytest.mark.parametrize("seed", SEEDS)
def test_property2_ngram_contributions_exact_and_ordered(seed, kind):
    """Feature: prediction-trace-script, Property 2: N-gram contributions are exact and ordered.

    **Validates: Requirements 3.3**
    """
    rng = np.random.default_rng(seed)
    vec, clf = VECTORIZERS[kind]
    text = _random_text(rng)
    X = vec.transform([text]).tocsr()
    cls = str(rng.choice(clf.classes_))
    coef = np.asarray(tp._coef_row_for(clf, cls)).ravel()
    top_n = int(rng.integers(1, 30))

    out = tp.top_ngram_contributions(vec, coef, X, top_n)

    names = list(vec.get_feature_names_out())
    index = {n: j for j, n in enumerate(names)}
    nnz = int((X.data != 0).sum())
    assert len(out) == min(top_n, nnz)
    for e in out:
        j = index[e["ngram"]]
        assert X[0, j] != 0
        assert e["tfidf"] == pytest.approx(X[0, j])
        assert e["coef"] == pytest.approx(coef[j])
        assert e["contribution"] == pytest.approx(X[0, j] * coef[j])
    contribs = [e["contribution"] for e in out]
    assert contribs == sorted(contribs, reverse=True)
    if kind == "union" and out:
        assert all(e["ngram"].startswith(("word__", "char__")) for e in out)


def test_ngram_contributions_accept_sparse_coef():
    vec, clf = VECTORIZERS["tfidf"]
    import scipy.sparse as sp
    X = vec.transform(["card charged twice"])
    coef = clf.coef_[0]
    dense = tp.top_ngram_contributions(vec, coef, X, 5)
    sparse = tp.top_ngram_contributions(vec, sp.csr_matrix(coef), X, 5)
    assert dense == sparse and dense


# ---- issue head and run_stage examples --------------------------------------

def _issue_bundle():
    texts = TEXTS
    teams = LABELS
    # "cards" has one issue -> _ConstantIssue; others have three -> LinearSVC
    # (three, not two: a binary LinearSVC yields one margin, which the
    # existing predict_issue_topk softmax collapses to a single issue)
    issues = ["c1" if y == "cards" else f"{y[0]}{i % 3}"
              for i, y in enumerate(teams)]
    vec = TfidfVectorizer().fit(texts)
    X = vec.transform(texts)
    hsvm = HierarchicalSVM().fit(X, teams, issues)
    labels = {"c1": "Card issue"}
    labels.update({f"{p}{k}": f"{p.upper()} issue {k}"
                   for p in "rm" for k in range(3)})
    return vec, {"vectorizer": vec, "model": hsvm, "issue_labels": labels}


def test_trace_issue_head_constant():
    vec, bundle = _issue_bundle()
    assert isinstance(bundle["model"].issue_clfs_["cards"], _ConstantIssue)
    X = vec.transform(["card charged twice"])
    out = tp.trace_issue_head(X, bundle, "cards", 3)
    assert out["status"] == "ok"
    assert out["kind"] == "constant"
    assert out["issues"] == [{"issue_id": "c1", "issue_label": "Card issue",
                              "prob": 1.0}]


def test_trace_issue_head_softmax_and_missing_team():
    vec, bundle = _issue_bundle()
    X = vec.transform(["late mortgage payment"])
    out = tp.trace_issue_head(X, bundle, "mortgage", 3)
    assert out["kind"] == "softmax(decision_function)"
    assert {i["issue_id"] for i in out["issues"]} == {"m0", "m1", "m2"}
    assert sum(i["prob"] for i in out["issues"]) == pytest.approx(1.0)
    assert all(i["issue_label"] for i in out["issues"])
    none = tp.trace_issue_head(X, bundle, "no_such_team", 3)
    assert none["issues"] == [] and none["kind"] == "none"


def test_run_stage_maps_import_error_to_skipped():
    def boom():
        raise ImportError("no torch")

    out = tp.run_stage("improved_model_2", boom)
    assert out["status"] == "skipped"
    assert "no torch" in out["reason"]
    assert out["stage"] == "improved_model_2"


def test_run_stage_other_errors_and_required():
    def bad():
        raise ValueError("bad shape")

    out = tp.run_stage("baseline_model_2", bad)
    assert out["status"] == "error"
    assert out["reason"] == "ValueError: bad shape"
    with pytest.raises(tp.TraceFatal):
        tp.run_stage("improved_model_1", bad, required=True)
    ok = tp.run_stage("input", tp.trace_input, "hello world")
    assert ok["status"] == "ok" and ok["n_words"] == 2


# ---- ensemble trace (Property 3 + _decide examples) -------------------------

from src.ensemble import gate as G  # noqa: E402
from src.ensemble.probs import TEAMS  # noqa: E402
from src.ensemble.stack import make_stacker, stack_features, stacker_proba  # noqa: E402
from src.ensemble.thresholds import route_mask  # noqa: E402

ENS_MODELS = ["improved_model_1", "improved_model_2", "baseline_model_2",
              "knn_embed"]
N_TEAMS = len(TEAMS)


def _fit_synthetic_stack(seed: int = 0, n: int = 400) -> dict:
    """A small stack bundle with a real stacker and gate fitted on random
    Dirichlet probability vectors (shape and keys match stack.joblib)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_TEAMS, size=n)
    mats = []
    for _ in ENS_MODELS:
        alpha = np.ones((n, N_TEAMS))
        alpha[np.arange(n), y] += 0.7  # models lean towards the true team
        mats.append(np.vstack([rng.dirichlet(a) for a in alpha]))
    # stacker on the first half, gate on the held-out second half so the
    # gate sees both correct and wrong stacker picks
    h = n // 2
    stacker = make_stacker().fit(stack_features([M[:h] for M in mats]), y[:h])
    rest = [M[h:] for M in mats]
    P = stacker_proba(stacker, stack_features(rest), N_TEAMS)
    chosen = P.argmax(axis=1)
    n_words = rng.integers(5, 300, size=n - h)
    Xg = G.prob_features(chosen, [P] + rest, n_words, N_TEAMS)
    gate = G.fit_gate(Xg, (chosen == y[h:]).astype(int))
    return {
        "model_name": "synthetic_stack",
        "scenario": "stack_gate",
        "models": list(ENS_MODELS),
        "primary_name": "improved_model_1",
        "teams": list(TEAMS),
        "stacker": stacker,
        "gate": gate,
        "gate_feature_names": G.prob_feature_names(["stack"] + ENS_MODELS,
                                                   TEAMS),
        "base_threshold": 0.6,
        "team_thresholds": {TEAMS[0]: {"threshold": 0.9}},
        "source_prior": np.full(N_TEAMS, 1.0 / N_TEAMS),
        "primary_fallback_policy": {"base": 0.7, "teams": None},
    }


SYNTH_STACK = _fit_synthetic_stack()


def _random_probs(rng, models=ENS_MODELS):
    return {m: rng.dirichlet(np.ones(N_TEAMS)) for m in models}


@pytest.mark.parametrize("seed", SEEDS)
def test_property3_missing_models_force_degraded_fallback(seed):
    """Feature: prediction-trace-script, Property 3: Missing models force the degraded fallback.

    **Validates: Requirements 6.1, 6.4**
    """
    rng = np.random.default_rng(seed)
    raw = dict(SYNTH_STACK)
    base = None if rng.random() < 0.15 else float(rng.uniform(0.05, 0.95))
    teams_policy = None
    if rng.random() < 0.5:
        picked = rng.choice(TEAMS, size=int(rng.integers(1, 4)), replace=False)
        teams_policy = {str(t): {"threshold": (None if rng.random() < 0.2
                                               else float(rng.uniform(0, 1)))}
                        for t in picked}
    raw["primary_fallback_policy"] = {"base": base, "teams": teams_policy}

    others = ENS_MODELS[1:]
    k = int(rng.integers(1, len(others) + 1))
    left_out = set(rng.choice(others, size=k, replace=False).tolist())
    probs = _random_probs(rng)
    model_probs = {m: v for m, v in probs.items() if m not in left_out}

    out = tp.trace_ensemble(raw, model_probs, n_words=int(rng.integers(1, 200)))

    expected_missing = [m for m in ENS_MODELS if m in left_out]
    primary = probs["improved_model_1"]
    j = int(primary.argmax())
    assert out["status"] == "ok"
    assert out["path"] == "fallback"
    assert out["scenario"] == "degraded"
    assert out["required_models"] == ENS_MODELS
    assert out["missing"] == expected_missing
    assert out["available_models"] == [m for m in ENS_MODELS
                                       if m not in left_out]
    assert out["reason_degraded"] == f"missing models: {expected_missing}"
    assert out["team"] == TEAMS[j]
    assert out["score"] == pytest.approx(primary[j])
    want_auto = (base is not None and bool(route_mask(
        [primary[j]], [TEAMS[j]], base, teams_policy)[0]))
    assert out["auto_route"] is want_auto


def test_trace_ensemble_error_when_primary_missing():
    rng = np.random.default_rng(0)
    probs = _random_probs(rng, ENS_MODELS[1:])
    out = tp.trace_ensemble(SYNTH_STACK, probs, n_words=10)
    assert out["status"] == "error"
    assert "improved_model_1" in out["reason"]


class _StubGate:
    def __init__(self, score):
        self.score = score
        self.seen = None

    def predict_proba(self, X):
        self.seen = np.asarray(X)
        return np.array([[1.0 - self.score, self.score]])


def _primary_gate_bundle(score, base=0.5, team_thresholds=None):
    return {
        "scenario": "primary_gate",
        "models": list(ENS_MODELS),
        "primary_name": "improved_model_1",
        "teams": list(TEAMS),
        "stacker": None,
        "gate": _StubGate(score),
        "base_threshold": base,
        "team_thresholds": team_thresholds or {},
        "source_prior": None,
    }


def _onehot_probs(top_by_model: dict[str, int]):
    out = {}
    for m, j in top_by_model.items():
        v = np.full(N_TEAMS, 0.02)
        v[j] = 1.0
        out[m] = v / v.sum()
    return out


def test_decide_primary_gate_auto_assisted_contested():
    agree = _onehot_probs({m: 2 for m in ENS_MODELS})
    disagree = _onehot_probs({m: (2 if i < 2 else 5)
                              for i, m in enumerate(ENS_MODELS)})

    bundle = _primary_gate_bundle(0.8, base=0.5)
    out = tp.trace_ensemble(bundle, agree, n_words=20)
    assert out["status"] == "ok" and out["path"] == "decide"
    assert out["team"] == TEAMS[2]
    assert out["scenario"] == "auto" and out["auto_route"] is True
    assert out["score"] == pytest.approx(0.8)
    assert out["threshold"] == pytest.approx(0.5)
    assert out["stacker_used"] is False and out["stack_probs"] is None
    assert out["missing"] == []
    n_feat = bundle["gate"].seen.shape[1]
    assert len(out["gate_features"]) == n_feat == out["n_gate_features"]
    assert out["gate_feature_names_match"] is True
    names = [f["name"] for f in out["gate_features"]]
    assert names == G.prob_feature_names(ENS_MODELS, TEAMS)

    out = tp.trace_ensemble(_primary_gate_bundle(0.3), agree, n_words=20)
    assert out["scenario"] == "assisted" and out["auto_route"] is False
    assert set(out["model_teams"].values()) == {TEAMS[2]}

    out = tp.trace_ensemble(_primary_gate_bundle(0.3), disagree, n_words=20)
    assert out["scenario"] == "contested" and out["auto_route"] is False
    assert out["model_teams"]["knn_embed"] == TEAMS[5]

    # a per-team threshold overrides the base one
    strict = _primary_gate_bundle(0.8, base=0.5,
                                  team_thresholds={TEAMS[2]: {"threshold": 0.95}})
    out = tp.trace_ensemble(strict, agree, n_words=20)
    assert out["threshold"] == pytest.approx(0.95)
    assert out["auto_route"] is False and out["scenario"] == "assisted"


def test_decide_with_fitted_stacker_and_gate():
    rng = np.random.default_rng(7)
    probs = _random_probs(rng)
    n_words = 42
    out = tp.trace_ensemble(SYNTH_STACK, probs, n_words=n_words)
    assert out["status"] == "ok" and out["path"] == "decide"
    assert out["stacker_used"] is True

    mats = [probs[m][None, :] for m in ENS_MODELS]
    P = stacker_proba(SYNTH_STACK["stacker"], stack_features(mats), N_TEAMS)[0]
    j = int(P.argmax())
    Xg = G.prob_features([j], [P[None, :]] + mats, [n_words], N_TEAMS)
    score = float(SYNTH_STACK["gate"].predict_proba(Xg)[0, 1])

    assert out["team"] == TEAMS[j]
    assert sum(out["stack_probs"].values()) == pytest.approx(1.0)
    assert out["stack_probs"][TEAMS[j]] == pytest.approx(P[j])
    assert out["score"] == pytest.approx(score)
    assert len(out["gate_features"]) == Xg.shape[1]
    assert out["gate_feature_names_match"] is True
    assert [f["value"] for f in out["gate_features"]] == pytest.approx(
        Xg[0].tolist())
    want_thr = 0.9 if TEAMS[j] == TEAMS[0] else 0.6
    assert out["threshold"] == pytest.approx(want_thr)
    assert out["auto_route"] is (score >= want_thr)
    assert out["prior_active"] is False
    assert out["source_prior"][TEAMS[0]] == pytest.approx(1.0 / N_TEAMS)
    assert "inactive" in out["label_shift_note"]
    assert out["model_teams"] == {m: TEAMS[int(probs[m].argmax())]
                                  for m in ENS_MODELS}


# ---- CLI, issue resolution and smoke tests ----------------------------------

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402

WHITESPACE = [" ", "\t", "\n", "\r", "\x0b", "\x0c", "  "]


@pytest.mark.parametrize("seed", SEEDS)
def test_property1_whitespace_text_exits_nonzero(seed, capsys):
    """Feature: prediction-trace-script, Property 1: Input stage reflects the text.

    **Validates: Requirements 1.4**
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 8))
    text = "".join(str(rng.choice(WHITESPACE)) for _ in range(n))
    code = tp.main([text, "--skip-embed"])
    assert code == 1
    assert "empty" in capsys.readouterr().err


def test_main_empty_stdin_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("   \n\t"))
    assert tp.main(["--skip-embed"]) == 1
    assert "empty" in capsys.readouterr().err


def test_main_missing_improved_model_1_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tp, "IM1_PATH", tmp_path / "missing.joblib")
    assert tp.main(["card charged twice", "--skip-embed"]) == 2
    assert "Improved Model 1 failed" in capsys.readouterr().err


def test_main_http_bad_scheme_exits_3(capsys):
    assert tp.main(["card charged twice", "--http", "ftp://x/predict"]) == 3
    assert "HTTP request failed" in capsys.readouterr().err


def test_resolve_issues_reuses_or_reranks():
    """**Validates: Requirements 7.1, 7.2**"""
    vec, bundle = _issue_bundle()
    X = vec.transform(["card charged twice late mortgage payment"])
    head = tp.trace_issue_head(X, bundle, "cards", 3)
    primary = {"team": "cards"}

    same = tp.resolve_issues(X, bundle, primary,
                             {"status": "ok", "team": "cards"}, 3,
                             issue_head=head)
    assert same["team"] == "cards"
    assert same["source"] == "improved_model_1"
    assert same["issues"] == head["issues"]

    diff = tp.resolve_issues(X, bundle, primary,
                             {"status": "ok", "team": "mortgage"}, 3,
                             issue_head=head)
    assert diff["team"] == "mortgage"
    assert diff["source"] == "re-ranked for ensemble team"
    assert diff["issues"] == tp.trace_issue_head(X, bundle, "mortgage",
                                                 3)["issues"]
    assert {i["issue_id"] for i in diff["issues"]} <= {"m0", "m1", "m2"}

    # ensemble unavailable -> primary team kept
    skipped = tp.resolve_issues(X, bundle, primary,
                                {"status": "skipped", "reason": "x"}, 3,
                                issue_head=head)
    assert skipped["team"] == "cards"
    assert skipped["source"] == "improved_model_1"


IM1_REAL = REPO_ROOT / "docs" / "metrics" / "improved_model_1" / "model.joblib"
SMOKE_TEXT = ("I was charged twice on my credit card for the same purchase "
              "and the bank refused to refund the duplicate charge.")
STAGE_KEYS = ["input", "improved_model_1", "issue_head", "baseline_model_2",
              "improved_model_2", "knn_embed", "ensemble", "issues", "summary"]


@pytest.mark.skipif(not IM1_REAL.exists(),
                    reason="Improved Model 1 artifact not present")
def test_smoke_run_trace_real_artifacts():
    args = argparse.Namespace(top_k=3, top_ngrams=5, skip_embed=True)
    res = tp.run_trace(SMOKE_TEXT, args)
    st = res["stages"]
    assert list(st) == STAGE_KEYS
    assert all(s["status"] in {"ok", "skipped", "error"} for s in st.values())
    for k in ("input", "improved_model_1", "issues", "summary"):
        assert st[k]["status"] == "ok", (k, st[k].get("reason"))
    assert st["improved_model_2"]["status"] == "skipped"
    assert st["knn_embed"]["status"] == "skipped"
    assert sum(st["improved_model_1"]["team_probs"].values()) == \
        pytest.approx(1.0, abs=1e-6)
    assert st["summary"]["route"] in {"auto", "review"}
    assert st["summary"]["why"]
    buf = io.StringIO()
    tp.print_text(res, out=buf)
    assert "== [8]" in buf.getvalue() or "Summary" in buf.getvalue()


@pytest.mark.skipif(not IM1_REAL.exists(),
                    reason="Improved Model 1 artifact not present")
def test_smoke_main_json_output_parses(capsys):
    """**Validates: Requirements 7.4**"""
    assert tp.main([SMOKE_TEXT, "--skip-embed", "--json", "--top-k", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "in-process"
    assert out["text"] == SMOKE_TEXT
    assert list(out["stages"]) == STAGE_KEYS
    assert "X" not in out["stages"]["improved_model_1"]
    assert len(out["stages"]["summary"]["issues"]) <= 2
