"""Synthetic fitted models and fakes for the explainability tests (Req 21.4).

Everything here is built in memory from generated data in a few seconds. No
artifact file is read and no network is used. The fitted objects use the same
builders as the real training code, so the explainers see the real classes:

* ``synthetic_im1()``: ``build_vectorizer`` + ``HierarchicalSVM`` with an
  isotonic team calibrator. Team ``T01`` has a constant issue head, ``T02`` a
  binary head and every other team a multiclass head.
* ``synthetic_bm2()``: ``build_pipeline`` wrapped in an isotonic
  ``CalibratedClassifierCV(FrozenEstimator(...))``.
* ``FakeEncoder``: deterministic hashed bag-of-words embeddings (dimension 16).
* ``synthetic_stack_bundle()``: a stacker and gate fitted on random Dirichlet
  probability vectors, in a bundle whose keys mirror ``train_stack``.
* ``fake_shap`` / ``fake_shap_wrong``: stand-ins for the ``shap`` module.
* Sentence generators with PII-shaped tokens and out-of-vocabulary words,
  as plain ``random.Random`` functions and as hypothesis strategies.

Builders are cached; each call returns a deep copy, so a test may mutate what
it gets (for example perturb a coefficient) without affecting other tests.
"""
from __future__ import annotations

import copy
import hashlib
import random
import re
import string
import types
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from hypothesis import strategies as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.explain.labels import load_tables

# ---- label space -----------------------------------------------------------

TEAMS: list[str] = sorted(load_tables().teams)
assert len(TEAMS) == 11, TEAMS

MODELS: list[str] = ["improved_model_1", "improved_model_2",
                     "baseline_model_2", "knn_embed"]

CONSTANT_TEAM = "T01"   # one issue  -> _ConstantIssue head
BINARY_TEAM = "T02"     # two issues -> LinearSVC with coef_ of shape (1, F)
# every other team uses three issues -> multiclass LinearSVC


def _issues_by_team() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {t: [] for t in TEAMS}
    for iid, entry in sorted(load_tables().issues.items()):
        out[entry.team_id].append(iid)
    return out


def _issue_plan() -> dict[str, list[str]]:
    """Issue ids used per team: 1 for CONSTANT_TEAM, 2 for BINARY_TEAM, else 3."""
    by_team = _issues_by_team()
    plan = {}
    for t in TEAMS:
        n = 1 if t == CONSTANT_TEAM else 2 if t == BINARY_TEAM else 3
        plan[t] = by_team[t][:n]
    return plan


ISSUE_PLAN: dict[str, list[str]] = _issue_plan()

# ---- sentence generators ---------------------------------------------------

# Words that tie a sentence to a team (strong signal) and to an issue.
_TEAM_WORDS = {
    "T01": ["mortgage", "escrow", "foreclosure"],
    "T02": ["creditreport", "bureau", "dispute"],
    "T03": ["collector", "collection", "harass"],
    "T04": ["checking", "overdraft", "deposit"],
    "T05": ["creditcard", "statement", "rewards"],
    "T06": ["studentloan", "servicer", "forbearance"],
    "T07": ["vehicle", "autoloan", "repossession"],
    "T08": ["paydayloan", "lender", "rollover"],
    "T09": ["moneytransfer", "wire", "remittance"],
    "T10": ["prepaid", "giftcard", "reload"],
    "T11": ["personalloan", "installment", "lease"],
}
_ISSUE_WORDS = ["incorrect", "unauthorized", "delayed", "closed", "missing",
                "denied", "duplicate", "late", "frozen"]
_FILLER = ["i", "the", "my", "account", "bank", "company", "they", "said",
           "called", "about", "and", "was", "not", "please", "help", "with",
           "on", "for", "again", "today"]


def _issue_word(team: str, issue_id: str) -> str:
    return _ISSUE_WORDS[ISSUE_PLAN[team].index(issue_id) * 3
                        + TEAMS.index(team) % 3]


def phone(rng: random.Random) -> str:
    return f"{rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(0, 9999):04d}"


def card(rng: random.Random) -> str:
    return " ".join(f"{rng.randint(0, 9999):04d}" for _ in range(4))


def ssn(rng: random.Random) -> str:
    return f"{rng.randint(100, 899)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"


def email(rng: random.Random) -> str:
    user = "".join(rng.choices(string.ascii_lowercase, k=6))
    return f"{user}@{rng.choice(['example.com', 'mail.org', 'test.net'])}"


def digit_run(rng: random.Random) -> str:
    return "".join(rng.choices(string.digits, k=rng.randint(10, 16)))


PII_GENERATORS = {"phone": phone, "card": card, "ssn": ssn, "email": email,
                  "digit_run": digit_run}


def oov_word(rng: random.Random) -> str:
    """A word no fixture vocabulary contains (prefix 'zq' never occurs)."""
    return "zq" + "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))


def make_sentence(rng: random.Random, team: str, issue_id: str | None = None,
                  pii_rate: float = 0.3, oov_rate: float = 0.3) -> str:
    """One sentence ending in '.', '!' or '?' with team and issue signal."""
    issue_id = issue_id or rng.choice(ISSUE_PLAN[team])
    words = rng.sample(_FILLER, k=rng.randint(3, 6))
    words += rng.sample(_TEAM_WORDS[team], k=2)
    words.append(_issue_word(team, issue_id))
    if rng.random() < oov_rate:
        words.append(oov_word(rng))
    if rng.random() < pii_rate:
        words.append(rng.choice(list(PII_GENERATORS.values()))(rng))
    rng.shuffle(words)
    return " ".join(words).capitalize() + rng.choice([".", ".", "!", "?"])


def make_text(rng: random.Random, team: str, issue_id: str | None = None,
              n_sentences: int | None = None, **kw) -> str:
    """A complaint of 1..4 sentences about one (team, issue)."""
    issue_id = issue_id or rng.choice(ISSUE_PLAN[team])
    n = n_sentences if n_sentences is not None else rng.randint(1, 4)
    sep = rng.choice([" ", " ", "\n"])
    return sep.join(make_sentence(rng, team, issue_id, **kw) for _ in range(n))


def make_corpus(n: int, seed: int = 0, **kw
                ) -> tuple[list[str], list[str], list[str]]:
    """``n`` texts cycling over every (team, issue) in ISSUE_PLAN.

    Returns (texts, team ids, issue ids)."""
    rng = random.Random(seed)
    pairs = [(t, i) for t in TEAMS for i in ISSUE_PLAN[t]]
    texts, teams, issues = [], [], []
    for k in range(n):
        t, i = pairs[k % len(pairs)]
        texts.append(make_text(rng, t, i, **kw))
        teams.append(t)
        issues.append(i)
    return texts, teams, issues


def pii_tokens(rng: random.Random) -> dict[str, str]:
    """One PII-shaped token of every kind."""
    return {k: g(rng) for k, g in PII_GENERATORS.items()}


# Hypothesis strategies over the same vocabulary.
_VOCAB = sorted({w for ws in _TEAM_WORDS.values() for w in ws}
                | set(_ISSUE_WORDS) | set(_FILLER))
_seeded = st.integers(min_value=0, max_value=2**32 - 1).map(random.Random)
st_pii = st.builds(lambda r, k: PII_GENERATORS[k](r), _seeded,
                   st.sampled_from(sorted(PII_GENERATORS)))
st_oov = st.text(alphabet=string.ascii_lowercase, min_size=3, max_size=8).map(
    lambda s: "zq" + s)
st_word = st.one_of(st.sampled_from(_VOCAB), st.sampled_from(_VOCAB), st_pii,
                    st_oov)
st_sentence = st.builds(
    lambda ws, end: " ".join(ws).capitalize() + end,
    st.lists(st_word, min_size=1, max_size=12),
    st.sampled_from([".", "!", "?"]))
st_text = st.builds(
    lambda ss, sep: sep.join(ss),
    st.lists(st_sentence, min_size=1, max_size=8),
    st.sampled_from([" ", "  ", "\n"]))

# ---- improved_model_1 ------------------------------------------------------


@dataclass
class SyntheticIM1:
    vectorizer: object            # FeatureUnion(word, char)
    hsvm: object                  # HierarchicalSVM with calibrated_team_
    texts: list[str]
    teams: list[str]
    issues: list[str]
    val_texts: list[str]
    val_teams: list[str]


@lru_cache(maxsize=4)
def _build_im1(compact: bool, seed: int) -> SyntheticIM1:
    from src.improved_model_1.features import build_vectorizer
    from src.improved_model_1.model import HierarchicalSVM

    texts, teams, issues = make_corpus(220, seed=seed)
    val_texts, val_teams, _ = make_corpus(132, seed=seed + 1)
    vec = build_vectorizer(use_char=True, word_max=500, char_max=500)
    X = vec.fit_transform(texts)
    hsvm = HierarchicalSVM().fit(X, teams, issues)
    hsvm.fit_team_calibrator(vec.transform(val_texts), val_teams)
    if compact:
        hsvm.compact()
    return SyntheticIM1(vec, hsvm, texts, teams, issues, val_texts, val_teams)


def synthetic_im1(compact: bool = False, seed: int = 0) -> SyntheticIM1:
    """IM1 fitted on 220 sentences over 11 teams. ``compact=True`` casts the
    coefficients to float32 as the saved artifact does."""
    return copy.deepcopy(_build_im1(compact, seed))


# ---- baseline_model_2 ------------------------------------------------------


@dataclass
class SyntheticBM2:
    calibrated: CalibratedClassifierCV   # what the service loads
    pipeline: object                     # the inner Pipeline(tfidf, clf)
    texts: list[str]
    teams: list[str]


@lru_cache(maxsize=2)
def _build_bm2(seed: int) -> SyntheticBM2:
    from src.baseline_model_2.train import build_pipeline

    texts, teams, _ = make_corpus(220, seed=seed)
    val_texts, val_teams, _ = make_corpus(132, seed=seed + 1)
    pipe = build_pipeline(max_features=500).fit(texts, teams)
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    cal.fit(val_texts, val_teams)
    return SyntheticBM2(cal, pipe, texts, teams)


def synthetic_bm2(seed: int = 0) -> SyntheticBM2:
    out = copy.deepcopy(_build_bm2(seed))
    # Keep `pipeline` pointing at the object inside the deep-copied calibrator.
    out.pipeline = out.calibrated.calibrated_classifiers_[0].estimator.estimator
    return out


# ---- encoder ---------------------------------------------------------------


class FakeEncoder:
    """Deterministic hashed bag-of-words encoder, dimension 16, L2-normalised.

    Same call shape as ``SentenceTransformer.encode``: a list of texts gives an
    ``(n, 16)`` array, a single string gives a ``(16,)`` array. Extra keyword
    arguments are accepted and ignored. Text without tokens encodes to zeros.
    """

    DIM = 16
    _TOKEN = re.compile(r"\w+")

    def __init__(self, raise_on_call: bool = False):
        self.raise_on_call = raise_on_call
        self.n_calls = 0
        self.n_encoded = 0
        self.batches: list[list[str]] = []

    @classmethod
    def embed(cls, text: str) -> np.ndarray:
        v = np.zeros(cls.DIM, dtype=np.float64)
        for tok in cls._TOKEN.findall(text.lower()):
            h = hashlib.md5(tok.encode("utf-8")).digest()
            v[h[0] % cls.DIM] += 1.0 if h[1] & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts, **_kwargs) -> np.ndarray:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        self.n_calls += 1
        self.batches.append(batch)
        if self.raise_on_call:
            raise RuntimeError("fake encoder failure")
        self.n_encoded += len(batch)
        out = np.vstack([self.embed(t) for t in batch]) if batch else \
            np.zeros((0, self.DIM))
        return out[0] if single else out


# ---- ensemble stack bundle -------------------------------------------------


@dataclass
class SyntheticStack:
    bundle: dict                     # keys mirror src.ensemble.train_stack
    probs: dict[str, np.ndarray]     # per-model (n, 11) probability matrices
    y: np.ndarray                    # true team index per row
    n_words: np.ndarray
    stack_probs: np.ndarray | None   # stacker output (stack scenarios)
    chosen: np.ndarray               # chosen team index the gate was fitted on
    gate_X: np.ndarray               # gate training matrix
    extra: dict = field(default_factory=dict)


def _dirichlet_probs(rng: np.random.Generator, y: np.ndarray, strength: float,
                     n_teams: int) -> np.ndarray:
    """Probability vectors that lean towards the true team (noisily)."""
    alpha = np.full((len(y), n_teams), 0.5)
    hit = rng.random(len(y)) < 0.75
    lean = np.where(hit, y, rng.integers(0, n_teams, len(y)))
    alpha[np.arange(len(y)), lean] += strength
    return np.vstack([rng.dirichlet(a) for a in alpha])


@lru_cache(maxsize=8)
def _build_stack(kind: str, with_reference: bool, scenario: str, n: int,
                 seed: int) -> SyntheticStack:
    from src.ensemble import gate as G
    from src.ensemble.label_shift import prior_from_labels
    from src.ensemble.stack import make_stacker, stack_features, stacker_proba
    from src.utils.config import NORTH_STAR_TARGET, TEAM_PRECISION_FLOOR

    if kind not in G.GATE_KINDS:
        raise ValueError(f"unknown gate kind {kind!r}")
    if scenario not in ("stack_gate", "gate_on_primary"):
        raise ValueError(f"unsupported scenario {scenario!r}")
    T = len(TEAMS)
    rng = np.random.default_rng(seed)
    y = rng.integers(0, T, n)
    probs = {m: _dirichlet_probs(rng, y, s, T)
             for m, s in zip(MODELS, (6.0, 4.0, 3.0, 2.0))}
    Pv = [probs[m] for m in MODELS]
    n_words = rng.integers(5, 400, n)

    use_stack = scenario.startswith("stack")
    stacker = make_stacker(1.0).fit(stack_features(Pv), y) if use_stack else None
    stack_P = stacker_proba(stacker, stack_features(Pv), T) if use_stack else None
    gate_mats = ([stack_P] + Pv) if use_stack else Pv
    chosen = gate_mats[0].argmax(1)
    Xg = G.prob_features(chosen, gate_mats, n_words, T)
    correct = (chosen == y).astype(int)
    gate = G.make_estimator(kind, seed).fit(Xg, correct)

    scores = gate.predict_proba(Xg)[:, 1]
    base = float(np.round(np.median(scores), 6))
    team_thr: dict[str, float | None] = {t: base for t in TEAMS}
    team_thr[TEAMS[-1]] = None   # a team that is never auto-routed
    team_thr[TEAMS[0]] = float(np.round(min(1.0, base + 0.05), 6))

    bundle = {
        "model_name": "ensemble_stack_v2", "models": list(MODELS),
        "teams": list(TEAMS), "scenario": scenario,
        "stacker": stacker,
        "gate": gate, "gate_kind": kind,
        "base_threshold": base, "team_thresholds": team_thr,
        "target": NORTH_STAR_TARGET,
        "margin": {"margin": 0.0025, "reason": "fixed"},
        "floor": TEAM_PRECISION_FLOOR,
        "gate_feature_names": G.prob_feature_names(
            (["stack"] if use_stack else []) + list(MODELS), TEAMS),
        "source_prior": prior_from_labels(np.asarray(TEAMS)[y], TEAMS).tolist(),
        "primary_name": MODELS[0],
        "primary_fallback_policy": {"base": base, "teams": None},
    }
    if with_reference:
        # Task 11.1 will make train_stack write this key; mirror it here.
        bundle["gate_reference_values"] = np.median(Xg, axis=0).tolist()
    assert len(bundle["gate_feature_names"]) == Xg.shape[1]
    return SyntheticStack(bundle, probs, y, n_words, stack_P, chosen, Xg)


def synthetic_stack_bundle(kind: str = "logreg", with_reference: bool = True,
                           scenario: str = "stack_gate", n: int = 600,
                           seed: int = 0) -> SyntheticStack:
    """Stacker + gate fitted on ``n`` random Dirichlet rows over 4 models.

    ``kind`` is ``"logreg"`` or ``"hgb"``. ``with_reference`` adds
    ``gate_reference_values`` (the gate training-matrix medians)."""
    return copy.deepcopy(_build_stack(kind, with_reference, scenario, n, seed))


# ---- fake shap -------------------------------------------------------------


def _path_attributions(model, X: np.ndarray, baseline: np.ndarray
                       ) -> np.ndarray:
    """Per-feature values that sum exactly to f(x) - f(baseline).

    Features are switched from the baseline to x one at a time; feature j gets
    the change in the raw margin at step j. The sum telescopes, so with
    ``expected_value = f(baseline)`` the result is exact like TreeSHAP.
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros_like(X)
    for r, x in enumerate(X):
        steps = np.tile(baseline, (X.shape[1] + 1, 1))
        for j in range(X.shape[1]):
            steps[j + 1:, j] = x[j]
        f = model.decision_function(steps)
        out[r] = np.diff(f)
    return out


def make_fake_shap(offset: float = 0.0) -> types.ModuleType:
    """A module object exposing ``TreeExplainer`` like ``shap``.

    ``TreeExplainer(model, model_output="raw").shap_values(X)`` returns an
    ``(n, F)`` array and ``expected_value`` a float. With ``offset == 0`` the
    values are exact: ``sum + expected_value == model.decision_function(X)``.
    A non-zero offset is added to every value, so the compatibility check
    fails and the explainer must fall back to ablation."""
    mod = types.ModuleType("shap")
    mod.__version__ = "0.0-fake" if offset == 0 else "0.0-fake-wrong"

    class TreeExplainer:
        def __init__(self, model, data=None, model_output="raw", **_kwargs):
            if model_output != "raw":
                raise ValueError("fake shap supports model_output='raw' only")
            self.model = model
            n_features = int(model.n_features_in_)
            self.baseline = np.zeros(n_features, dtype=np.float64)
            self.expected_value = float(
                model.decision_function(self.baseline[None, :])[0])
            self.calls = 0

        def shap_values(self, X):
            self.calls += 1
            return _path_attributions(self.model, X, self.baseline) + offset

    mod.TreeExplainer = TreeExplainer
    return mod


fake_shap = make_fake_shap()
fake_shap_wrong = make_fake_shap(offset=0.5)


__all__ = [
    "TEAMS", "MODELS", "ISSUE_PLAN", "CONSTANT_TEAM", "BINARY_TEAM",
    "PII_GENERATORS", "phone", "card", "ssn", "email", "digit_run", "oov_word",
    "pii_tokens", "make_sentence", "make_text", "make_corpus",
    "st_pii", "st_oov", "st_word", "st_sentence", "st_text",
    "SyntheticIM1", "synthetic_im1", "SyntheticBM2", "synthetic_bm2",
    "FakeEncoder", "SyntheticStack", "synthetic_stack_bundle",
    "make_fake_shap", "fake_shap", "fake_shap_wrong",
]


# ---- improved_model_2 classifier -------------------------------------------


@dataclass
class SyntheticIM2:
    classifier: object               # LogisticRegression on FakeEncoder embeddings
    team_classes: list[str]
    texts: list[str]
    teams: list[str]


@lru_cache(maxsize=2)
def _build_im2(seed: int) -> SyntheticIM2:
    from sklearn.linear_model import LogisticRegression

    texts, teams, _ = make_corpus(220, seed=seed)
    X = np.vstack([FakeEncoder.embed(t) for t in texts])
    clf = LogisticRegression(max_iter=1000).fit(X, teams)
    return SyntheticIM2(clf, [str(c) for c in clf.classes_], texts, teams)


def synthetic_im2(seed: int = 0) -> SyntheticIM2:
    """A small IM2-like classifier: LogisticRegression on ``FakeEncoder``
    embeddings of 220 synthetic complaints. Pair it with ``FakeEncoder()``."""
    return copy.deepcopy(_build_im2(seed))
