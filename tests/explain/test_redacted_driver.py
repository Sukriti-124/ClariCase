"""Property test for the Redacted_Driver (task 6.3, design Property 37).

The scored ``text`` carries PII-shaped tokens; the explainers see
``masked_text = mask(text)`` (optionally with one more vocabulary word
dropped), so some non-zero n-grams of the scored text are not derivable from
the Masked_Text. The expected folded contributions are recomputed here from
the fitted models and the vectorizers' own analyzers, not read back from the
sections.
"""
from __future__ import annotations

import math
import random
import re

import numpy as np
from hypothesis import given, settings, target
from hypothesis import strategies as st

from src.app.pii import mask
from src.explain.explainers.bm2 import explain_bm2, inner_pipeline
from src.explain.explainers.im1 import explain_im1
from tests.explain.fixtures import (TEAMS, card, make_text, phone, ssn, st_pii,
                                    st_text, synthetic_bm2, synthetic_im1)

TOL = 1e-6
_MODELS: dict[object, object] = {}


def _im1(compact: bool):
    key = ("im1", compact)
    if key not in _MODELS:
        _MODELS[key] = synthetic_im1(compact=compact)
    return _MODELS[key]


def _bm2():
    if "bm2" not in _MODELS:
        _MODELS["bm2"] = synthetic_bm2()
    return _MODELS["bm2"]


# ---- independent recomputation -------------------------------------------

def _derivable(name: str, masked: str, vec) -> bool:
    """Is feature ``name`` produced by its analyzer on ``masked`` and
    unchanged by ``mask`` (with and without its FeatureUnion prefix)?"""
    parts = dict(getattr(vec, "transformer_list", ()))
    if parts:
        prefix, _, gram = name.partition("__")
        sub = parts.get(prefix)
    else:
        sub, gram = vec, name
    if sub is None or not gram:
        return False
    grams = {g.lower() for g in sub.build_analyzer()(masked)}
    return gram.lower() in grams and mask(name) == name and mask(gram) == gram


def _row(vec, text):
    x = vec.transform([text]).tocsr().astype(np.float64)
    x.sum_duplicates()
    return x


def _contribs(x, W, beta, rows_or_sign):
    """Per-feature contributions, constant and raw difference in float64."""
    cols = x.indices
    if isinstance(rows_or_sign, tuple):
        a, b = rows_or_sign
        c = x.data * (W[a, cols] - W[b, cols])
        const = float(beta[a] - beta[b])
        sc = np.asarray(x @ W[[a, b]].T).ravel() + beta[[a, b]]
        raw = float(sc[0] - sc[1])
    else:
        s = rows_or_sign
        c = x.data * (s * W[0, cols])
        const = float(s * beta[0])
        raw = float(s * (np.asarray(x @ W[[0]].T).ravel()[0] + beta[0]))
    return c, const, raw


def _check(drivers, x, contribs, const, raw, names, masked, vec,
           section_raw, section_const, residual, method, plain_method,
           redacted_contribution):
    """Property 37 for one section (or issue sub-section); returns 1 when a
    Redacted_Driver is present, else 0."""
    nz = [(str(names[f]), float(c)) for f, c in zip(x.indices.tolist(),
                                                   contribs.tolist()) if c != 0.0]
    hidden = [c for nm, c in nz if not _derivable(nm, masked, vec)]
    expected = math.fsum(hidden)

    red = [d for d in drivers if d.kind == "redacted"]
    assert len(red) <= 1
    if expected != 0.0:
        assert len(red) == 1
        r = red[0]
        assert (r.label, r.text, r.value, r.span) == (None, None, None, None)
        assert math.isclose(r.contribution, expected, rel_tol=1e-12, abs_tol=1e-12)
        assert redacted_contribution == r.contribution
    else:
        assert red == [] and redacted_contribution is None

    # No reported text Driver is non-derivable, and each is a real contribution.
    by_name = dict(nz)
    for d in drivers:
        if d.kind in ("ngram", "sentence"):
            assert d.kind == "ngram" and _derivable(d.label, masked, vec)
            assert math.isclose(d.contribution, by_name[d.label], abs_tol=1e-12)

    # Exactness over all non-zero features, folded ones included.
    assert math.isclose(section_raw, raw, abs_tol=1e-9)
    assert section_const == const
    assert abs(math.fsum(contribs.tolist()) + const - raw) <= TOL
    assert residual <= TOL
    assert method == plain_method == "exact"
    return len(red)


def _check_im1(m, text, masked, chosen):
    cal_classes = list(m.hsvm.calibrated_team_.classes_)
    sec = explain_im1(m.vectorizer, m.hsvm, cal_classes, text, masked, chosen, [])
    plain = explain_im1(m.vectorizer, m.hsvm, cal_classes, text, text, chosen, [])
    vec = m.vectorizer
    names = vec.get_feature_names_out()
    x = _row(vec, text)
    d = sec.details
    assert sec.status == "ok"

    svc = m.hsvm.team_clf
    cls = [str(c) for c in svc.classes_]
    W = np.asarray(svc.coef_, dtype=np.float64)
    beta = np.asarray(svc.intercept_, dtype=np.float64).ravel()
    c, const, raw = _contribs(x, W, beta, (cls.index(d.top_team),
                                           cls.index(d.runner_up)))
    assert (d.top_team, d.runner_up) == (plain.details.top_team,
                                         plain.details.runner_up)
    assert d.raw_diff == plain.details.raw_diff
    n_red = _check(sec.drivers, x, c, const, raw, names, masked, vec, d.raw_diff,
                   d.constant, d.residual, sec.method, plain.method,
                   d.redacted_contribution if x.nnz else None)

    iss, piss = d.issue, plain.details.issue
    clf = m.hsvm.issue_clfs_[chosen]
    if iss.kind == "constant":
        assert iss.drivers == []
        return n_red
    icls = [str(k) for k in clf.classes_]
    Wi = np.asarray(clf.coef_, dtype=np.float64)
    bi = np.asarray(clf.intercept_, dtype=np.float64).ravel()
    if iss.kind == "linear_binary":
        key = 1.0 if iss.issue_id == icls[1] else -1.0
    else:
        key = (icls.index(iss.issue_id), icls.index(iss.compared_issue))
    ci, consti, rawi = _contribs(x, Wi, bi, key)
    assert (iss.issue_id, iss.compared_issue) == (piss.issue_id, piss.compared_issue)
    n_red += _check(iss.drivers, x, ci, consti, rawi, names, masked, vec,
                    iss.raw_diff, iss.constant, iss.residual, iss.method,
                    piss.method,
                    getattr(iss, "redacted_contribution", None) if x.nnz else None)
    return n_red


def _check_bm2(m, text, masked, chosen):
    sec = explain_bm2(m.calibrated, text, masked, chosen)
    plain = explain_bm2(m.calibrated, text, text, chosen)
    pipe = inner_pipeline(m.calibrated)
    tfidf, nb = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    cls = [str(k) for k in nb.classes_]
    x = _row(tfidf, text)
    d = sec.details
    assert sec.status == "ok" and d.runner_up == plain.details.runner_up
    W = np.asarray(nb.feature_log_prob_, dtype=np.float64)
    prior = np.asarray(nb.class_log_prior_, dtype=np.float64)
    c, const, raw = _contribs(x, W, prior, (cls.index(chosen), cls.index(d.runner_up)))
    return _check(sec.drivers, x, c, const, raw, tfidf.get_feature_names_out(),
                  masked, tfidf, d.raw_diff, d.constant, d.residual, sec.method,
                  plain.method, getattr(d, "redacted_contribution", None))


# ---- strategies ------------------------------------------------------------

_WORD = re.compile(r"[a-z]+")


@st.composite
def st_case(draw):
    """``(text, masked_text)``: text with at least one PII token; masked is
    ``mask(text)``, sometimes with one more word dropped."""
    base = draw(st_text)
    pii = draw(st.lists(st_pii, min_size=1, max_size=3))
    pos = draw(st.integers(0, len(base)))
    head, tail = base[:pos], base[pos:]
    text = f"{head} {' '.join(pii)} {tail}".strip()
    masked = mask(text)
    words = sorted(set(_WORD.findall(masked.lower())))
    if words and draw(st.booleans()):
        w = draw(st.sampled_from(words))
        masked = re.sub(rf"(?i)\b{w}\b", " ", masked)
    return text, masked


# Feature: model-explainability, Property 37: Redacted_Driver keeps exactness
@settings(max_examples=100, deadline=None)
@given(case=st_case(), chosen=st.sampled_from(TEAMS), compact=st.booleans())
def test_redacted_driver_keeps_exactness(case, chosen, compact):
    """**Validates: Requirements 1.6, 1.12, 3.4, 4.3**"""
    text, masked = case
    n_red = _check_im1(_im1(compact), text, masked, chosen)
    n_red += _check_bm2(_bm2(), text, masked, chosen)
    target(float(n_red), label="redacted sections")


# Feature: model-explainability, Property 37: Redacted_Driver keeps exactness
def test_redacted_driver_present_for_injected_pii():
    """A deterministic case where every section folds a non-zero sum.

    **Validates: Requirements 1.6, 1.12, 3.4, 4.3**"""
    rng = random.Random(7)
    team = "T04"                       # multiclass issue head
    text = (make_text(rng, team, pii_rate=0.0, oov_rate=0.0, n_sentences=2)
            + f" Call {phone(rng)} card {card(rng)} ssn {ssn(rng)} today.")
    # mask() hides the PII; dropping the team words also hides in-vocabulary
    # word and char n-grams the IM1 heads weight.
    masked = re.sub(r"(?i)\b(checking|deposit|overdraft)\b", " ", mask(text))
    assert masked != text
    for compact in (False, True):
        assert _check_im1(_im1(compact), text, masked, team) == 2
    assert _check_bm2(_bm2(), text, masked, team) == 1
