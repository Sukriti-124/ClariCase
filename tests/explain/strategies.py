"""Hypothesis strategies for valid DecisionRecords (Req 2.4, 21.4).

``records()`` draws ``DecisionRecord`` objects and ``record_dicts()`` draws
the plain dicts they are built from. Every draw passes the schema validators
in ``src.explain.record``. Coverage:

* floats: subnormals, +/-0.0, huge and tiny magnitudes, and values within an
  ulp of a 6-decimal rounding boundary (``x.xxxxxx5``);
* unicode strings: arbitrary non-surrogate text plus fixed samples (accents,
  CJK, emoji, RTL, NUL, zero-width and combining characters);
* every section status (``ok``, ``skipped``, ``error``), with ``reason`` and
  ``drivers`` consistent with the status;
* absent fields: optional fields left out, and ``Absent`` markers wherever the
  schema allows one;
* both serving paths, every scenario (streamlit is always ``primary_only``)
  and both issue orders (``probability``, ``agentic``).

Floats that the record validators need to stay non-zero (the contribution of
a redacted Driver) are kept at magnitude >= 1e-5, so they survive the 6-decimal
rounding of ``canonical_json`` and a re-parse.

``records(serving_path=..., scenario=..., issue_order=..., status=...)`` pins
any of those dimensions; leave an argument as ``None`` to draw it.
"""
from __future__ import annotations

import math
import sys
from typing import Any

from hypothesis import strategies as st

from src.explain.record import (AGENTIC_RATIONALE, CALIBRATION_NOTE,
                                SECTION_NAMES, DecisionRecord)

STATUSES = ("ok", "skipped", "error")
SERVING_PATHS = ("streamlit", "platform")
SCENARIOS = ("auto", "assisted", "contested", "degraded", "primary_only")
ISSUE_ORDERS = ("probability", "agentic")
METHODS = ("exact", "approximate")

# Plausible techniques per explainer. The schema accepts any technique, but
# realistic pairs make the records easier to read when a test fails.
_TECHNIQUES = {
    "improved_model_1": ("linear",),
    "baseline_model_2": ("nb_llr",),
    "improved_model_2": ("occlusion",),
    "knn_embed": ("neighbours",),
    "stacker": ("linear", "shap", "ablation", "constant"),
    "gate": ("shap", "ablation", "constant"),
    "routing": ("rules",),
    "agentic": ("rules",),
}

# ---------------------------------------------------------------------------
# Floats
# ---------------------------------------------------------------------------

_MIN_SUBNORMAL = 5e-324
_MIN_NORMAL = sys.float_info.min
_SPECIAL_FLOATS = (0.0, -0.0, _MIN_SUBNORMAL, -_MIN_SUBNORMAL, _MIN_NORMAL,
                   -_MIN_NORMAL, _MIN_NORMAL / 3, 1e-7, -1e-7, 4.9999e-7, 5e-7,
                   -5e-7, 5.0001e-7, 1e-6, 0.1234565, 0.5000005, -0.0000005,
                   0.9999995, 1.0, -1.0, 2.5e-7, 1e300, -1e300,
                   sys.float_info.max, -sys.float_info.max)


def _near_boundary(lo: int, hi: int) -> st.SearchStrategy[float]:
    """Floats at or within one ulp of ``(n + 0.5) / 1e6`` for n in [lo, hi]."""
    return st.builds(
        lambda n, step: _step((n + 0.5) / 1e6, step),
        st.integers(lo, hi), st.sampled_from((-1, 0, 1)))


def _step(x: float, step: int) -> float:
    if step < 0:
        return math.nextafter(x, -math.inf)
    if step > 0:
        return math.nextafter(x, math.inf)
    return x


st_float = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False),       # includes subnormals
    st.floats(-1e3, 1e3, allow_nan=False),
    st.floats(-1e-300, 1e-300, allow_nan=False),            # tiny and subnormal
    st.sampled_from(_SPECIAL_FLOATS),
    _near_boundary(-10**7, 10**7),
)

_PROB_SPECIALS = tuple(x for x in _SPECIAL_FLOATS if 0.0 <= x <= 1.0)
st_prob = st.one_of(
    st.floats(0.0, 1.0),
    st.floats(0.0, 1e-300),
    st.sampled_from(_PROB_SPECIALS),
    _near_boundary(0, 999_998),                              # stays in (0, 1)
)

# Non-zero after 6-decimal rounding, for redacted Driver contributions.
st_nonzero_float = st.one_of(
    st.floats(1e-5, 1e6), st.floats(-1e6, -1e-5),
    _near_boundary(10, 10**6), _near_boundary(-10**6, -11))

st_nonneg_int = st.integers(0, 10**9)

# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------

_UNICODE_SAMPLES = ("", " ", "é", "naïve café", "日本語テキスト", "🙂👍🏽",
                    "مرحبا بالعالم", "שלום", "a\u0000b", "zero\u200bwidth",
                    "e\u0301", "tab\tnew\nline", "\"quoted\" \\ back",
                    "\U0001F600\U0001F4A9", "Ω≈ç√∫", "\ufeffbom")

st_text = st.one_of(st.text(max_size=12), st.sampled_from(_UNICODE_SAMPLES))

_nonblank_char = st.characters(
    blacklist_categories=("Cs", "Cc", "Zs", "Zl", "Zp")).filter(
        lambda c: not c.isspace())
# A string with a non-whitespace character, so ``s.strip()`` is non-empty.
st_nonblank = st.builds(lambda a, c, b: a + c + b, st_text, _nonblank_char,
                        st_text)
st_id = st.one_of(st.text(min_size=1, max_size=8),
                  st.sampled_from(("T1", "team_ü", "issue-日本", "🙂")))

# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

st_absent = st.builds(lambda r: {"absent": True, "reason": r},
                      st.text(min_size=1, max_size=12))


def _opt(**fields: st.SearchStrategy) -> st.SearchStrategy[dict]:
    """A dict where every field may be left out (an absent field)."""
    return st.fixed_dictionaries({}, optional=fields)


def _maybe(s: st.SearchStrategy) -> st.SearchStrategy:
    return st.one_of(st.none(), s)


def _dict_of(values: st.SearchStrategy) -> st.SearchStrategy[dict]:
    return st.dictionaries(st_id, values, max_size=3)


@st.composite
def st_span(draw) -> dict:
    start = draw(st_nonneg_int)
    return {"index": draw(st.integers(0, 50)), "start": start,
            "end": start + draw(st.integers(0, 10**6))}


st_plain_driver = st.builds(
    lambda kind, extra, c: {"kind": kind, **extra, "contribution": c},
    st.sampled_from(("ngram", "sentence", "model", "feature_group")),
    _opt(label=_maybe(st_text), text=_maybe(st_text), value=_maybe(st_float),
         span=_maybe(st_span())),
    st_float)
st_redacted_driver = st.builds(
    lambda c: {"kind": "redacted", "contribution": c}, st_nonzero_float)


@st.composite
def st_drivers(draw, allowed: bool = True) -> list[dict]:
    """At most one redacted Driver, at a random position; [] if not allowed."""
    if not allowed:
        return []
    drivers = draw(st.lists(st_plain_driver, max_size=4))
    if draw(st.booleans()):
        drivers.insert(draw(st.integers(0, len(drivers))),
                       draw(st_redacted_driver))
    return drivers


# ---------------------------------------------------------------------------
# Per-explainer details
# ---------------------------------------------------------------------------

_st_isotonic = st.fixed_dictionaries({
    "raw_decision": st_float, "isotonic_pre_norm": st_prob,
    "calibrated_prob": st_prob})


@st.composite
def st_issue_subsection(draw, drivers_allowed: bool) -> dict:
    kind = draw(st.sampled_from(("constant", "linear_multiclass", "linear_binary")))
    sub = {"kind": kind, "method": draw(st.sampled_from(METHODS)),
           "technique": "constant" if kind == "constant" else "linear",
           "issue_id": draw(st_id)}
    if kind != "constant":
        sub["drivers"] = draw(st_drivers(drivers_allowed))
    sub.update(draw(_opt(
        compared_issue=_maybe(st_id), constant=_maybe(st_float),
        raw_diff=_maybe(st_float), residual=_maybe(st_float),
        served_diff=_maybe(st_float),
        served_decision_deviation=_maybe(st_float),
        n_nonzero=_maybe(st_nonneg_int),
        redacted_contribution=_maybe(st_float),
        ranking_source=_maybe(st.sampled_from(
            ("calibrated", "softmax(decision_function)"))))))
    return sub


def _im1(ok: bool) -> st.SearchStrategy[dict]:
    return _opt(top_team=_maybe(st_id), runner_up=_maybe(st_id),
                raw_diff=_maybe(st_float), constant=_maybe(st_float),
                residual=_maybe(st_float), served_diff=_maybe(st_float),
                served_decision_deviation=_maybe(st_float),
                n_nonzero=_maybe(st_nonneg_int),
                redacted_contribution=_maybe(st_float),
                uncalibrated_source=_maybe(st_text),
                isotonic=_maybe(_dict_of(_st_isotonic)),
                calibration_note=_maybe(st.one_of(st.just(CALIBRATION_NOTE),
                                                  st_text)),
                issue=_maybe(st_issue_subsection(ok)))


_BM2 = _opt(chosen_team=_maybe(st_id), runner_up=_maybe(st_id),
            raw_diff=_maybe(st_float), constant=_maybe(st_float),
            residual=_maybe(st_float), n_nonzero=_maybe(st_nonneg_int),
            redacted_contribution=_maybe(st_float),
            calibrated_prob_chosen=_maybe(st_prob),
            uncalibrated_llr=_maybe(st_float),
            explains=_maybe(st.just("uncalibrated_naive_bayes_score")),
            note=_maybe(st_text))

_IM2 = _opt(chosen_team=_maybe(st_id),
            k=_maybe(st.one_of(st.integers(-5, 500), st_text)),
            k_eff=_maybe(st.integers(1, 200)),
            k_default_applied=_maybe(st.booleans()),
            n_sentences=_maybe(st_nonneg_int), n_spans=_maybe(st_nonneg_int),
            spans_merged=_maybe(st.booleans()),
            n_encoded_texts=_maybe(st_nonneg_int), p_full=_maybe(st_prob),
            team_probs=_maybe(_dict_of(st_prob)))

_st_neighbour = st.builds(
    lambda rank, sim, team, extra, cid: {"rank": rank, "similarity": sim,
                                         "team_id": team, **extra,
                                         "complaint_id": cid},
    st.integers(1, 200), st_float, st_id, _opt(issue_id=_maybe(st_id)), st_id)

_KNN = _opt(k=_maybe(st_nonneg_int), n_index=_maybe(st_nonneg_int),
            effective_k=_maybe(st_nonneg_int),
            team_distribution=_maybe(_dict_of(st_prob)),
            max_abs_diff_vs_served=_maybe(st_float),
            neighbours=st.lists(_st_neighbour, max_size=3),
            reason_ids=_maybe(st_text))

_STACKER = _opt(case=_maybe(st.sampled_from(("stacker_agrees",
                                             "stacker_disagrees"))),
                chosen_team=_maybe(st_id), comparison_team=_maybe(st_id),
                intercept_diff=_maybe(st_float), logit_diff=_maybe(st_float),
                residual=_maybe(st_float),
                ranking=_maybe(st.lists(st_id, max_size=4)),
                carrying_model=_maybe(st_text),
                label_shift_active=_maybe(st.booleans()),
                p_before=_maybe(_dict_of(st_prob)),
                p_after=_maybe(_dict_of(st_prob)),
                top_before=_maybe(st_id), top_after=_maybe(st_id),
                shift_changed_top=_maybe(st.booleans()))

_float_or_absent = _maybe(st.one_of(st_float, st_absent))
_prob_or_absent = _maybe(st.one_of(st_prob, st_absent))

_GATE = _opt(gate_kind=_maybe(st_text), score=_maybe(st_prob),
             threshold=_float_or_absent,
             score_ge_threshold=_maybe(st.booleans()),
             unit=_maybe(st.sampled_from(("log-odds", "change in gate score"))),
             constant=_maybe(st_float), residual=_maybe(st_float))

_ROUTING = _opt(score=_prob_or_absent, threshold=_float_or_absent,
                threshold_reason=_maybe(st_text),
                threshold_kind=_maybe(st.sampled_from(("base", "per-team"))),
                threshold_team=_maybe(st_id), rule=_maybe(st_text),
                pre_rescue_auto=_maybe(st.booleans()),
                scenario=_maybe(st.sampled_from(SCENARIOS)),
                model_teams=_maybe(_dict_of(st.one_of(st_id, st_absent))),
                pre_llm_team=_maybe(st.one_of(st_id, st_absent)),
                label_shift_active=_maybe(st.booleans()),
                prior_n_seen=_maybe(st.one_of(st_nonneg_int, st_absent)),
                prior_min_rows=_maybe(st_nonneg_int),
                missing_models=_maybe(st.lists(st_id, max_size=3)),
                fallback_threshold=_float_or_absent,
                primary_score=_prob_or_absent,
                rescue_threshold=_float_or_absent,
                llm_kept_team=_maybe(st.booleans()),
                pre_llm_auto=_maybe(st.booleans()),
                rescued=_maybe(st.booleans()))

_task_status = _maybe(st.sampled_from(("chosen", "failed", "not_asked")))
_AGENTIC = _opt(team_candidates=_maybe(st.one_of(st.lists(st_id, max_size=4),
                                                 st_absent)),
                team_choice=_maybe(st.one_of(st_id, st_absent)),
                rescued=_maybe(st.one_of(st.booleans(), st_absent)),
                issue_candidates=_maybe(st.one_of(st.lists(st_id, max_size=4),
                                                  st_absent)),
                issue_choice=_maybe(st.one_of(st_id, st_absent)),
                team_status=_task_status, team_status_reason=_maybe(st_text),
                issue_status=_task_status, issue_status_reason=_maybe(st_text),
                team_changed=_maybe(st.booleans()),
                team_confidence_refers_to=_maybe(st_text),
                rationale=st.just(AGENTIC_RATIONALE))

_DETAILS = {"baseline_model_2": _BM2, "improved_model_2": _IM2,
            "knn_embed": _KNN, "stacker": _STACKER, "gate": _GATE,
            "routing": _ROUTING, "agentic": _AGENTIC}


# ---------------------------------------------------------------------------
# Sections, outcome, record
# ---------------------------------------------------------------------------

@st.composite
def section_dicts(draw, name: str, status: str | None = None) -> dict:
    """A section dict for explainer ``name`` that passes its validators."""
    status = status or draw(st.sampled_from(STATUSES))
    ok = status == "ok"
    sec: dict[str, Any] = {"explainer": name, "status": status,
                           "method": draw(st.sampled_from(METHODS)),
                           "technique": draw(st.sampled_from(_TECHNIQUES[name]))}
    if ok:
        # Left out, null, or blank: all count as "no reason".
        reason = draw(st.sampled_from(("omit", None, "", "   ")))
        if reason != "omit":
            sec["reason"] = reason
    else:
        sec["reason"] = draw(st_nonblank)
    if draw(st.booleans()):
        sec["drivers"] = draw(st_drivers(ok))
    if draw(st.integers(0, 4)):              # sometimes rely on the default
        sec["details"] = draw(_im1(ok) if name == "improved_model_1"
                              else _DETAILS[name])
    return sec


_st_issue = st.builds(
    lambda iid, label, disp, p: {"issue_id": iid, "label": label,
                                 "display_label": disp, "prob": p},
    st_id, st_text, st_text, st_prob)


@st.composite
def outcome_dicts(draw, scenario: str | None = None,
                  issue_order: str | None = None) -> dict:
    issue_order = issue_order or draw(st.sampled_from(ISSUE_ORDERS))
    issues = draw(st.lists(_st_issue, max_size=3,
                           unique_by=lambda i: i["issue_id"]))
    if issue_order == "probability":
        issues.sort(key=lambda i: -i["prob"])
    elif len(issues) > 1 and draw(st.booleans()):
        issues.sort(key=lambda i: i["prob"])   # agentic may rank against prob
    for rank, issue in enumerate(issues, start=1):
        issue["rank"] = rank
    auto = draw(st.booleans())
    return {"chosen_team": {"id": draw(st_id), "name": draw(st_text)},
            "top_issues": issues, "issue_order": issue_order,
            "auto_route": auto, "route": "auto" if auto else "review",
            "scenario": scenario or draw(st.sampled_from(SCENARIOS))}


@st.composite
def record_dicts(draw, serving_path: str | None = None,
                 scenario: str | None = None, issue_order: str | None = None,
                 status: str | None = None) -> dict:
    """A record dict that ``DecisionRecord.model_validate`` accepts.

    ``status`` pins every section's status; otherwise each is drawn.
    """
    serving_path = serving_path or draw(st.sampled_from(SERVING_PATHS))
    if serving_path == "streamlit":
        if scenario not in (None, "primary_only"):
            raise ValueError("the streamlit path only has scenario 'primary_only'")
        scenario = "primary_only"
    order = draw(st.permutations(SECTION_NAMES))
    rec: dict[str, Any] = {
        "request_id": draw(st_text), "created_at": draw(st_text),
        "serving_path": serving_path,
        "model_versions": draw(st.dictionaries(st_id, st_text, max_size=4)),
        "outcome": draw(outcome_dicts(scenario, issue_order)),
        "sections": {n: draw(section_dicts(n, status)) for n in order},
    }
    if draw(st.booleans()):
        rec["schema_version"] = "1"
    timings = draw(st.one_of(st.just("omit"), st.none(),
                             st.dictionaries(st_id, st.integers(0, 10**7),
                                             max_size=4)))
    if timings != "omit":
        rec["timings"] = timings
    return rec


def records(serving_path: str | None = None, scenario: str | None = None,
            issue_order: str | None = None,
            status: str | None = None) -> st.SearchStrategy[DecisionRecord]:
    """Valid ``DecisionRecord`` objects; see ``record_dicts`` for the arguments."""
    return record_dicts(serving_path, scenario, issue_order, status).map(
        DecisionRecord.model_validate)


st_record_dict = record_dicts()
st_record = records()
