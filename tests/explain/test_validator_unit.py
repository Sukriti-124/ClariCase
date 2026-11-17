"""Unit tests for the Faithfulness_Validator (src/explain/validator.py).

Requirements 13.1 to 13.11, 15.1 to 15.3, 15.5. Template acceptance over
records built from the real label tables (Req 13.12, 15.9).
"""
from __future__ import annotations

import pytest

from src.explain.labels import load_tables
from src.explain.narration import CitedFact, Narration
from src.explain.record import SECTION_NAMES, DecisionRecord
from src.explain.template import AUTO_PHRASES as T_AUTO
from src.explain.template import REVIEW_PHRASES as T_REVIEW
from src.explain.template import template_narration
from src.explain.validator import (AUTO_PHRASES, REVIEW_PHRASES, FailedCheck,
                                   ValidationResult, ValidatorContext,
                                   default_context, number_matches,
                                   validate)

TABLES = load_tables()
MASKED = ("I was charged an escrow fee I did not owe. The servicer won't answer my calls! "
          "My account number is XXXX and the fee keeps growing.")
SENT_1 = "I was charged an escrow fee I did not owe."
SENT_2 = "The servicer won't answer my calls!"
TECH = {"improved_model_1": "linear", "baseline_model_2": "nb_llr",
        "improved_model_2": "occlusion", "knn_embed": "neighbours",
        "stacker": "linear", "gate": "shap", "routing": "rules", "agentic": "rules"}


def _skipped(name: str) -> dict:
    return {"explainer": name, "status": "skipped", "method": "exact",
            "technique": TECH[name], "reason": "not run on this path"}


def _ok(name: str, drivers=(), **details) -> dict:
    return {"explainer": name, "status": "ok", "method": "exact", "technique": TECH[name],
            "drivers": list(drivers), "details": details}


def _other_team(team: str) -> str:
    teams = sorted(TABLES.teams)
    return teams[(teams.index(team) + 1) % len(teams)]


def real_record(issue_id: str, auto: bool = True, path: str = "platform",
                neighbour_issue: str | None = None) -> DecisionRecord:
    """A realistic record whose names and labels come from the label tables."""
    entry = TABLES.issues[issue_id]
    team = entry.team_id
    same_team = sorted(i for i, e in TABLES.issues.items() if e.team_id == team and i != issue_id)
    served = [issue_id] + same_team[:2]
    probs = [0.6213457, 0.2501, 0.0912][:len(served)]
    top = [{"rank": r, "issue_id": i, "label": TABLES.issues[i].label,
            "display_label": TABLES.issues[i].display_label, "prob": p}
           for r, (i, p) in enumerate(zip(served, probs), start=1)]
    other = _other_team(team)
    secs = {n: _skipped(n) for n in SECTION_NAMES}
    secs["improved_model_1"] = _ok(
        "improved_model_1",
        [{"kind": "ngram", "label": "word__escrow", "value": 0.31, "contribution": 0.8123},
         {"kind": "ngram", "label": "char__ fee", "value": 0.12, "contribution": -0.2049},
         {"kind": "ngram", "label": "word__servicer", "value": 0.2, "contribution": 0.1555},
         {"kind": "redacted", "contribution": 0.0421}],
        top_team=team, runner_up=other, raw_diff=1.2345, constant=-0.31,
        residual=0.0, n_nonzero=42,
        isotonic={team: {"raw_decision": 0.73, "isotonic_pre_norm": 0.81,
                         "calibrated_prob": 0.7712}},
        issue={"kind": "linear_multiclass", "method": "exact", "technique": "linear",
               "issue_id": issue_id, "compared_issue": served[1] if len(served) > 1 else None,
               "drivers": [{"kind": "ngram", "label": "word__fee", "value": 0.4,
                            "contribution": 0.402}]})
    secs["routing"] = _ok(
        "routing", score=0.7712, threshold=0.65 if auto else 0.9,
        threshold_kind="base", scenario="auto" if auto else "contested",
        model_teams={"improved_model_1": team, "baseline_model_2": other,
                     "improved_model_2": team, "knn_embed": team})
    if path == "streamlit":
        secs["routing"]["details"]["model_teams"] = {"improved_model_1": team}
        secs["routing"]["details"]["scenario"] = "primary_only"
    else:
        secs["baseline_model_2"] = _ok(
            "baseline_model_2",
            [{"kind": "ngram", "label": "escrow", "value": 1.0, "contribution": 2.75},
             {"kind": "ngram", "label": "calls", "value": 1.0, "contribution": -0.5}],
            chosen_team=team, runner_up=other, raw_diff=2.25, constant=0.0)
        secs["improved_model_2"] = _ok(
            "improved_model_2",
            [{"kind": "sentence", "text": SENT_1, "span": {"index": 0, "start": 0, "end": 42},
              "contribution": 0.3333},
             {"kind": "sentence", "text": SENT_2, "span": {"index": 1, "start": 43, "end": 78},
              "contribution": -0.0412}],
            chosen_team=team, k_eff=8, p_full=0.66, team_probs={team: 0.66, other: 0.2})
        nb_issue = neighbour_issue or sorted(
            i for i, e in TABLES.issues.items() if e.team_id == other)[0]
        secs["knn_embed"] = _ok(
            "knn_embed", k=10, n_index=5000, effective_k=10,
            team_distribution={team: 0.7, other: 0.3},
            neighbours=[{"rank": 1, "similarity": 0.912, "team_id": team,
                         "issue_id": issue_id, "complaint_id": "4821937"},
                        {"rank": 2, "similarity": 0.874, "team_id": other,
                         "issue_id": nb_issue, "complaint_id": "5550123"}])
        secs["stacker"] = _ok(
            "stacker",
            [{"kind": "model", "label": "improved_model_1", "contribution": 1.1},
             {"kind": "model", "label": "knn_embed", "contribution": 0.4}],
            case="stacker_agrees", chosen_team=team, comparison_team=other,
            ranking=[team, other], top_after=team)
        secs["gate"] = _ok(
            "gate",
            [{"kind": "feature_group", "label": "model:improved_model_1", "contribution": 0.9},
             {"kind": "feature_group", "label": "chosen_team", "contribution": -0.2},
             {"kind": "feature_group", "label": "complaint_length", "contribution": 0.05}],
            score=0.7712, threshold=0.65, unit="log-odds")
    scenario = "primary_only" if path == "streamlit" else ("auto" if auto else "contested")
    return DecisionRecord.model_validate({
        "request_id": "req-1", "created_at": "2025-06-01T12:00:00.000Z",
        "serving_path": path, "model_versions": {"improved_model_1": "sha256:abc"},
        "outcome": {"chosen_team": {"id": team, "name": TABLES.teams[team]},
                    "top_issues": top, "issue_order": "probability", "auto_route": auto,
                    "route": "auto" if auto else "review", "scenario": scenario},
        "sections": secs, "timings": {"total": 812}})


def ctx(include_text: bool = True, masked: str | None = MASKED,
        model: str = "llama-3.3-70b-versatile") -> ValidatorContext:
    return default_context(masked_text=masked, include_text=include_text,
                           narrator_model=model)


REC = real_record("T06_I009")
TEMPLATE = template_narration(REC)


def with_(n: Narration = TEMPLATE, consumer: str | None = None, agent: str | None = None,
          facts=None) -> Narration:
    return Narration(consumer_summary=n.consumer_summary if consumer is None else consumer,
                     agent_summary=n.agent_summary if agent is None else agent,
                     cited_facts=list(n.cited_facts) if facts is None else facts)


def checks(res: ValidationResult) -> set[str]:
    return {f.check for f in res.failed}


def reject(n: Narration, rec=REC, c: ValidatorContext | None = None) -> ValidationResult:
    res = validate(n, rec, c or ctx())
    assert not res.accepted and res.failed
    return res


AGENT_OK = "Outcome: the complaint was auto-routed to chosen team T06 (Mortgage Team)."


# ---------------------------------------------------------------------------
# Template acceptance (Req 13.12, 15.9; Property 27 on real label names)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("issue_id", sorted(TABLES.issues))
def test_template_accepted_for_every_real_issue(issue_id):
    for i, (auto, path) in enumerate(((True, "platform"), (False, "platform"),
                                      (True, "streamlit"), (False, "streamlit"))):
        rec = real_record(issue_id, auto=auto, path=path)
        n = template_narration(rec)
        res = validate(n, rec, ctx(include_text=bool(i % 2)))
        assert res == ValidationResult(True, ()), (issue_id, auto, path, res.failed)


def test_template_with_banned_word_in_rank1_label_is_accepted():
    rec = real_record("T01_I012")
    n = template_narration(rec)
    assert "credit score" in n.consumer_summary
    assert validate(n, rec, ctx()).accepted
    # outside the outcome values the same word is still banned
    res = reject(with_(n, consumer=n.consumer_summary + " Your score was high."), rec)
    assert FailedCheck("consumer_banned_word", "score", "banned word") in res.failed


def test_neighbour_label_equal_to_served_issue_is_allowed():
    # the neighbour's issue (another team) shares the served rank-1 display label
    rec = real_record("T06_I005", neighbour_issue="T07_I005")
    n = template_narration(rec)
    assert "Incorrect information on your report" in n.consumer_summary
    assert validate(n, rec, ctx()).accepted


def test_record_input_forms_are_equivalent():
    a = validate(TEMPLATE, REC, ctx())
    b = validate(TEMPLATE, REC.model_dump(mode="python"), ctx())
    assert a == b and a.accepted


def test_phrase_tables_match_template():
    assert AUTO_PHRASES == T_AUTO and REVIEW_PHRASES == T_REVIEW


# ---------------------------------------------------------------------------
# Mentions (Req 13.1, 13.2)
# ---------------------------------------------------------------------------

def test_team_id_and_name_not_in_record_fail():
    res = reject(with_(agent=AGENT_OK + " Also considered T05."))
    assert ("team_mention", "T05") in {(f.check, f.offending) for f in res.failed}
    res = reject(with_(agent=AGENT_OK + " Also the prepaid  CARD team."))
    assert ("team_mention", "prepaid card team") in {(f.check, f.offending) for f in res.failed}


def test_team_in_record_and_paraphrase_pass():
    # T07 is the runner-up (other team) of T06; the paraphrase is no mention
    agent = AGENT_OK + " Runner-up T07 (Student Loan Team); the credit bureau team was not used."
    assert validate(with_(agent=agent), REC, ctx()).accepted


def test_issue_id_and_label_not_in_record_fail():
    res = reject(with_(agent=AGENT_OK + " Issue T03_I002."))
    assert ("issue_mention", "T03_I002") in {(f.check, f.offending) for f in res.failed}
    res = reject(with_(agent=AGENT_OK + " Issue Credit card :: Closing your account."))
    assert "issue_mention" in checks(res)


def test_lower_case_team_id_is_not_a_mention():
    assert validate(with_(agent=AGENT_OK + " n-gram t05 seen."), REC, ctx()).accepted


# ---------------------------------------------------------------------------
# Numbers (Req 13.4, 13.5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token", ["0.8123", "0.812", "0.81", "0.8", "1", "62.1%", "62%",
                                   "62.13%", "-0.205", "42", "1.234"])
def test_faithful_numbers_pass(token):
    assert validate(with_(agent=f"{AGENT_OK} Value {token} noted."), REC, ctx()).accepted


@pytest.mark.parametrize("token", ["0.813", "0.5", "12", "63%", "-0.2049%", "3.5"])
def test_unfaithful_numbers_fail(token):
    res = reject(with_(agent=f"{AGENT_OK} Value {token} noted."))
    assert [f.offending for f in res.failed if f.check == "number"] == [token]


def test_exemptions_ids_quotes_and_small_counts():
    agent = AGENT_OK + ' Ranked 3 issues across 11 teams; issue T06_I009; "escrow".'
    assert validate(with_(agent=agent), REC, ctx()).accepted
    assert "number" in checks(reject(with_(agent=AGENT_OK + " There were 4.0 issues.")))


def test_huge_values_round_without_error():
    assert number_matches("1" + "0" * 300 + ".000", [1e300])
    assert number_matches("0.5000000", [0.5]) and not number_matches("0.6", [0.5])
    assert number_matches("87.8%", [0.878]) and number_matches("0.878", [0.878])
    assert not number_matches("150%", [1.5])            # percent only for p in [0, 1]


# ---------------------------------------------------------------------------
# Quoted phrases (Req 13.3)
# ---------------------------------------------------------------------------

def test_quoted_phrases_against_drivers_and_text():
    ok = AGENT_OK + ' Drivers "Escrow", " fee", \u201cthe servicer won\'t  answer\u201d.'
    assert validate(with_(agent=ok), REC, ctx()).accepted
    # clean_text form: "won't" becomes "will not"
    assert validate(with_(agent=AGENT_OK + ' Text "servicer will not answer".'),
                    REC, ctx()).accepted
    bad = with_(agent=AGENT_OK + ' Text "my neighbour complained".')
    assert [f.offending for f in reject(bad).failed] == ["my neighbour complained"]


def test_masked_text_quotes_need_include_text():
    n = with_(agent=AGENT_OK + ' Text "keeps growing".')
    assert validate(n, REC, ctx(include_text=True)).accepted
    assert checks(reject(n, c=ctx(include_text=False))) == {"quoted_phrase"}
    # a Driver sentence still matches without the text
    n = with_(agent=AGENT_OK + f' Sentence "{SENT_2}".')
    assert validate(n, REC, ctx(include_text=False, masked=None)).accepted


# ---------------------------------------------------------------------------
# Cited facts (Req 13.6)
# ---------------------------------------------------------------------------

def test_cited_facts():
    def run(path, value):
        return validate(with_(facts=[CitedFact(path=path, value=value)]), REC, ctx())
    assert run("outcome.top_issues.0.prob", 0.6213457).accepted        # canonical 0.621346
    assert run("sections.routing.details.score", 0.7712 + 9e-7).accepted
    assert run("outcome.auto_route", True).accepted
    assert run("sections.stacker.details.ranking", ["T06", "T07"]).accepted
    assert checks(run("outcome.top_issues.5.prob", 0.1)) == {"cited_fact_path"}
    assert checks(run("request_id", "req-1")) == {"cited_fact_path"}     # not in the hash view
    assert checks(run("sections.routing.details.score", 0.7712 + 2e-6)) == {"cited_fact_value"}
    assert checks(run("outcome.auto_route", 1)) == {"cited_fact_value"}
    assert checks(run("outcome.chosen_team.id", "T6")) == {"cited_fact_value"}
    assert checks(run("sections.routing.details.score", "0.7712")) == {"cited_fact_value"}
    assert checks(run("outcome.chosen_team", "T06")) == {"cited_fact_value"}


# ---------------------------------------------------------------------------
# Required statements (Req 13.7)
# ---------------------------------------------------------------------------

def test_required_team_and_outcome():
    res = reject(with_(agent="Outcome: the complaint was auto-routed."))
    assert ("required_team", "agent_summary") in {(f.check, f.offending) for f in res.failed}
    res = reject(with_(agent="Chosen team T06."))
    assert "required_outcome" in checks(res)
    res = reject(with_(agent=AGENT_OK + " It was not sent to human review."))
    assert ("opposite_outcome", "human review") in {(f.check, f.offending) for f in res.failed}
    # a quoted opposite phrase is a quoted phrase, not an outcome statement
    rec = real_record("T06_I009")
    c = ctx(masked=MASKED + " I asked for a human review.")
    assert validate(with_(agent=AGENT_OK + ' Text "a human review".'), rec, c).accepted


def test_review_outcome_needs_review_phrase():
    rec = real_record("T06_I009", auto=False)
    n = template_narration(rec)
    assert validate(n, rec, ctx()).accepted
    res = reject(with_(n, consumer="Your complaint was auto-routed to the Mortgage Team. "
                                    "The main issue we identified is: Trouble during "
                                    "payment process."), rec)
    assert {"required_outcome", "opposite_outcome"} <= checks(res)


# ---------------------------------------------------------------------------
# Consumer rules (Req 13.8, 15.1 to 15.3, 15.5)
# ---------------------------------------------------------------------------

BASE = TEMPLATE.consumer_summary


@pytest.mark.parametrize("extra,check,offending", [
    (" About 62% sure.", "consumer_percent", "%"),
    (" Value 0.771.", "consumer_number", "0.771"),
    (" The threshold was met.", "consumer_banned_word", "threshold"),
    (" Our confidence is high.", "consumer_banned_word", "confidence"),
    (" Probabilities agree.", "consumer_banned_word", "probabilities"),
    (" The Stacker agreed.", "consumer_model_name", "stacker"),
    (" Naive  Bayes agreed.", "consumer_model_name", "naive Bayes"),
    (" Written by llama-3.3-70b-versatile.", "consumer_model_name",
     "llama-3.3-70b-versatile"),
    (" Your complaint_length mattered.", "consumer_gate_feature", "complaint_length"),
    (" Similar to complaint 5550123.", "consumer_neighbour", "5550123"),
])
def test_consumer_bans(extra, check, offending):
    res = reject(with_(consumer=BASE + extra))
    assert (check, offending) in {(f.check, f.offending) for f in res.failed}


def test_consumer_neighbour_issue_label_not_served_fails():
    nb = sorted(i for i, e in TABLES.issues.items() if e.team_id == "T07")[0]
    label = TABLES.issues[nb].display_label
    res = reject(with_(consumer=BASE + f" Like {label}."))
    assert "consumer_neighbour" in checks(res)


def test_consumer_length_and_required_name_and_label():
    res = reject(with_(consumer=BASE + " word" * 120))
    assert "consumer_length" in checks(res)
    res = reject(with_(consumer="Your complaint was auto-routed to chosen team T06."))
    assert {"consumer_team_name", "consumer_issue_label"} <= checks(res)
    assert "required_team" not in checks(res)          # the id satisfies Req 13.7
    full_label = "Mortgage :: Trouble during payment process"
    assert validate(with_(consumer=f"Your complaint was auto-routed to the Mortgage Team: "
                                    f"{full_label}."), REC, ctx()).accepted


def test_agent_summary_may_use_consumer_banned_terms():
    agent = AGENT_OK + " The stacker score 0.771 >= threshold 0.650 (77.1%)."
    assert validate(with_(agent=agent), REC, ctx()).accepted


# ---------------------------------------------------------------------------
# Determinism, ordering and internal errors (Req 13.9 to 13.11)
# ---------------------------------------------------------------------------

def test_failed_checks_sorted_and_deterministic():
    n = with_(consumer=BASE + " 62% score T05 T05 0.999.", agent="nothing here 0.4444")
    a, b = validate(n, REC, ctx()), validate(n, REC, ctx())
    assert a == b and not a.accepted
    keys = [(f.check, f.offending) for f in a.failed]
    assert keys == sorted(keys) and len(set(a.failed)) == len(a.failed)


def test_internal_error_rejects():
    res = validate(TEMPLATE, object(), ctx())
    assert not res.accepted
    assert [f.check for f in res.failed] == ["internal_error"]
