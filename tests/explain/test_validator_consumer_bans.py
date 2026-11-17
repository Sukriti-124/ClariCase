"""Property 29: the validator rejects every banned element in a consumer summary.

**Validates: Requirements 13.8, 15.1, 15.2, 15.3, 15.5**
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.explain.template import template_narration
from src.explain.validator import (CONSUMER_WORD_MAX,
                                   MODEL_NAMES, validate)
from src.explain.validator import _Facts
from tests.explain.test_validator_unit import ctx, real_record, with_

REC = real_record("T06_I009")
BASE = template_narration(REC).consumer_summary


KEYED = sorted({str(v) for v in _Facts(REC, ctx()).banned_numbers.values
                if "." in str(v) and v not in (0, 1)})

PERCENT = st.integers(1, 99).map(lambda n: f"About {n}% sure.")
BANNED_WORD = st.sampled_from(
    ["probability", "probabilities", "score", "scores", "threshold", "thresholds",
     "confidence"]).map(lambda w: f"The {w} was noted.")
MODEL = st.sampled_from(sorted(MODEL_NAMES) + ["llama-3.3-70b-versatile"]).map(
    lambda m: f"Produced by {m}.")
GATE = st.just("Your complaint_length mattered.")
LENGTH = st.integers(CONSUMER_WORD_MAX, CONSUMER_WORD_MAX + 40).map(lambda n: " word" * n)
KEYED_NUM = (st.sampled_from(KEYED).map(lambda v: f"Value {v}.")
             if KEYED else st.just("Value 0.771."))

# 5550123 is the neighbour complaint id in the record under test
NEIGHBOUR = st.sampled_from(["complaint", "case", "ticket"]).map(
    lambda w: f"Similar to {w} 5550123.")

BANS = st.one_of(PERCENT, BANNED_WORD, MODEL, GATE, NEIGHBOUR, LENGTH, KEYED_NUM)


# Feature: model-explainability, Property 29: Consumer-summary bans are enforced
@settings(max_examples=100, deadline=None)
@given(BANS)
def test_consumer_bans_rejected(extra):
    res = validate(with_(consumer=BASE + " " + extra.strip()), REC, ctx())
    assert not res.accepted
    assert any(f.check.startswith("consumer_") for f in res.failed)



