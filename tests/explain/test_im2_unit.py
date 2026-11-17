"""Unit tests for the IM2_Explainer (sentence occlusion, Req 5)."""
from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.probs import TEAMS, reorder
from src.explain.explainers.im2 import explain_im2, merge_spans, resolve_k
from tests.explain.fixtures import FakeEncoder, synthetic_im2

M = synthetic_im2()


def _p(text: str) -> np.ndarray:
    X = FakeEncoder.embed(text)[None, :]
    return reorder(M.classifier.predict_proba(X), M.team_classes, TEAMS)[0]


def _run(text, *, enabled=True, k=None, masked_served=False, encoder=None):
    enc = encoder or FakeEncoder()
    P = _p(text)
    chosen = TEAMS[int(P.argmax())]
    served = dict(zip(TEAMS, P.tolist()))
    emb = FakeEncoder.embed(text) if masked_served else None
    sec = explain_im2(M.classifier, M.team_classes, enc, text, chosen, served,
                      enabled, k, masked_served, emb)
    return sec, enc, chosen, served


TEXT = ("My mortgage escrow was wrong. The bank said foreclosure! "
        "I called again today?")


@pytest.mark.parametrize("k,expected", [
    (None, (20, True)), (5, (5, False)), ("7", (7, False)), (" 200 ", (200, False)),
    (0, (20, True)), (201, (20, True)), ("abc", (20, True)), ("1.5", (20, True)),
    (True, (20, True)), (3.0, (20, True)),
])
def test_resolve_k(k, expected):
    assert resolve_k(k) == expected


def test_merge_spans_divmod_sizes():
    sents = [(i * 10, i * 10 + 5) for i in range(7)]
    spans = merge_spans(sents, 3)          # sizes 3, 2, 2
    assert spans == [(0, 25), (30, 45), (50, 65)]
    assert merge_spans(sents, 10) == sents


def test_drivers_follow_definition_and_order():
    sec, enc, chosen, _ = _run(TEXT)
    assert (sec.status, sec.method, sec.technique) == ("ok", "approximate", "occlusion")
    assert enc.n_calls == 1 and len(enc.batches[0]) == 4      # baseline + 3 variants
    assert sec.details.n_encoded_texts == 4
    assert sec.details.n_sentences == 3 and sec.details.spans_merged is False
    p_full = _p(TEXT)[TEAMS.index(chosen)]
    starts = [d.span.start for d in sec.drivers]
    assert starts == sorted(starts) and [d.span.index for d in sec.drivers] == [0, 1, 2]
    for d in sec.drivers:
        assert d.kind == "sentence" and d.text == TEXT[d.span.start:d.span.end]
        variant = (TEXT[:d.span.start] + " " + TEXT[d.span.end:]).strip()
        assert d.contribution == pytest.approx(
            p_full - _p(variant)[TEAMS.index(chosen)], abs=1e-9)


def test_served_masked_embedding_skips_baseline():
    sec, enc, _, _ = _run(TEXT, masked_served=True)
    assert enc.n_calls == 1 and len(enc.batches[0]) == 3
    assert sec.details.n_encoded_texts == 3


def test_single_sentence_against_empty_input():
    text = "My mortgage escrow payment is incorrect."
    sec, _, chosen, _ = _run(text)
    assert len(sec.drivers) == 1
    j = TEAMS.index(chosen)
    assert sec.drivers[0].contribution == pytest.approx(_p(text)[j] - _p("")[j], abs=1e-6)


def test_merged_spans_within_budget():
    text = " ".join(f"Sentence number {i} about escrow." for i in range(9))
    sec, enc, _, _ = _run(text, k="4")
    assert len(enc.batches[0]) == 5                       # baseline + 4 spans
    assert sec.details.spans_merged is True and sec.details.n_sentences == 9
    assert sec.details.k_eff == 4 and sec.details.k_default_applied is False
    assert len(sec.drivers) == 4


def test_disabled_skips_without_encoding():
    sec, enc, _, served = _run(TEXT, enabled=False)
    assert sec.status == "skipped" and sec.reason == "occlusion disabled"
    assert sec.drivers == [] and enc.n_calls == 0
    assert sec.details.team_probs == pytest.approx(served)
    assert (sec.method, sec.technique) == ("approximate", "occlusion")


def test_no_sentences_skips():
    sec, enc, _, _ = _run("   \n  ")
    assert sec.status == "skipped" and sec.reason == "no sentences"
    assert sec.drivers == [] and enc.n_calls == 0


def test_encoder_failure_is_error_with_team_probs():
    sec, _, _, served = _run(TEXT, encoder=FakeEncoder(raise_on_call=True))
    assert sec.status == "error" and "encoder failure" in sec.reason
    assert sec.drivers == []
    assert sec.details.team_probs == pytest.approx(served)
