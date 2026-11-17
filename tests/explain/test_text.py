"""Unit tests for src/explain/text.py (task 4.1; extended by task 4.4)."""
from __future__ import annotations

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.app.pii import mask
from src.explain.record import Driver, Span
from src.explain.text import (derivable_ngram, derivable_sentence,
                              fold_redacted, norm, segment, strip_prefix)
from tests.explain.fixtures import synthetic_bm2, synthetic_im1


def test_norm():
    assert norm("  Hello\t\n  WORLD  ") == "hello world"
    assert norm("") == ""


def test_segment_offsets_trimmed():
    t = "  First one.  Second?\nThird line\n\n  last!  "
    spans = segment(t)
    assert [t[s:e] for s, e in spans] == ["First one.", "Second?", "Third line", "last!"]
    for s, e in spans:
        assert t[s:e] == t[s:e].strip()


def test_segment_no_split_without_whitespace():
    assert segment("v1.2 is out. ok") == [(0, 12), (13, 15)]


@pytest.mark.parametrize("t", ["", "   ", "\n\n\t"])
def test_segment_empty(t):
    assert segment(t) == []


def test_derivable_ngram_im1_feature_union():
    vec = synthetic_im1().vectorizer
    text = mask("My card was charged twice. Call 555-123-4567.")
    names = vec.get_feature_names_out()
    word = [n for n in names if n.startswith("word__")]
    char = [n for n in names if n.startswith("char__")]
    assert word and char
    in_text = {n for n in names if derivable_ngram(n, text, vec)}
    assert any(n.startswith("char__") for n in in_text)
    # Every derivable name comes from the matching analyzer on the text.
    parts = dict(vec.transformer_list)
    for n in in_text:
        part, _, gram = n.partition("__")
        assert gram in set(parts[part].build_analyzer()(text))
    assert not derivable_ngram("word__zzzqqq", text, vec)
    assert not derivable_ngram("nope__card", text, vec)
    assert not derivable_ngram("word__5551234567", text, vec)


def test_derivable_ngram_plain_tfidf_bm2_case_insensitive():
    tfidf = synthetic_bm2().pipeline.named_steps["tfidf"]
    assert derivable_ngram("charged", "I was CHARGED twice", tfidf)
    assert not derivable_ngram("refund", "I was charged twice", tfidf)
    plain = TfidfVectorizer(lowercase=False).fit(["Charged twice"])
    assert derivable_ngram("charged", "Charged twice", plain)


def test_strip_prefix():
    assert strip_prefix("char__ ab") == " ab"
    assert strip_prefix("word__card fee") == "card fee"
    assert strip_prefix("card") == "card"


def test_derivable_sentence():
    m = "Sentence one. Sentence two."
    assert derivable_sentence("Sentence two.", m)
    assert not derivable_sentence("", m)
    assert not derivable_sentence("sentence two.", m)


def _ng(label, c):
    return Driver(kind="ngram", label=label, value=0.1, contribution=c)


def test_fold_redacted_pii_ngrams():
    masked = mask("email me at a.b@example.com or 4111 1111 1111 1111 card")
    drivers = [_ng("word__card", 0.5), _ng("word__example", -0.25),
               _ng("word__4111", 0.125), _ng("word__1234567", 0.25),
               Driver(kind="model", label="im1", contribution=1.0)]
    kept, red = fold_redacted(drivers, lambda t: strip_prefix(t) in norm(masked))
    assert [d.label for d in kept] == ["word__card", "im1"]
    assert red is not None and red.kind == "redacted"
    assert red.label is red.text is red.value is red.span is None
    assert red.contribution == pytest.approx(-0.25 + 0.125 + 0.25)


def test_fold_redacted_mask_check_even_if_derivable():
    kept, red = fold_redacted([_ng("word__1234567", 0.5)], lambda t: True)
    assert kept == [] and red.contribution == 0.5


def test_fold_redacted_absent_when_sum_zero():
    kept, red = fold_redacted([_ng("word__x1", 0.5), _ng("word__x2", -0.5),
                               _ng("word__ok", 1.0)],
                              lambda t: t == "word__ok")
    assert [d.label for d in kept] == ["word__ok"] and red is None
    assert fold_redacted([], lambda t: True) == ([], None)


def test_fold_redacted_sentences():
    m = "I paid. Call XXXX."
    s = [Driver(kind="sentence", text="I paid.", span=Span(index=0, start=0, end=7),
                contribution=0.3),
         Driver(kind="sentence", text="Call 555-123-4567.",
                span=Span(index=1, start=8, end=18), contribution=0.2)]
    kept, red = fold_redacted(s, lambda t: derivable_sentence(t, m))
    assert [d.text for d in kept] == ["I paid."] and red.contribution == 0.2
