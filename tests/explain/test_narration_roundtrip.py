"""Property test for the Narration printer/parser round trip (Req 12.1, 12.5 to 12.7)."""
from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from src.explain.narration import (AGENT_MAX, CONSUMER_MAX, FACTS_MAX, NARRATION_FIELDS,
                                   PATH_MAX, CitedFact, Narration, parse_narration,
                                   print_narration)

# Unicode text without lone surrogates, which are not encodable text.
_ALPHABET = st.characters(blacklist_categories=("Cs",))


def _bounded_text(max_len: int) -> st.SearchStrategy[str]:
    """Text whose stripped length is 1..max_len, with the bounds drawn often."""
    exact = st.sampled_from([1, max_len]).flatmap(
        lambda n: st.text(_ALPHABET, min_size=n, max_size=n))
    any_len = st.text(_ALPHABET, min_size=1, max_size=max_len)
    padded = st.tuples(st.sampled_from(["", " ", "\n\t "]), any_len,
                       st.sampled_from(["", "  ", "\n"])).map(lambda t: "".join(t))
    return st.one_of(exact, any_len, padded).filter(
        lambda s: 1 <= len(s.strip()) <= max_len)


_values = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(_ALPHABET, max_size=50),
)

_paths = st.one_of(
    st.text(_ALPHABET, min_size=1, max_size=PATH_MAX),
    st.text(_ALPHABET, min_size=PATH_MAX, max_size=PATH_MAX),
)

_facts = st.builds(CitedFact, path=_paths, value=_values)

_narrations = st.builds(
    Narration,
    consumer_summary=_bounded_text(CONSUMER_MAX),
    agent_summary=_bounded_text(AGENT_MAX),
    cited_facts=st.lists(_facts, min_size=0, max_size=FACTS_MAX),
)


# Feature: model-explainability, Property 21: Narration printer/parser round trip
@settings(max_examples=100, deadline=None)
@given(n=_narrations)
def test_narration_print_parse_round_trip(n: Narration) -> None:
    """**Validates: Requirements 12.1, 12.5, 12.6, 12.7**"""
    text = print_narration(n)

    # 12.5: exactly the three fields, in the fixed order.
    obj = json.loads(text)
    assert tuple(obj.keys()) == NARRATION_FIELDS
    assert [f["path"] for f in obj["cited_facts"]] == [f.path for f in n.cited_facts]

    # 12.1 / 12.7: the parser accepts the printed text and returns an equal Narration.
    result = parse_narration(text)
    assert result.ok, result.errors
    assert result.errors == []
    assert result.narration == n

    # 12.6 / 12.7: printing again (and printing an equal copy) is byte-identical.
    again = print_narration(result.narration)
    assert again.encode("utf-8") == text.encode("utf-8")
    copy = Narration(**n.model_dump())
    assert print_narration(copy).encode("utf-8") == text.encode("utf-8")
