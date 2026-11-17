"""Property 4: canonical serialisation round trip (Req 2.1, 2.2, 2.4)."""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from hypothesis import given, settings
from pydantic import BaseModel

from src.explain.record import DecisionRecord, canonical_json, parse_record
from tests.explain.strategies import st_record

_WS = " \t\n\r"


def _scan(text: str) -> list[str]:
    """Check token layout; return the problems found.

    Outside strings there must be no whitespace. Inside strings, a ``\\uXXXX``
    escape may only encode a control character (< U+0020): every other
    character, including non-ASCII, must be written literally.
    """
    problems: list[str] = []
    in_str = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                nxt = text[i + 1]
                if nxt == "u":
                    code = int(text[i + 2:i + 6], 16)
                    if code >= 0x20:
                        problems.append(f"escape \\u{code:04x} at {i}")
                    i += 6
                    continue
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in _WS:
            problems.append(f"whitespace {ch!r} at {i}")
        i += 1
    return problems


def _load_checked(text: str) -> tuple[Any, list[str], list[str]]:
    """``json.loads`` that records key-order problems and float tokens."""
    order_problems: list[str] = []
    float_tokens: list[str] = []

    def pairs(items: list[tuple[str, Any]]) -> dict:
        keys = [k for k, _ in items]
        # Python str comparison is by code point.
        if keys != sorted(keys):
            order_problems.append(f"unsorted keys {keys}")
        return dict(items)

    def flt(tok: str) -> float:
        float_tokens.append(tok)
        return float(tok)

    obj = json.loads(text, object_pairs_hook=pairs, parse_float=flt)
    return obj, order_problems, float_tokens


def _assert_equal_values(model: Any, js: Any, path: str = "") -> None:
    """The parsed object's field values equal the JSON values (Req 2.2)."""
    where = path or "<root>"
    if isinstance(model, BaseModel):
        assert isinstance(js, dict), where
        assert set(js) == set(type(model).model_fields), where
        for k, v in js.items():
            _assert_equal_values(getattr(model, k), v, f"{path}.{k}" if path else k)
    elif isinstance(model, dict):
        assert isinstance(js, dict) and set(js) == set(model), where
        for k, v in js.items():
            _assert_equal_values(model[k], v, f"{path}.{k}" if path else k)
    elif isinstance(model, (list, tuple)):
        assert isinstance(js, list) and len(js) == len(model), where
        for i, (m, v) in enumerate(zip(model, js)):
            _assert_equal_values(m, v, f"{path}.{i}" if path else str(i))
    else:
        assert type(model) is type(js), f"{where}: {type(model)} vs {type(js)}"
        assert model == js, f"{where}: {model!r} != {js!r}"


# Feature: model-explainability, Property 4: Canonical serialisation round trip
@settings(max_examples=100, deadline=None)
@given(st_record)
def test_canonical_serialisation_round_trip(record: DecisionRecord) -> None:
    text = canonical_json(record)
    data = text.encode("utf-8")

    # Req 2.1: sorted keys at every level, no inter-token whitespace,
    # literal UTF-8 for non-ASCII characters.
    assert _scan(text) == []
    obj, order_problems, float_tokens = _load_checked(text)
    assert order_problems == []

    # Req 2.1: at most 6 decimals per float, no negative zero.
    for tok in float_tokens:
        assert Decimal(tok).as_tuple().exponent >= -6, tok
        value = float(tok)
        assert value == round(value, 6), tok
        assert not tok.startswith("-0") or value != 0.0, tok
        assert tok not in ("-0.0", "-0"), tok

    # Req 2.4: serialise -> parse -> serialise is byte-identical.
    parsed = parse_record(data)
    assert isinstance(parsed, DecisionRecord)
    assert canonical_json(parsed.model_dump()).encode("utf-8") == data
    assert canonical_json(parsed).encode("utf-8") == data
    assert canonical_json(parse_record(text)) == text

    # Req 2.2: the parsed record's field values equal the JSON values.
    _assert_equal_values(parsed, obj)
