"""Narration model, tolerant parser and canonical printer (Req 12.1 to 12.7).

``parse_narration`` takes a raw backend reply, finds the first top-level JSON
object in it, drops unknown keys and checks the Narration schema. It never
raises. On failure it returns ``ParseResult(ok=False, narration=None,
errors=[...])`` where each error is ``"<field path>: <kind>"`` with kind
``missing``, ``wrong_type`` or ``out_of_bounds`` (or a reply-level error such
as ``"<reply>: no_json_object"``).

``print_narration`` writes the three fields in a fixed order with compact
separators and ``ensure_ascii=False``. Numeric Cited_Fact values use the same
float normalisation as the Decision_Record (6 dp, no ``-0.0``). The models
apply that normalisation and strip the summaries on construction, so
print -> parse -> print is byte-identical and the parsed Narration equals the
original.
"""
from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import (BaseModel, ConfigDict, Field, StrictStr,
                      field_validator)

from src.explain.record import _normalise

CONSUMER_MAX = 1200
AGENT_MAX = 4000
PATH_MAX = 256
FACTS_MAX = 50
NARRATION_FIELDS: tuple[str, ...] = ("consumer_summary", "agent_summary", "cited_facts")

FallbackReason = Literal["backend_failure", "timeout", "no_api_key", "invalid_config",
                         "validator_rejection", "parse_error"]


def _is_scalar(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, str)):
        return True
    return isinstance(v, float) and math.isfinite(v)


def _valid_value(v: Any) -> bool:
    """JsonScalarOrList: a finite JSON scalar or a flat list of them."""
    if isinstance(v, list):
        return all(_is_scalar(x) for x in v)
    return _is_scalar(v)


def _norm_value(v: Any) -> Any:
    return _normalise(v, "", [])


def _strip_bounded(v: str, name: str, max_len: int) -> str:
    s = v.strip()
    if not 1 <= len(s) <= max_len:
        raise ValueError(f"{name} must be 1..{max_len} characters after stripping")
    return s


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CitedFact(_Frozen):
    path: StrictStr = Field(min_length=1, max_length=PATH_MAX)
    value: Any

    @field_validator("value", mode="before")
    @classmethod
    def _value(cls, v: Any) -> Any:
        if isinstance(v, tuple):
            v = list(v)
        if not _valid_value(v):
            raise ValueError("value must be a finite JSON scalar or a list of them")
        return _norm_value(v)


class _NarrationFields(_Frozen):
    consumer_summary: StrictStr
    agent_summary: StrictStr
    cited_facts: list[CitedFact] = Field(default_factory=list, max_length=FACTS_MAX)

    @field_validator("consumer_summary")
    @classmethod
    def _consumer(cls, v: str) -> str:
        return _strip_bounded(v, "consumer_summary", CONSUMER_MAX)

    @field_validator("agent_summary")
    @classmethod
    def _agent(cls, v: str) -> str:
        return _strip_bounded(v, "agent_summary", AGENT_MAX)


class Narration(_NarrationFields):
    """The LLM (or template) narration: two summaries and the Cited_Facts."""


class ValidationFailure(_Frozen):
    check: str
    offending: str
    reason: str


class ValidationSummary(_Frozen):
    accepted: bool
    failed: list[ValidationFailure] = Field(default_factory=list)


class Explanation(_NarrationFields):
    """A Narration plus its provenance (Data Models, Explanation)."""
    source: Literal["llm", "template"]
    fallback_reason: FallbackReason | None = None
    validation: ValidationSummary
    record_hash: str
    created_at: str

    def narration(self) -> Narration:
        return Narration(consumer_summary=self.consumer_summary,
                         agent_summary=self.agent_summary,
                         cited_facts=list(self.cited_facts))


class ParseResult(_Frozen):
    ok: bool
    narration: Narration | None = None
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

_DECODER = json.JSONDecoder()


def _first_object(raw: str) -> dict | None:
    """The first ``{`` from which ``raw_decode`` yields a dict (Req 12.2)."""
    i = raw.find("{")
    while i != -1:
        try:
            obj, _ = _DECODER.raw_decode(raw, i)
        except (ValueError, RecursionError):
            obj = None
        if isinstance(obj, dict):
            return obj
        i = raw.find("{", i + 1)
    return None


def _check_summary(obj: dict, name: str, max_len: int, errors: list[str]) -> None:
    if name not in obj:
        errors.append(f"{name}: missing")
    elif not isinstance(obj[name], str):
        errors.append(f"{name}: wrong_type")
    elif not 1 <= len(obj[name].strip()) <= max_len:
        errors.append(f"{name}: out_of_bounds")


def _check_facts(obj: dict, errors: list[str]) -> list[dict]:
    """Check ``cited_facts`` and return the items with unknown keys dropped."""
    if "cited_facts" not in obj:
        errors.append("cited_facts: missing")
        return []
    facts = obj["cited_facts"]
    if not isinstance(facts, list):
        errors.append("cited_facts: wrong_type")
        return []
    if len(facts) > FACTS_MAX:
        errors.append("cited_facts: out_of_bounds")
    cleaned: list[dict] = []
    for i, item in enumerate(facts):
        where = f"cited_facts.{i}"
        if not isinstance(item, dict):
            errors.append(f"{where}: wrong_type")
            continue
        if "path" not in item:
            errors.append(f"{where}.path: missing")
        elif not isinstance(item["path"], str):
            errors.append(f"{where}.path: wrong_type")
        elif not 1 <= len(item["path"]) <= PATH_MAX:
            errors.append(f"{where}.path: out_of_bounds")
        if "value" not in item:
            errors.append(f"{where}.value: missing")
        elif not _valid_value(item["value"]):
            errors.append(f"{where}.value: wrong_type")
        cleaned.append({k: item[k] for k in ("path", "value") if k in item})
    return cleaned


def _parse(raw: Any) -> ParseResult:
    if not isinstance(raw, str):
        return ParseResult(ok=False, errors=["<reply>: wrong_type"])
    if not raw.strip():
        return ParseResult(ok=False, errors=["<reply>: empty"])
    obj = _first_object(raw)
    if obj is None:
        return ParseResult(ok=False, errors=["<reply>: no_json_object"])
    errors: list[str] = []
    _check_summary(obj, "consumer_summary", CONSUMER_MAX, errors)
    _check_summary(obj, "agent_summary", AGENT_MAX, errors)
    facts = _check_facts(obj, errors)
    if errors:
        return ParseResult(ok=False, errors=errors)
    n = Narration(consumer_summary=obj["consumer_summary"],
                  agent_summary=obj["agent_summary"], cited_facts=facts)
    return ParseResult(ok=True, narration=n)


def parse_narration(raw: str) -> ParseResult:
    """Parse a backend reply into a Narration. Never raises (Req 12.1 to 12.4)."""
    try:
        return _parse(raw)
    except Exception as e:  # defensive: the parser is total
        return ParseResult(ok=False, errors=[f"<reply>: internal_error ({type(e).__name__})"])


# --------------------------------------------------------------------------
# Printer
# --------------------------------------------------------------------------

def print_narration(n: Narration) -> str:
    """Canonical Narration JSON text (Req 12.5 to 12.7)."""
    obj = {
        "consumer_summary": n.consumer_summary,
        "agent_summary": n.agent_summary,
        "cited_facts": [{"path": f.path, "value": _norm_value(f.value)}
                        for f in n.cited_facts],
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
