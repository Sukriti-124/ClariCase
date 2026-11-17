"""Record_Builder: one schema-validated Decision_Record per request (design §10).

``build_record`` fills every section the caller did not pass with the
serving-path skip (Req 1.2, 1.3, 1.9), checks the routing invariant (a broken
routing section becomes an ``error`` section, design §8), validates through
the Pydantic schema (Req 1.1, 1.2, 1.12), and then runs two checks:

* ``check_outcome_matches(record, served)``: the record's decision equals the
  served one and every team/issue value resolves via ``src.explain.labels``
  (Req 1.5, 1.7, 1.11).
* ``check_no_raw_text(record, raw_text, masked_text)``: no raw text or PII
  span in any section string, and every text Driver is derivable from the
  Masked_Text (Req 1.6, 1.12).

Any failure raises ``BuildError``; no partial record is ever returned. The
caller turns it into a failure entry with ``build_failure`` (Req 1.10).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterator, Literal, Mapping, TypedDict

from pydantic import ValidationError

from src.app.pii import mask, scan
from src.explain import labels as _labels
from src.explain.explainers import error_section, skipped_section
from src.explain.explainers.agentic import SKIP_NOT_CONSULTED, SKIP_STREAMLIT
from src.explain.explainers.routing import routing_rule_holds
from src.explain.record import (SECTION_NAMES, DecisionRecord,
                                Driver, NonFiniteError, Outcome, Section,
                                canonical_json, make_section)
from src.explain.text import (derivable_ngram, derivable_sentence, driver_text,
                              norm, strip_prefix)

__all__ = ["BuildError", "BuildFailure", "RoutingRuleError", "STREAMLIT_RUN",
           "build_record", "build_failure", "check_outcome_matches",
           "check_no_raw_text", "skip_reason"]

ServingPath = Literal["streamlit", "platform"]

# Sections the Streamlit path runs; the others are always skipped (Req 1.3).
STREAMLIT_RUN: frozenset[str] = frozenset({"improved_model_1", "routing"})
NGRAM_SECTIONS: tuple[str, ...] = ("improved_model_1", "baseline_model_2")


class BuildError(Exception):
    """The Decision_Record could not be built. ``error_type`` names the cause:
    ``invalid_input``, ``schema``, ``non_finite``, ``outcome_mismatch``,
    ``label_mismatch`` or ``raw_text``."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        super().__init__(f"{error_type}: {message}")


class RoutingRuleError(ValueError):
    """The routing section breaks the invariant of Req 9.7."""


class BuildFailure(TypedDict):
    request_id: str
    error_type: str
    at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_failure(request_id: str, exc: BaseException) -> BuildFailure:
    """The failure entry ``{request_id, error_type, at}`` (Req 1.10)."""
    error_type = (exc.error_type if isinstance(exc, BuildError)
                  else type(exc).__name__)
    return {"request_id": str(request_id), "error_type": error_type, "at": _now()}


def skip_reason(name: str, serving_path: str) -> str:
    """The reason of the skip that fills a section the caller did not pass."""
    if serving_path == "platform" and name == "agentic":
        return SKIP_NOT_CONSULTED                      # Req 1.9
    return SKIP_STREAMLIT                              # Req 1.3


# --------------------------------------------------------------------------
# Section assembly
# --------------------------------------------------------------------------

def _as_section(name: str, sec: Any) -> Section:
    if isinstance(sec, Section):
        return sec
    if isinstance(sec, Mapping):
        fields = {k: v for k, v in sec.items() if k != "explainer"}
        return make_section(name, **fields)
    raise TypeError(f"section {name!r} is a {type(sec).__name__}, not a Section")


def _assemble(serving_path: str, sections: Mapping[str, Any],
              final_auto: bool) -> dict[str, Section]:
    unknown = sorted(set(sections) - set(SECTION_NAMES))
    if unknown:
        raise BuildError("invalid_input", f"unknown section names {unknown}")
    out: dict[str, Section] = {}
    for name in SECTION_NAMES:
        if name not in sections or sections[name] is None:
            out[name] = skipped_section(name, skip_reason(name, serving_path))
            continue
        try:
            sec = _as_section(name, sections[name])
        except (ValidationError, ValueError, TypeError) as e:
            raise BuildError("schema", f"section {name!r} is invalid "
                             f"({type(e).__name__})") from e
        if sec.explainer != name:
            raise BuildError("schema", f"sections[{name!r}] holds explainer "
                             f"{sec.explainer!r}")
        if (serving_path == "streamlit" and name not in STREAMLIT_RUN
                and sec.status != "skipped"):
            raise BuildError("invalid_input", f"section {name!r} must be skipped "
                             "on the streamlit path")
        out[name] = sec
    routing = out["routing"]
    if not routing_rule_holds(routing, final_auto=final_auto):
        out["routing"] = error_section("routing", RoutingRuleError(
            "routing section does not satisfy the auto-route rule"))
    return out


# --------------------------------------------------------------------------
# Outcome check (Req 1.5, 1.7, 1.11)
# --------------------------------------------------------------------------

def _decision_view(o: Outcome) -> tuple:
    return (o.chosen_team.id,
            [(i.issue_id, i.prob) for i in o.top_issues],
            o.issue_order, o.auto_route, o.route, o.scenario)


def check_outcome_matches(record: DecisionRecord, served: Outcome | Mapping,
                          tables: _labels.LabelTables | None = None) -> None:
    """Raise ``BuildError`` unless the record's decision equals ``served``.

    Compared: Chosen_Team id, top-issue ids, probabilities and order,
    Issue_Order, auto-route flag, route and scenario. Every team and issue
    value must also match the label tables (Req 1.5), and an ``agentic``
    Issue_Order needs the agentic issue pick first (Req 1.1).
    """
    if not isinstance(served, Outcome):
        try:
            served = Outcome.model_validate(served)
        except ValidationError as e:
            raise BuildError("invalid_input", "served outcome is invalid") from e
    o = record.outcome
    names = ("chosen_team", "top_issues", "issue_order", "auto_route", "route",
             "scenario")
    diff = [n for n, a, b in zip(names, _decision_view(o), _decision_view(served))
            if a != b]
    if diff:
        raise BuildError("outcome_mismatch", f"record differs from the served "
                         f"output in {diff}")
    t = tables or _labels.load_tables()
    team = o.chosen_team
    if t.teams.get(team.id) != team.name:
        raise BuildError("label_mismatch", f"team {team.id!r} does not match "
                         "TEAM_MAPPING")
    for i in o.top_issues:
        e = t.issues.get(i.issue_id)
        if e is None or (e.label, e.display_label) != (i.label, i.display_label):
            raise BuildError("label_mismatch", f"issue {i.issue_id!r} does not "
                             "match the issue label table")
    if o.issue_order == "agentic":
        ag = record.sections["agentic"]
        pick = ag.details.issue_choice if ag.status == "ok" else None
        if not o.top_issues or not isinstance(pick, str) \
                or o.top_issues[0].issue_id != pick:
            raise BuildError("outcome_mismatch", "issue_order is 'agentic' but the "
                             "agentic issue pick is not first")


# --------------------------------------------------------------------------
# Raw-text check (Req 1.6, 1.12)
# --------------------------------------------------------------------------

def _strings(obj: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(k, str):
                yield p + "<key>", k
            yield from _strings(v, p)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}.{i}")


def _text_drivers(record: DecisionRecord) -> Iterator[tuple[str, str, Driver]]:
    for name, sec in record.sections.items():
        for i, d in enumerate(sec.drivers):
            yield name, f"sections.{name}.drivers.{i}", d
        if name == "improved_model_1" and sec.details.issue is not None:
            for i, d in enumerate(sec.details.issue.drivers):
                yield name, f"sections.{name}.details.issue.drivers.{i}", d


def _clean(s: str) -> str:
    """``src.data.prepare.clean_text`` on one string, lower-cased."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()


def _ngram_ok(gram: str, name: str, masked_text: str,
              vectorizers: Mapping[str, Any] | None) -> bool:
    vec = (vectorizers or {}).get(name)
    if vec is not None:
        return derivable_ngram(gram, masked_text, vec)
    # No vectorizer: the bare n-gram must occur in the cleaned Masked_Text.
    bare = _clean(strip_prefix(gram))
    return bool(bare) and bare in _clean(masked_text)


def _mask_invariant(s: str) -> bool:
    bare = strip_prefix(s)
    return mask(s) == s and mask(bare) == bare


def check_no_raw_text(record: DecisionRecord, raw_text: str | None,
                      masked_text: str | None,
                      vectorizers: Mapping[str, Any] | None = None) -> None:
    """Raise ``BuildError("raw_text", ...)`` when a section leaks text.

    * No string in ``sections`` holds a PII span of ``raw_text`` and, when the
      raw text differs from the Masked_Text, none holds the raw text itself.
    * Every sentence Driver is a Masked_Text substring; every n-gram Driver is
      derivable via ``derivable_ngram`` with the section's vectorizer
      (``vectorizers[name]``), or, without one, its bare n-gram occurs in the
      cleaned, lower-cased Masked_Text. Both satisfy ``mask(t) == t``.

    Messages name paths only, never the offending text.
    """
    raw = raw_text or ""
    masked = masked_text or ""
    spans = {f.text for f in scan(raw) if f.text.strip()} if raw else set()
    raw_norm = norm(raw) if raw and raw != masked else ""
    sections = record.model_dump(mode="python")["sections"]
    for path, s in _strings(sections, "sections"):
        if any(sp in s for sp in spans):
            raise BuildError("raw_text", f"PII span found at {path}")
        if raw_norm and raw_norm in norm(s):
            raise BuildError("raw_text", f"raw complaint text found at {path}")
    for name, path, d in _text_drivers(record):
        if d.kind not in ("ngram", "sentence"):
            continue
        t = driver_text(d)
        if not t or not _mask_invariant(t):
            raise BuildError("raw_text", f"text Driver at {path} is not masked")
        ok = (derivable_sentence(t, masked) if d.kind == "sentence"
              else _ngram_ok(t, name, masked, vectorizers))
        if not ok:
            raise BuildError("raw_text", f"text Driver at {path} is not "
                             "derivable from the Masked_Text")


# --------------------------------------------------------------------------
# build_record
# --------------------------------------------------------------------------

def build_record(*, request_id: str, serving_path: ServingPath,
                 model_versions: Mapping[str, str], outcome: Outcome | Mapping,
                 sections: Mapping[str, Section | Mapping | None],
                 created_at: str | None = None,
                 served: Outcome | Mapping | None = None,
                 raw_text: str | None = None, masked_text: str | None = None,
                 vectorizers: Mapping[str, Any] | None = None,
                 timings: Mapping[str, int] | None = None,
                 tables: _labels.LabelTables | None = None) -> DecisionRecord:
    """Assemble and check one Decision_Record; raise ``BuildError`` on failure.

    ``served`` is the served decision (defaults to ``outcome``, which then
    only gets the label checks). ``masked_text`` is ``pii.mask(raw_text)``
    when omitted. ``vectorizers`` maps ``improved_model_1`` /
    ``baseline_model_2`` to the vectorizer their n-grams came from.
    """
    if serving_path not in ("streamlit", "platform"):
        raise BuildError("invalid_input", f"unknown serving path {serving_path!r}")
    try:
        out = outcome if isinstance(outcome, Outcome) else Outcome.model_validate(outcome)
    except ValidationError as e:
        raise BuildError("schema", "outcome is invalid") from e
    secs = _assemble(serving_path, sections or {}, out.auto_route)
    try:
        record = DecisionRecord(
            request_id=request_id, created_at=created_at or _now(),
            serving_path=serving_path, model_versions=dict(model_versions),
            outcome=out, sections=secs,
            timings=dict(timings) if timings is not None else None)
    except ValidationError as e:
        raise BuildError("schema", f"record is invalid ({e.error_count()} "
                         "errors)") from e
    try:
        canonical_json(record)
    except NonFiniteError as e:
        raise BuildError("non_finite", str(e)) from e
    check_outcome_matches(record, out if served is None else served, tables)
    if masked_text is None and raw_text is not None:
        masked_text = mask(raw_text)
    check_no_raw_text(record, raw_text, masked_text, vectorizers)
    return record
