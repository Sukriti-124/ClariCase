"""Decision_Record schema (schema_version "1") and canonical serialisation.

The schema follows the Data Models section of design.md. Every model is
``extra="forbid"`` and ``frozen=True`` and rejects non-finite floats.

``Section`` is the common base of eight per-explainer section models. Each
subclass fixes ``explainer`` to one name and ``details`` to that explainer's
details model, so ``AnySection`` is a union discriminated by ``explainer``.
Build sections with ``make_section(...)`` (or the subclass directly); a bare
``Section(...)`` is rejected because its ``details`` would be untyped.

Canonical JSON, the parser and Record_Hash (task 2.3) live at the end of this
module.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal, Union

from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                      model_validator)

SCHEMA_VERSION = "1"

SectionName = Literal["improved_model_1", "baseline_model_2", "improved_model_2",
                      "knn_embed", "stacker", "gate", "routing", "agentic"]
SECTION_NAMES: tuple[str, ...] = ("improved_model_1", "baseline_model_2",
                                  "improved_model_2", "knn_embed", "stacker",
                                  "gate", "routing", "agentic")

Status = Literal["ok", "skipped", "error"]
Method = Literal["exact", "approximate"]
Technique = Literal["linear", "nb_llr", "occlusion", "neighbours", "shap",
                    "ablation", "constant", "rules"]
Scenario = Literal["auto", "assisted", "contested", "degraded", "primary_only"]

Prob = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegInt = Annotated[int, Field(ge=0)]

CALIBRATION_NOTE = ("isotonic mapping is monotonic non-decreasing per class; "
                    "final probability is normalised across all 11 teams")
AGENTIC_RATIONALE = "the agentic chooser provides no rationale"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Absent(_Model):
    """A value a requirement says to "record as absent", with the reason."""
    absent: Literal[True] = True
    reason: Annotated[str, Field(min_length=1)]


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------

class Span(_Model):
    index: NonNegInt
    start: NonNegInt
    end: NonNegInt

    @model_validator(mode="after")
    def _ordered(self) -> "Span":
        if self.end < self.start:
            raise ValueError("span end must be >= start")
        return self


class Driver(_Model):
    kind: Literal["ngram", "sentence", "model", "feature_group", "redacted"]
    label: str | None = None
    text: str | None = None
    value: float | None = None
    span: Span | None = None
    contribution: float

    @model_validator(mode="after")
    def _redacted_is_bare(self) -> "Driver":
        if self.kind == "redacted":
            filled = [f for f in ("label", "text", "value", "span")
                      if getattr(self, f) is not None]
            if filled:
                raise ValueError(f"redacted Driver must not carry {filled}")
            if self.contribution == 0:
                raise ValueError("redacted Driver must have a non-zero contribution")
        return self


def _check_drivers(drivers: list[Driver], where: str) -> None:
    n = sum(d.kind == "redacted" for d in drivers)
    if n > 1:
        raise ValueError(f"{where} holds {n} redacted Drivers; at most one is allowed")


# --------------------------------------------------------------------------
# Per-explainer details. Every field is optional so skipped/error sections can
# carry partial details (for example IM2 team_probs or the gate fields).
# --------------------------------------------------------------------------

class IsotonicTrace(_Model):
    raw_decision: float
    isotonic_pre_norm: Prob
    calibrated_prob: Prob


class IssueSubsection(_Model):
    kind: Literal["constant", "linear_multiclass", "linear_binary"]
    method: Method
    technique: Literal["linear", "constant"]
    issue_id: str
    compared_issue: str | None = None
    drivers: list[Driver] = Field(default_factory=list)
    constant: float | None = None
    raw_diff: float | None = None
    residual: float | None = None
    served_diff: float | None = None
    served_decision_deviation: float | None = None
    n_nonzero: NonNegInt | None = None
    redacted_contribution: float | None = None
    ranking_source: Literal["calibrated", "softmax(decision_function)"] | None = None

    @model_validator(mode="after")
    def _check(self) -> "IssueSubsection":
        _check_drivers(self.drivers, "issue sub-section")
        if (self.kind == "constant") != (self.technique == "constant"):
            raise ValueError("technique 'constant' is used exactly for kind 'constant'")
        if self.kind == "constant" and self.drivers:
            raise ValueError("a constant issue head has no Drivers")
        return self


class IM1Details(_Model):
    top_team: str | None = None
    runner_up: str | None = None
    raw_diff: float | None = None
    constant: float | None = None
    residual: float | None = None
    served_diff: float | None = None
    served_decision_deviation: float | None = None
    n_nonzero: NonNegInt | None = None
    redacted_contribution: float | None = None
    uncalibrated_source: str | None = None
    isotonic: dict[str, IsotonicTrace] | None = None
    calibration_note: str | None = None
    issue: IssueSubsection | None = None


class BM2Details(_Model):
    chosen_team: str | None = None
    runner_up: str | None = None
    raw_diff: float | None = None
    constant: float | None = None
    residual: float | None = None
    n_nonzero: NonNegInt | None = None
    redacted_contribution: float | None = None
    calibrated_prob_chosen: Prob | None = None
    uncalibrated_llr: float | None = None
    explains: Literal["uncalibrated_naive_bayes_score"] | None = None
    note: str | None = None


class IM2Details(_Model):
    chosen_team: str | None = None
    k: int | str | None = None              # raw configured value
    k_eff: Annotated[int, Field(ge=1, le=200)] | None = None
    k_default_applied: bool | None = None
    n_sentences: NonNegInt | None = None
    n_spans: NonNegInt | None = None
    spans_merged: bool | None = None
    n_encoded_texts: NonNegInt | None = None
    p_full: Prob | None = None
    team_probs: dict[str, Prob] | None = None


class Neighbour(_Model):
    rank: Annotated[int, Field(ge=1)]
    similarity: float
    team_id: str
    issue_id: str | None = None
    complaint_id: str


class KNNDetails(_Model):
    k: NonNegInt | None = None
    n_index: NonNegInt | None = None
    effective_k: NonNegInt | None = None
    team_distribution: dict[str, Prob] | None = None
    max_abs_diff_vs_served: float | None = None
    neighbours: list[Neighbour] = Field(default_factory=list)
    reason_ids: str | None = None


class StackerDetails(_Model):
    case: Literal["stacker_agrees", "stacker_disagrees"] | None = None
    chosen_team: str | None = None
    comparison_team: str | None = None
    intercept_diff: float | None = None
    logit_diff: float | None = None
    residual: float | None = None
    ranking: list[str] | None = None
    carrying_model: str | None = None
    label_shift_active: bool | None = None
    p_before: dict[str, Prob] | None = None
    p_after: dict[str, Prob] | None = None
    top_before: str | None = None
    top_after: str | None = None
    shift_changed_top: bool | None = None


class GateDetails(_Model):
    gate_kind: str | None = None
    score: Prob | None = None
    threshold: float | Absent | None = None
    score_ge_threshold: bool | None = None
    unit: Literal["log-odds", "change in gate score"] | None = None
    constant: float | None = None
    residual: float | None = None


class RoutingDetails(_Model):
    score: Prob | Absent | None = None
    threshold: float | Absent | None = None
    threshold_reason: str | None = None
    threshold_kind: Literal["base", "per-team"] | None = None
    threshold_team: str | None = None
    rule: str | None = None
    pre_rescue_auto: bool | None = None
    scenario: Scenario | None = None
    model_teams: dict[str, str | Absent] | None = None
    pre_llm_team: str | Absent | None = None
    label_shift_active: bool | None = None
    prior_n_seen: NonNegInt | Absent | None = None
    prior_min_rows: NonNegInt | None = None
    missing_models: list[str] | None = None
    fallback_threshold: float | Absent | None = None
    primary_score: Prob | Absent | None = None
    rescue_threshold: float | Absent | None = None
    llm_kept_team: bool | None = None
    pre_llm_auto: bool | None = None
    rescued: bool | None = None


TaskStatus = Literal["chosen", "failed", "not_asked"]


class AgenticDetails(_Model):
    team_candidates: list[str] | Absent | None = None
    team_choice: str | Absent | None = None
    rescued: bool | Absent | None = None
    issue_candidates: list[str] | Absent | None = None
    issue_choice: str | Absent | None = None
    team_status: TaskStatus | None = None
    team_status_reason: str | None = None
    issue_status: TaskStatus | None = None
    issue_status_reason: str | None = None
    team_changed: bool | None = None
    team_confidence_refers_to: str | None = None
    rationale: Literal["the agentic chooser provides no rationale"] = AGENTIC_RATIONALE


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

class Section(_Model):
    """Common fields and checks. Use a per-explainer subclass."""
    explainer: SectionName
    status: Status
    reason: str | None = None
    method: Method
    technique: Technique
    drivers: list[Driver] = Field(default_factory=list)
    details: Any = None

    @model_validator(mode="after")
    def _check_section(self) -> "Section":
        expected = SECTION_MODELS.get(self.explainer)
        if type(self) is not expected:
            raise ValueError(f"section {self.explainer!r} must be built as "
                             f"{expected.__name__} (use make_section)")
        has_reason = bool(self.reason and self.reason.strip())
        if has_reason != (self.status != "ok"):
            raise ValueError("reason must be non-empty exactly when status is not 'ok'")
        if self.status != "ok" and self.drivers:
            raise ValueError("drivers must be empty when status is not 'ok'")
        _check_drivers(self.drivers, f"section {self.explainer!r}")
        return self


class IM1Section(Section):
    explainer: Literal["improved_model_1"]
    details: IM1Details = Field(default_factory=IM1Details)

    @model_validator(mode="after")
    def _issue_drivers(self) -> "IM1Section":
        issue = self.details.issue
        if self.status != "ok" and issue is not None and issue.drivers:
            raise ValueError("issue sub-section drivers must be empty when status is not 'ok'")
        return self


class BM2Section(Section):
    explainer: Literal["baseline_model_2"]
    details: BM2Details = Field(default_factory=BM2Details)


class IM2Section(Section):
    explainer: Literal["improved_model_2"]
    details: IM2Details = Field(default_factory=IM2Details)


class KNNSection(Section):
    explainer: Literal["knn_embed"]
    details: KNNDetails = Field(default_factory=KNNDetails)


class StackerSection(Section):
    explainer: Literal["stacker"]
    details: StackerDetails = Field(default_factory=StackerDetails)


class GateSection(Section):
    explainer: Literal["gate"]
    details: GateDetails = Field(default_factory=GateDetails)


class RoutingSection(Section):
    explainer: Literal["routing"]
    details: RoutingDetails = Field(default_factory=RoutingDetails)


class AgenticSection(Section):
    explainer: Literal["agentic"]
    details: AgenticDetails = Field(default_factory=AgenticDetails)


SECTION_MODELS: dict[str, type[Section]] = {
    "improved_model_1": IM1Section, "baseline_model_2": BM2Section,
    "improved_model_2": IM2Section, "knn_embed": KNNSection,
    "stacker": StackerSection, "gate": GateSection,
    "routing": RoutingSection, "agentic": AgenticSection,
}
DETAILS_MODELS: dict[str, type[_Model]] = {
    name: cls.model_fields["details"].annotation for name, cls in SECTION_MODELS.items()
}

AnySection = Annotated[Union[IM1Section, BM2Section, IM2Section, KNNSection,
                             StackerSection, GateSection, RoutingSection,
                             AgenticSection], Field(discriminator="explainer")]


def make_section(explainer: str, **fields: Any) -> Section:
    """Build the section model for ``explainer``; ``details`` may be a dict."""
    if explainer not in SECTION_MODELS:
        raise ValueError(f"unknown section name {explainer!r}")
    return SECTION_MODELS[explainer](explainer=explainer, **fields)


# --------------------------------------------------------------------------
# Outcome and record
# --------------------------------------------------------------------------

class TeamRef(_Model):
    id: str
    name: str


class IssueRef(_Model):
    rank: Annotated[int, Field(ge=1)]
    issue_id: str
    label: str
    display_label: str
    prob: Prob


class Outcome(_Model):
    chosen_team: TeamRef
    top_issues: Annotated[list[IssueRef], Field(max_length=3)]
    issue_order: Literal["probability", "agentic"]
    auto_route: bool
    route: Literal["auto", "review"]
    scenario: Scenario

    @model_validator(mode="after")
    def _check(self) -> "Outcome":
        ranks = [i.rank for i in self.top_issues]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"top_issues ranks must be 1..n in order, got {ranks}")
        ids = [i.issue_id for i in self.top_issues]
        if len(set(ids)) != len(ids):
            raise ValueError("top_issues contain a duplicate issue_id")
        if self.issue_order == "probability":
            probs = [i.prob for i in self.top_issues]
            if any(a < b for a, b in zip(probs, probs[1:])):
                raise ValueError("top_issues must be in descending prob order "
                                 "while issue_order is 'probability'")
        if (self.route == "auto") != self.auto_route:
            raise ValueError("route must be 'auto' exactly when auto_route is true")
        return self


class DecisionRecord(_Model):
    schema_version: Literal["1"] = SCHEMA_VERSION
    request_id: str
    created_at: str
    serving_path: Literal["streamlit", "platform"]
    model_versions: dict[str, str]
    outcome: Outcome
    sections: dict[SectionName, AnySection]
    timings: dict[str, int] | None = None

    @model_validator(mode="after")
    def _check(self) -> "DecisionRecord":
        missing = [n for n in SECTION_NAMES if n not in self.sections]
        if missing:
            raise ValueError(f"sections is missing {missing}")
        for name, sec in self.sections.items():
            if sec.explainer != name:
                raise ValueError(f"sections[{name!r}] holds explainer {sec.explainer!r}")
        if self.serving_path == "streamlit" and self.outcome.scenario != "primary_only":
            raise ValueError("scenario must be 'primary_only' on the streamlit path")
        return self


# --------------------------------------------------------------------------
# Canonical JSON, parser and Record_Hash (Req 2.1-2.8, 16.5, 16.6)
# --------------------------------------------------------------------------

SUPPORTED_VERSIONS: frozenset[str] = frozenset({"1"})
HASH_EXCLUDED_FIELDS: tuple[str, ...] = ("request_id", "created_at", "timings")


class NonFiniteError(ValueError):
    """Raised by ``canonical_json`` when the input holds NaN or +/-inf."""

    def __init__(self, paths: list[str]):
        self.paths = list(paths)
        super().__init__("non-finite float at " + ", ".join(self.paths))


class RecordParseError(ValueError):
    """Raised by ``parse_record``; no DecisionRecord is returned.

    ``kind`` is ``"decode"`` (invalid UTF-8), ``"json"`` (malformed JSON,
    duplicate key or a NaN/Infinity token), ``"version"`` or ``"schema"``.
    ``offset`` is the byte offset of a decode or JSON failure. ``errors`` is a
    list of ``(dotted_path, message)``; ``paths`` holds just the paths.
    """

    def __init__(self, kind: str, message: str, *, offset: int | None = None,
                 errors: list[tuple[str, str]] | None = None):
        self.kind = kind
        self.offset = offset
        self.errors = list(errors or [])
        self.paths = [p for p, _ in self.errors]
        detail = message if offset is None else f"{message} at byte offset {offset}"
        if self.errors:
            detail += ": " + "; ".join(f"{p or '<root>'}: {m}" for p, m in self.errors)
        super().__init__(detail)


def _join(path: str, part: object) -> str:
    return f"{path}.{part}" if path else str(part)


def _normalise(obj: Any, path: str, bad: list[str]) -> Any:
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path or "<root>")
            return obj
        x = round(obj, 6)
        return 0.0 if x == 0 else x          # removes -0.0
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"dict key {k!r} at {path or '<root>'} is not a str")
            out[k] = _normalise(v, _join(path, k), bad)
        return out
    if isinstance(obj, (list, tuple)):
        return [_normalise(v, _join(path, i), bad) for i, v in enumerate(obj)]
    if isinstance(obj, BaseModel):
        return _normalise(obj.model_dump(mode="python"), path, bad)
    raise TypeError(f"value of type {type(obj).__name__} at {path or '<root>'} "
                    "is not JSON-serialisable")


def canonical_json(record: DecisionRecord | dict) -> str:
    """Canonical JSON text (Req 2.1). Encode with UTF-8 to get the bytes.

    Raises ``NonFiniteError`` naming every NaN/inf path, before any output.
    """
    bad: list[str] = []
    obj = _normalise(record, "", bad)
    if bad:
        raise NonFiniteError(bad)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


class _DuplicateKey(Exception):
    pass


class _NonFiniteToken(Exception):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey(k)
        out[k] = v
    return out


def _reject_constant(name: str) -> Any:
    raise _NonFiniteToken(name)


def _find_outside_strings(text: str, token: str) -> int | None:
    """Character index of the first ``token`` that is not inside a JSON string."""
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif text.startswith(token, i):
            return i
    return None


def _byte_offset(text: str, pos: int) -> int:
    return len(text[:pos].encode("utf-8", errors="surrogatepass"))


def _data_path(obj: Any, loc: tuple, missing: bool) -> str:
    """Map a pydantic ``loc`` to a dotted path through the input data.

    Elements that are not keys of the data (discriminator tags, union member
    names) are dropped. A missing field keeps its final name.
    """
    parts: list[str] = []
    cur = obj
    for i, el in enumerate(loc):
        if isinstance(cur, dict) and isinstance(el, str) and el in cur:
            parts.append(el)
            cur = cur[el]
        elif isinstance(cur, list) and isinstance(el, int) and 0 <= el < len(cur):
            parts.append(str(el))
            cur = cur[el]
        elif missing and i == len(loc) - 1:
            parts.append(str(el))
    return ".".join(parts)


def parse_record(data: bytes | str) -> DecisionRecord:
    """Parse Decision_Record JSON (Req 2.2, 2.3, 2.7); raises RecordParseError."""
    if isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("utf-8")
        except UnicodeDecodeError as e:
            raise RecordParseError("decode", "invalid UTF-8", offset=e.start) from None
    else:
        text = data
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicates,
                         parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise RecordParseError("json", f"malformed JSON ({e.msg})",
                               offset=_byte_offset(text, e.pos)) from None
    except _DuplicateKey as e:
        raise RecordParseError("json", f"duplicate key {e.args[0]!r}") from None
    except _NonFiniteToken as e:
        pos = _find_outside_strings(text, e.args[0])
        raise RecordParseError(
            "json", f"non-finite number {e.args[0]!r} is not valid JSON",
            offset=None if pos is None else _byte_offset(text, pos)) from None
    if not isinstance(obj, dict):
        raise RecordParseError("schema", "invalid Decision_Record",
                               errors=[("", "record must be a JSON object")])
    if "schema_version" in obj and obj["schema_version"] not in SUPPORTED_VERSIONS:
        raise RecordParseError(
            "version", "unsupported schema version",
            errors=[("schema_version", f"{obj['schema_version']!r} is not one of "
                     f"{sorted(SUPPORTED_VERSIONS)}")])
    # Strict mode; pydantic's strict float accepts JSON integers (and rejects
    # bools and strings), which is the one lax coercion the design allows.
    try:
        return DecisionRecord.model_validate(obj, strict=True)
    except ValidationError as e:
        errors: list[tuple[str, str]] = []
        for err in e.errors():
            p = _data_path(obj, tuple(err["loc"]), err["type"] == "missing")
            item = (p, err["msg"])
            if item not in errors:
                errors.append(item)
        raise RecordParseError("schema", "invalid Decision_Record",
                               errors=errors) from None


def hash_view(record: DecisionRecord | dict) -> dict:
    """The record as a dict without request_id, created_at and timings."""
    d = record.model_dump(mode="python") if isinstance(record, BaseModel) else dict(record)
    for k in HASH_EXCLUDED_FIELDS:
        d.pop(k, None)
    return d


def record_hash(record: DecisionRecord | dict, cfg: Any,
                masked_text: str | None) -> str:
    """Record_Hash: 64 lowercase hex chars (Req 2.5, 2.8, 16.5, 16.6).

    ``cfg`` needs ``hash_settings()`` and ``include_text``. The masked text is
    hashed only when ``cfg.include_text`` is true.
    """
    payload = {"record": hash_view(record), "narrator": cfg.hash_settings(),
               "masked_text": masked_text if cfg.include_text else None}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
