"""Narrator: turns a Decision_Record into a Narration through an LLM backend.

This module currently holds the prompt builder (design §14, Req 11.4 to 11.7,
11.10). The prompt is plain text in a fixed order:

1. role and rules
2. the Narration JSON schema
3. ``<<<DECISION_RECORD`` block with ``canonical_json(hash_view(record))``
4. ``<<<COMPLAINT`` block with ``masked_text[:1500]``, only when
   ``cfg.include_text`` is true
5. the final instruction

The record block markers are parsed by ``MockBackend`` in
``src/agentic/llm.py``; keep their exact format.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.explain.cache import ExplanationCache
from src.explain.config import MAX_RETRIES, PROMPT_VERSION, NarratorConfig
from src.explain.narration import (AGENT_MAX, CONSUMER_MAX, FACTS_MAX, PATH_MAX,
                                   Explanation, Narration, ValidationFailure,
                                   ValidationSummary, parse_narration)
from src.explain.record import DecisionRecord, canonical_json, hash_view, record_hash
from src.explain.store import AuditEntry, utc_now_iso
from src.explain.template import template_explanation
from src.explain.validator import ValidatorContext, default_context, validate

__all__ = ["PROMPT_VERSION", "MAX_COMPLAINT_CHARS", "NARRATION_SCHEMA",
           "RECORD_OPEN", "RECORD_CLOSE", "COMPLAINT_OPEN", "COMPLAINT_CLOSE",
           "FINAL_INSTRUCTION", "build_prompt", "Narrator", "NarrationOutcome"]

MAX_COMPLAINT_CHARS = 1500           # Req 11.5

RECORD_OPEN = "<<<DECISION_RECORD"
RECORD_CLOSE = "DECISION_RECORD>>>"
COMPLAINT_OPEN = "<<<COMPLAINT"
COMPLAINT_CLOSE = "COMPLAINT>>>"

# Narration JSON schema (design §Data Models), bounds shared with narration.py.
NARRATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["consumer_summary", "agent_summary", "cited_facts"],
    "properties": {
        "consumer_summary": {"type": "string", "minLength": 1, "maxLength": CONSUMER_MAX},
        "agent_summary": {"type": "string", "minLength": 1, "maxLength": AGENT_MAX},
        "cited_facts": {
            "type": "array",
            "maxItems": FACTS_MAX,
            "items": {
                "type": "object",
                "required": ["path", "value"],
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": PATH_MAX},
                    "value": {},
                },
            },
        },
    },
}

# The rules must not contain the literal record/complaint markers, so the
# Mock_Backend finds the real record block.
_RULES = (
    "You explain a complaint-routing decision that has already been made. "
    "Never change or second-guess the decision. Use only facts in the decision "
    "record below.\n"
    "Write both summaries in one reply.\n"
    "consumer_summary: at most 120 words, for the consumer. Name the team by its "
    "exact name, name the rank-1 issue label, and say exactly \"auto-routed\" or "
    "\"sent to human review\". Use no numbers, no percentages, no model names, "
    "and none of the words probability, score, threshold or confidence. Never "
    "mention other complaints.\n"
    "agent_summary: for a support agent. Name the top-3 Drivers of each section "
    "with status \"ok\", each model's top team, the models that disagree, the "
    "scenario, and the score compared with the threshold.\n"
    "Put quotes only around exact n-grams or sentences from the record or the "
    "complaint. Every number must be copied from the record.\n"
    "cited_facts: cite each fact you use as {\"path\": \"...\", \"value\": ...}, "
    "where path is the dotted path of the value in the decision record "
    "(list items by index, for example outcome.top_issues.0.label)."
)

FINAL_INSTRUCTION = "Reply with one JSON object only."


def build_prompt(record: DecisionRecord | dict, cfg: NarratorConfig,
                 masked_text: str | None) -> str:
    """The Narrator prompt for one record (design §14, Req 11.4 to 11.7).

    The record is serialised with the Record_Serializer after dropping
    request_id, created_at and timings (``hash_view``). The complaint block is
    present only when ``cfg.include_text`` is true and holds at most the first
    1500 characters of ``masked_text``; the caller must pass the Masked_Text
    (Req 11.7). Nothing from ``cfg`` other than ``include_text`` is used, so the
    API key and Agent_Token never reach the prompt.
    """
    parts = [
        _RULES,
        "Narration JSON schema:\n"
        + json.dumps(NARRATION_SCHEMA, sort_keys=True, separators=(",", ":")),
        f"{RECORD_OPEN}\n{canonical_json(hash_view(record))}\n{RECORD_CLOSE}",
    ]
    if cfg.include_text:
        text = (masked_text or "")[:MAX_COMPLAINT_CHARS]
        parts.append(f"{COMPLAINT_OPEN}\n{text}\n{COMPLAINT_CLOSE}")
    parts.append(FINAL_INSTRUCTION)
    return "\n\n".join(parts)


@dataclass
class NarrationOutcome:
    """Result of one ``Narrator.narrate`` call.

    ``audit`` holds one entry per backend reply or failure (Req 20.1, 20.5);
    the caller appends them to the Explanation_Store. ``backend_calls`` counts
    calls to ``backend.complete_text`` (0 on a cache hit).
    """
    explanation: Explanation
    audit: list[AuditEntry] = field(default_factory=list)
    backend_calls: int = 0


class Narrator:
    """Turns a Decision_Record into an Explanation (design section 14)."""

    def __init__(self, cfg: NarratorConfig, backend=None, prompt_cache=None,
                 explanation_cache: ExplanationCache | None = None, labels=None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.backend = backend
        self.prompt_cache = prompt_cache
        self.explanation_cache = explanation_cache
        self.labels = labels            # optional (team_names, issue_labels) mappings
        self.clock = clock

    # -- helpers -----------------------------------------------------------

    def _backend(self):
        if self.backend is None:
            from src.agentic.llm import get_backend
            c = self.cfg
            if c.backend == "mock":
                self.backend = get_backend("mock", mode=c.mock_mode, delay_s=c.mock_delay_s)
            else:
                self.backend = get_backend(
                    c.backend, base_url=c.base_url, api_key=c.api_key, model=c.model,
                    timeout_s=c.timeout_s, max_retries=MAX_RETRIES, clock=self.clock)
        return self.backend

    def _cache_model(self) -> str:
        # Mock modes must not share PromptCache entries.
        return self.cfg.model if self.cfg.backend != "mock" else f"mock-{self.cfg.mock_mode}"

    def _context(self, masked_text: str | None) -> ValidatorContext:
        ctx = default_context(masked_text, self.cfg.include_text, self.cfg.model)
        if self.labels is not None:
            teams, issues = self.labels
            ctx = ValidatorContext(masked_text=masked_text, include_text=self.cfg.include_text,
                                   team_names=teams, issue_labels=issues,
                                   narrator_model=self.cfg.model)
        return ctx

    # -- main --------------------------------------------------------------

    def narrate(self, record, masked_text: str | None, *, deadline: float) -> NarrationOutcome:
        cfg = self.cfg
        # Frozen deep copy: the decision cannot change (Req 11.9).
        if isinstance(record, DecisionRecord):
            rec = record.model_copy(deep=True)
        else:
            rec = DecisionRecord.model_validate(copy.deepcopy(record))
        h = record_hash(rec, cfg, masked_text)
        audit: list[AuditEntry] = []
        calls = 0

        def template(reason: str, failed=()) -> NarrationOutcome:
            return NarrationOutcome(
                template_explanation(rec, reason, failed, record_hash=h,
                                     created_at=utc_now_iso()), audit, calls)

        if not cfg.enabled:
            return template("invalid_config")
        if cfg.backend == "openai_compat" and not cfg.api_key:
            return template("no_api_key")

        if self.explanation_cache is not None:
            try:
                cached = self.explanation_cache.get(h)
            except Exception:
                cached = None
            if cached is not None:
                return NarrationOutcome(self._llm(cached, h), audit, 0)

        prompt = build_prompt(rec, cfg, masked_text)
        model = self._cache_model()
        psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        def log(source: str, **kw) -> None:
            audit.append(AuditEntry(
                request_id=rec.request_id, created_at=utc_now_iso(), backend=cfg.backend,
                model=model, prompt_version=PROMPT_VERSION, prompt_sha256=psha,
                record_hash=h, explanation_source=source, **kw))

        raw = None
        if self.prompt_cache is not None:
            try:
                raw = self.prompt_cache.get(model, prompt)
            except Exception:
                raw = None
        if raw is None:
            if self.clock() >= deadline:
                log("template", failure_reason="timeout")
                return template("timeout")
            try:
                calls += 1
                raw = self._backend().complete_text(prompt, max_tokens=cfg.max_tokens,
                                                    deadline=deadline)
            except Exception as e:
                reason = "timeout" if isinstance(e, TimeoutError) else "backend_failure"
                log("template", failure_reason=f"{reason}: {type(e).__name__}")
                return template(reason)
            if self.prompt_cache is not None:
                try:
                    self.prompt_cache.put(model, prompt, raw)
                except Exception:
                    pass

        parsed = parse_narration(raw)
        failed: tuple = ()
        reason = "parse_error"
        accepted = False
        validation = None
        if parsed.ok:
            try:
                result = validate(parsed.narration, rec, self._context(masked_text))
                accepted, failed, reason = result.accepted, result.failed, "validator_rejection"
            except Exception:
                reason = "validator_rejection"
            validation = {"accepted": accepted,
                          "failed": [{"check": f.check, "offending": f.offending,
                                      "reason": f.reason} for f in failed]}
        if accepted:
            log("llm", raw_reply=raw, parsed=parsed.narration, validation=validation)
            if self.explanation_cache is not None:
                try:
                    self.explanation_cache.put(h, parsed.narration)
                except Exception:
                    pass
            return NarrationOutcome(self._llm(parsed.narration, h), audit, calls)

        log("template", raw_reply=raw, parsed=parsed.narration if parsed.ok else None,
            parse_error=None if parsed.ok else "; ".join(parsed.errors),
            validation=validation, failure_reason=reason)
        if self.prompt_cache is not None:
            try:
                self.prompt_cache.delete(model, prompt)
            except Exception:
                pass
        return template(reason, failed)

    @staticmethod
    def _llm(n: Narration, h: str) -> Explanation:
        return Explanation(
            consumer_summary=n.consumer_summary, agent_summary=n.agent_summary,
            cited_facts=list(n.cited_facts), source="llm", fallback_reason=None,
            validation=ValidationSummary(accepted=True, failed=[]),
            record_hash=h, created_at=utc_now_iso())