"""Orchestration of one explanation: sections, record, narration, store (design §10).

* ``explain_streamlit`` runs the IM1 and Routing explainers (every other section
  is skipped by the builder, Req 1.3), builds the record and narrates it with
  the deadline ``now + cfg.timeout_s + 3`` (Req 11.1).
* ``explain_platform`` builds the stacker, gate, routing and agentic sections
  from a ``PlatformDecisionContext`` and merges the sections the svm-api and
  embed-api produced. An upstream failure becomes an ``error`` section with the
  reason ``upstream explain failed: <service>`` (Req 1.4).
* ``narrate_and_store`` builds the record, stores it (or a failure entry on
  ``BuildError``, Req 1.10), narrates, and stores the Explanation (Req 14.6,
  14.7) or marks the entry failed when the template raises (Req 17.7).

Every explainer runs through ``run_explainer``, so an exception only produces
that section's ``error`` (Req 1.4). Nothing here changes the served decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from src.explain.builder import BuildError, build_record
from src.explain.explainers import error_section, run_explainer, skipped_section
from src.explain.explainers.agentic import record_agentic
from src.explain.explainers.gate import explain_gate
from src.explain.explainers.im1 import explain_im1
from src.explain.explainers.routing import (PlatformDecisionContext,
                                            explain_routing_platform,
                                            explain_routing_streamlit)
from src.explain.explainers.stacker import explain_stacker
from src.explain.narrator import NarrationOutcome, Narrator
from src.explain.record import DecisionRecord, Outcome, Section
from src.explain.store import ExplanationStore

log = logging.getLogger(__name__)

__all__ = ["ExplainResult", "UpstreamFailure", "UPSTREAM_SERVICES",
           "explain_streamlit", "explain_platform", "narrate_and_store"]

# Sections produced by another service on the Platform.
UPSTREAM_SERVICES: dict[str, str] = {
    "improved_model_1": "svm-api", "baseline_model_2": "svm-api",
    "improved_model_2": "embed-api", "knn_embed": "embed-api",
}


@dataclass(frozen=True)
class UpstreamFailure:
    """Marks an upstream ``/explain`` call that failed or timed out."""
    service: str


@dataclass
class ExplainResult:
    """``record`` is None after a ``BuildError``; ``narration`` is None when the
    record was not built or the template raised (``error`` then says why)."""
    record: DecisionRecord | None = None
    narration: NarrationOutcome | None = None
    error: str | None = None


# --------------------------------------------------------------------------
# Streamlit
# --------------------------------------------------------------------------

def explain_streamlit(*, request_id: str, raw_text: str, masked_text: str,
                      vectorizer: Any, hsvm: Any, team_classes: Any,
                      served_issues: Any, confidence: float,
                      global_threshold: float | None,
                      outcome: Outcome | Mapping, model_versions: Mapping[str, str],
                      narrator: Narrator, store: ExplanationStore | None = None,
                      n: int | None = None, timings: Mapping[str, int] | None = None,
                      vectorizers: Mapping[str, Any] | None = None) -> ExplainResult:
    """Explain one Streamlit prediction (IM1 + Routing; the rest are skipped)."""
    chosen = outcome.chosen_team.id if isinstance(outcome, Outcome) \
        else outcome["chosen_team"]["id"]
    kw = {} if n is None else {"n": n}
    sections = {
        "improved_model_1": run_explainer(
            "improved_model_1", explain_im1, vectorizer, hsvm, team_classes,
            raw_text, masked_text, chosen, served_issues, **kw),
        "routing": run_explainer("routing", explain_routing_streamlit,
                                 confidence, global_threshold),
    }
    deadline = narrator.clock() + narrator.cfg.deadline_s("streamlit")
    return narrate_and_store(
        store=store, narrator=narrator, request_id=request_id,
        serving_path="streamlit", model_versions=model_versions, outcome=outcome,
        sections=sections, raw_text=raw_text, masked_text=masked_text,
        deadline=deadline, timings=timings, vectorizers=vectorizers)


# --------------------------------------------------------------------------
# Platform
# --------------------------------------------------------------------------

def _stack_and_gate(ctx: PlatformDecisionContext) -> dict[str, Section]:
    d, b = ctx.decision, ctx.bundle
    missing = list(ctx.missing_models or ())
    if missing or d.scenario == "degraded":
        reason = f"degraded: missing models {missing}"
        return {"stacker": skipped_section("stacker", reason),
                "gate": skipped_section("gate", reason)}
    group = (["stack"] if b.uses_stacker else []) + list(b.models)
    return {
        "stacker": run_explainer(
            "stacker", explain_stacker, b.raw.get("stacker"), b.models, b.teams,
            ctx.probs, d.team, d.stack_raw_probs, d.stack_probs,
            bool(d.prior_active), bool(b.uses_stacker)),
        "gate": run_explainer(
            "gate", explain_gate, b.raw["gate"], b.raw["gate_kind"],
            b.raw["gate_feature_names"], d.gate_features,
            b.raw.get("gate_reference_values"), group, float(d.score),
            d.threshold),
    }


def explain_platform(ctx: PlatformDecisionContext,
                     upstream_sections: Mapping[str, Any] | None = None
                     ) -> dict[str, Section | Mapping]:
    """Sections for one Platform request, keyed by section name.

    ``upstream_sections`` maps a section name to the section the other service
    returned, or to an ``UpstreamFailure``. Names not listed are left out, so
    the builder fills them with the serving-path skip.
    """
    sections: dict[str, Any] = {}
    try:
        sections.update(_stack_and_gate(ctx))
    except Exception as exc:        # a malformed bundle must not stop the rest
        for name in ("stacker", "gate"):
            sections[name] = error_section(name, exc)
    sections["routing"] = run_explainer("routing", explain_routing_platform, ctx)
    sections["agentic"] = run_explainer(
        "agentic", record_agentic, ctx.agentic_info, "platform", ctx.pre_llm_team)
    for name, sec in (upstream_sections or {}).items():
        if isinstance(sec, UpstreamFailure):
            service = sec.service or UPSTREAM_SERVICES.get(name, "unknown")
            sections[name] = error_section(
                name, RuntimeError(f"upstream explain failed: {service}"))
            sections[name] = sections[name].model_copy(
                update={"reason": f"upstream explain failed: {service}"})
        elif sec is not None:
            sections[name] = sec
    return sections


# --------------------------------------------------------------------------
# Build, store, narrate
# --------------------------------------------------------------------------

def narrate_and_store(*, store: ExplanationStore | None, narrator: Narrator,
                      request_id: str, serving_path: str,
                      model_versions: Mapping[str, str],
                      outcome: Outcome | Mapping,
                      sections: Mapping[str, Any], raw_text: str | None,
                      masked_text: str | None, deadline: float,
                      served: Outcome | Mapping | None = None,
                      timings: Mapping[str, int] | None = None,
                      vectorizers: Mapping[str, Any] | None = None
                      ) -> ExplainResult:
    """Build the record, store it, narrate it and store the Explanation.

    ``store`` may be None (Streamlit has no store). Never raises for a build or
    template failure; the result carries ``error`` instead.
    """
    try:
        record = build_record(
            request_id=request_id, serving_path=serving_path,
            model_versions=model_versions, outcome=outcome, sections=sections,
            served=served, raw_text=raw_text, masked_text=masked_text,
            vectorizers=vectorizers, timings=timings)
    except BuildError as exc:
        log.warning("record build failed for %s: %s", request_id, exc.error_type)
        if store is not None:
            store.put_failure(request_id, exc.error_type)
        return ExplainResult(error=f"record build failed: {exc.error_type}")

    if store is not None:
        store.put_record(request_id, record, masked_text)
    try:
        result = narrator.narrate(record, masked_text, deadline=deadline)
    except Exception as exc:
        reason = f"template error: {type(exc).__name__}"
        log.warning("narration failed for %s: %s", request_id, reason)
        if store is not None:
            store.mark_failed(request_id, reason)
        return ExplainResult(record=record, error=reason)

    if store is not None:
        for entry in result.audit:
            store.add_audit(request_id, entry)
        store.put_explanation(request_id, result.explanation)
    return ExplainResult(record=record, narration=result)
