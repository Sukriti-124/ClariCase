"""Unit tests for src/explain/pipeline.py (task 21.1).

**Validates: Requirements 1.3, 1.4, 1.8, 1.10, 11.1, 14.6, 14.7, 17.7**
"""
from __future__ import annotations

import time

from src.explain.config import NarratorConfig
from src.explain.explainers.routing import explain_routing_streamlit
from src.explain.narrator import Narrator
from src.explain.pipeline import (ExplainResult, UpstreamFailure, explain_platform,
                                  explain_streamlit, narrate_and_store)
from src.explain.explainers import skipped_section
from src.explain.record import Driver
from src.explain.store import ExplanationStore
from tests.explain.test_builder_unit import RAW, _im1, _outcome


def _narrator() -> Narrator:
    return Narrator(NarratorConfig(backend="mock", mock_mode="failure_parse"))


def _sections():
    return {"improved_model_1": _im1(Driver(kind="ngram", label="word__refund",
                                            contribution=0.5)),
            "routing": explain_routing_streamlit(0.9, 0.5)}


def _run(store, narrator=None, **kw):
    args = dict(store=store, narrator=narrator or _narrator(), request_id="r1",
                serving_path="streamlit", model_versions={"improved_model_1": "v1"},
                outcome=_outcome(), sections=_sections(), raw_text=RAW,
                masked_text=RAW, deadline=time.monotonic() + 100)
    args.update(kw)
    return narrate_and_store(**args)


def test_success_stores_record_audit_and_explanation():
    store = ExplanationStore()
    res = _run(store)
    entry = store.get("r1")
    assert res.error is None and entry.status == "ready"
    assert entry.record == res.record and entry.explanation is not None
    assert len(entry.audit) == len(res.narration.audit) == 1


def test_build_error_stores_failure_and_skips_narration():
    store = ExplanationStore()
    res = _run(store, served=_outcome(auto=False))
    assert res.record is None and res.narration is None
    entry = store.get("r1")
    assert entry.status == "failed" and entry.record is None
    assert entry.failure["error_type"] == "outcome_mismatch"


def test_template_error_marks_failed_and_keeps_record():
    class Boom(Narrator):
        def narrate(self, *a, **k):
            raise ValueError("x")

    store = ExplanationStore()
    res = _run(store, Boom(NarratorConfig(backend="mock")))
    entry = store.get("r1")
    assert res.error == "template error: ValueError"
    assert entry.status == "failed" and entry.record is not None
    assert entry.failure["reason"] == "template error: ValueError"


def test_no_store_still_returns_result():
    res = _run(None)
    assert isinstance(res, ExplainResult) and res.narration is not None


def test_explain_streamlit_deadline_and_skips():
    seen = {}

    class Spy(Narrator):
        def narrate(self, record, masked_text, *, deadline):
            seen["d"] = deadline
            return super().narrate(record, masked_text, deadline=deadline)

    cfg = NarratorConfig(backend="mock", mock_mode="failure_parse", timeout_s=10.0)
    n = Spy(cfg, clock=lambda: 100.0)
    # Broken IM1 inputs: the section becomes an error, the record still builds.
    res = explain_streamlit(
        request_id="r2", raw_text=RAW, masked_text=RAW, vectorizer=None, hsvm=None,
        team_classes=None, served_issues=None, confidence=0.9, global_threshold=0.5,
        outcome=_outcome(), model_versions={"improved_model_1": "v1"}, narrator=n)
    assert seen["d"] == 113.0
    secs = res.record.sections
    assert secs["improved_model_1"].status == "error"
    assert secs["routing"].status == "ok"
    assert secs["stacker"].reason == "not available on this serving path"


def test_explain_platform_upstream_failure_and_bad_ctx():
    class Ctx:  # malformed context: platform explainers must become error sections
        decision = bundle = None
        pre_llm_team = None
        agentic_info = None
        missing_models = ()

    secs = explain_platform(Ctx(), {"improved_model_1": UpstreamFailure("svm-api"),
                                    "knn_embed": UpstreamFailure("embed-api"),
                                    "baseline_model_2": skipped_section("baseline_model_2", "n/a")})
    assert secs["improved_model_1"].status == "error"
    assert secs["improved_model_1"].reason == "upstream explain failed: svm-api"
    assert secs["knn_embed"].reason == "upstream explain failed: embed-api"
    assert secs["stacker"].status == "error" and secs["routing"].status == "error"
    assert secs["agentic"].reason == "not consulted"
    assert secs["baseline_model_2"].status == "skipped"
