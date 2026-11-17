"""Unit tests for src/explain/store.py with an injected clock."""
from __future__ import annotations

import pytest
from hypothesis import find

from src.app.pii import mask
from src.explain.narration import Explanation, Narration, ValidationSummary
from src.explain.store import AuditEntry, ExplanationStore
from tests.explain.conftest import FakeClock
from tests.explain.strategies import st_record

RECORD = find(st_record, lambda r: True)


def _audit(rid: str, created_at: str, **kw) -> AuditEntry:
    base = dict(request_id=rid, created_at=created_at, backend="mock", model="m",
                prompt_version="v1", prompt_sha256="0" * 64, record_hash="a" * 64,
                explanation_source="llm")
    base.update(kw)
    return AuditEntry(**base)


def _explanation(summary: str = "Routed to T01.") -> Explanation:
    return Explanation(consumer_summary=summary, agent_summary="agent text",
                       cited_facts=[], source="template", fallback_reason="timeout",
                       validation=ValidationSummary(accepted=True),
                       record_hash="a" * 64, created_at="2025-01-01T00:00:00.000Z")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(clock: FakeClock) -> ExplanationStore:
    return ExplanationStore(ttl_s=100, max_ids=3, clock=clock.monotonic)


def test_lifecycle_pending_ready(store):
    store.put_record("r1", RECORD, "masked text")
    e = store.get("r1")
    assert e.status == "pending" and e.record == RECORD and e.masked_text == "masked text"
    assert store.put_explanation("r1", _explanation())
    e = store.get("r1")
    assert e.status == "ready" and e.explanation.source == "template"
    assert e.explanation.fallback_reason == "timeout"


def test_mark_failed_keeps_record(store):
    store.put_record("r1", RECORD, "t")
    assert store.mark_failed("r1", "explanation timed out")
    e = store.get("r1")
    assert e.status == "failed" and e.record == RECORD
    assert e.failure["reason"] == "explanation timed out"


def test_put_failure_stores_no_record(store):
    store.put_failure("r1", "BuildError")
    e = store.get("r1")
    assert e.status == "failed" and e.record is None
    assert e.failure["error_type"] == "BuildError"
    assert e.failure["reason"] == "record build failed: BuildError"
    assert not store.put_explanation("r1", _explanation())


def test_unknown_id_and_writes_to_unknown_id(store):
    assert store.get("nope") is None
    assert not store.put_explanation("nope", _explanation())
    assert not store.mark_failed("nope", "x")
    assert not store.add_audit("nope", _audit("nope", "2025-01-01T00:00:00.000Z"))
    assert store.get("nope") is None


def test_oldest_first_eviction(store):
    for rid in ("a", "b", "c", "d"):
        store.put_record(rid, RECORD, "t")
    assert len(store) == 3
    assert store.get("a") is None
    assert all(store.get(r) is not None for r in ("b", "c", "d"))


def test_ttl_eviction_on_get(store, clock):
    store.put_record("r1", RECORD, "t")
    clock.advance(100)
    assert store.get("r1") is not None          # exactly ttl_s old: kept
    clock.advance(0.001)
    assert store.get("r1") is None
    assert len(store) == 0


def test_audit_sorted_by_timestamp_with_insertion_tiebreak(store):
    store.put_record("r1", RECORD, "t")
    store.add_audit("r1", _audit("r1", "2025-01-01T00:00:02.000Z", raw_reply="second"))
    store.add_audit("r1", _audit("r1", "2025-01-01T00:00:01.000Z", raw_reply="first"))
    store.add_audit("r1", _audit("r1", "2025-01-01T00:00:02.000Z", raw_reply="third"))
    first = store.get("r1").audit
    assert [a.raw_reply for a in first] == ["first", "second", "third"]
    # Append-only: a new entry leaves earlier ones unchanged.
    store.add_audit("r1", _audit("r1", "2025-01-01T00:00:03.000Z", failure_reason="timeout",
                                 explanation_source="template"))
    after = store.get("r1").audit
    assert after[:3] == first and len(after) == 4


def test_add_audit_masks_raw_reply_and_marks_redacted(store):
    store.put_record("r1", RECORD, "t")
    reply = '{"consumer_summary": "Call 555-123-4567"}'
    parsed = Narration(consumer_summary="Call 555-123-4567", agent_summary="a",
                       cited_facts=[])
    store.add_audit("r1", _audit("r1", "2025-01-01T00:00:00.000Z", raw_reply=reply,
                                 parsed=parsed))
    store.add_audit("r1", _audit("r1", "2025-01-01T00:00:01.000Z", raw_reply="clean"))
    masked, clean = store.get("r1").audit
    assert masked.redacted and "555-123-4567" not in masked.raw_reply
    assert masked.raw_reply == mask(reply)
    assert masked.parsed.consumer_summary == "Call XXXX"
    assert not clean.redacted and clean.raw_reply == "clean"


def test_audit_request_id_must_match(store):
    store.put_record("r1", RECORD, "t")
    with pytest.raises(ValueError):
        store.add_audit("r1", _audit("r2", "2025-01-01T00:00:00.000Z"))


def test_get_returns_snapshot(store):
    store.put_record("r1", RECORD, "t")
    store.add_audit("r1", _audit("r1", "2025-01-01T00:00:00.000Z"))
    snap = store.get("r1")
    snap.audit.clear()
    snap.status = "ready"
    e = store.get("r1")
    assert e.status == "pending" and len(e.audit) == 1
