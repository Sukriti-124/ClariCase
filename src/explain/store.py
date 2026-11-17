"""In-memory Explanation_Store with TTL/LRU bounds and audit entries.

One ``StoreEntry`` per request_id (Req 17.8, 20.7). The store is an
``OrderedDict`` behind a ``threading.Lock``: inserting a new request_id beyond
``max_ids`` evicts the oldest one, and ``get`` evicts entries older than
``ttl_s`` (measured with the injected ``clock``).

Lifecycle:

    put_record       -> status "pending"  (record + masked text)
    put_failure      -> status "failed"   (no record; Req 1.10)
    put_explanation  -> status "ready"    (Req 14.6, 14.7)
    mark_failed      -> status "failed"   (record kept; Req 17.7)
    add_audit        -> append-only audit entry (Req 20.1, 20.4, 20.5)

Every text that enters the store passes through ``src.app.pii.mask``. An audit
entry whose raw reply (or another text field) changes under masking is stored
in masked form with ``redacted=True`` (Req 20.2, 20.6). The full prompt is
never stored, only its SHA-256 (Req 20.1).

``get`` returns a snapshot: the audit list is a fresh list sorted by
``created_at`` with insertion index as the tie-break (Req 20.3), so callers
cannot modify stored entries.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict

from src.app.pii import mask
from src.explain.narration import CitedFact, Explanation, Narration
from src.explain.record import DecisionRecord

Status = Literal["pending", "ready", "failed"]


def utc_now_iso() -> str:
    """UTC timestamp in ISO 8601 with millisecond precision, e.g. ``...T12:00:00.000Z``."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AuditEntry(BaseModel):
    """One backend reply (or failed call) for a request_id (Req 20.1, 20.5)."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    created_at: str
    backend: str
    model: str
    prompt_version: str
    prompt_sha256: str
    record_hash: str
    raw_reply: str | None = None
    redacted: bool = False
    parsed: Narration | None = None
    parse_error: str | None = None
    validation: dict[str, Any] | None = None
    failure_reason: str | None = None
    explanation_source: Literal["llm", "template"]


@dataclass
class StoreEntry:
    status: Status
    record: DecisionRecord | None = None
    masked_text: str | None = None
    explanation: Explanation | None = None
    audit: list[AuditEntry] = field(default_factory=list)
    failure: dict | None = None
    inserted_at: float = 0.0


# ---------------------------------------------------------------------------
# Masking helpers (Req 20.2, 20.6)
# ---------------------------------------------------------------------------

def _m(text: str | None) -> str | None:
    return mask(text) if text else text


def _mask_facts(facts: list[CitedFact]) -> list[CitedFact]:
    out = []
    for f in facts:
        v = f.value
        if isinstance(v, str):
            v = mask(v)
        elif isinstance(v, list):
            v = [mask(x) if isinstance(x, str) else x for x in v]
        out.append(f if v == f.value else f.model_copy(update={"value": v}))
    return out


def _mask_summaries(n: Narration | Explanation) -> Any:
    """Mask the summaries and string Cited_Fact values without re-validating
    (masking can change lengths, and the stored form must be the masked one)."""
    update = {"consumer_summary": mask(n.consumer_summary),
              "agent_summary": mask(n.agent_summary),
              "cited_facts": _mask_facts(list(n.cited_facts))}
    if all(update[k] == getattr(n, k) for k in update):
        return n
    return n.model_copy(update=update)


def _mask_audit(entry: AuditEntry) -> AuditEntry:
    update: dict[str, Any] = {
        "raw_reply": _m(entry.raw_reply),
        "parse_error": _m(entry.parse_error),
        "failure_reason": _m(entry.failure_reason),
    }
    if entry.parsed is not None:
        update["parsed"] = _mask_summaries(entry.parsed)
    if all(update[k] == getattr(entry, k) for k in update):
        return entry
    update["redacted"] = True
    return entry.model_copy(update=update)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@dataclass
class _Slot:
    entry: StoreEntry
    audit: list[tuple[int, AuditEntry]] = field(default_factory=list)
    next_index: int = 0


class ExplanationStore:
    """Thread-safe in-memory store keyed by request_id (Assumption A8)."""

    def __init__(self, ttl_s: float = 86_400, max_ids: int = 10_000,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_ids < 1:
            raise ValueError("max_ids must be at least 1")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self.ttl_s = ttl_s
        self.max_ids = max_ids
        self._clock = clock
        self._lock = threading.Lock()
        self._data: OrderedDict[str, _Slot] = OrderedDict()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # -- writes -------------------------------------------------------------

    def _insert(self, rid: str, entry: StoreEntry) -> None:
        """Insert or replace ``rid`` as the newest id; evict the oldest beyond
        ``max_ids``. Caller holds the lock."""
        entry.inserted_at = self._clock()
        self._data.pop(rid, None)
        self._data[rid] = _Slot(entry)
        while len(self._data) > self.max_ids:
            self._data.popitem(last=False)

    def put_record(self, rid: str, record: DecisionRecord, masked_text: str | None) -> None:
        """Store the Decision_Record with status ``pending``."""
        with self._lock:
            self._insert(rid, StoreEntry(status="pending", record=record,
                                         masked_text=_m(masked_text)))

    def put_failure(self, rid: str, error_type: str) -> None:
        """Record a build failure with no Decision_Record (Req 1.10)."""
        error_type = mask(str(error_type))
        failure = {"request_id": rid, "error_type": error_type, "at": utc_now_iso(),
                   "reason": f"record build failed: {error_type}"}
        with self._lock:
            self._insert(rid, StoreEntry(status="failed", failure=failure))

    def put_explanation(self, rid: str, explanation: Explanation) -> bool:
        """Attach the Explanation and set status ``ready``. Returns False if
        ``rid`` has no stored Decision_Record (unknown, evicted or a failure)."""
        explanation = _mask_summaries(explanation)
        with self._lock:
            slot = self._data.get(rid)
            if slot is None or slot.entry.record is None:
                return False
            slot.entry.explanation = explanation
            slot.entry.status = "ready"
            slot.entry.failure = None
            return True

    def mark_failed(self, rid: str, reason: str) -> bool:
        """Set status ``failed`` and keep the Decision_Record (Req 17.7)."""
        with self._lock:
            slot = self._data.get(rid)
            if slot is None:
                return False
            slot.entry.status = "failed"
            slot.entry.failure = {"request_id": rid, "reason": mask(str(reason)),
                                  "at": utc_now_iso()}
            return True

    def add_audit(self, rid: str, entry: AuditEntry) -> bool:
        """Append a masked audit entry; earlier entries are never changed
        (Req 20.4, 20.6). Returns False if ``rid`` is not stored."""
        if entry.request_id != rid:
            raise ValueError("audit entry request_id does not match rid")
        entry = _mask_audit(entry)
        with self._lock:
            slot = self._data.get(rid)
            if slot is None:
                return False
            slot.audit.append((slot.next_index, entry))
            slot.next_index += 1
            return True

    # -- reads --------------------------------------------------------------

    def get(self, rid: str) -> StoreEntry | None:
        """A snapshot of the entry, or None if unknown or older than ``ttl_s``."""
        with self._lock:
            slot = self._data.get(rid)
            if slot is None:
                return None
            if self._clock() - slot.entry.inserted_at > self.ttl_s:
                del self._data[rid]
                return None
            audit = [e for _, e in sorted(slot.audit,
                                          key=lambda p: (p[1].created_at, p[0]))]
            failure = dict(slot.entry.failure) if slot.entry.failure else None
            return replace(slot.entry, audit=audit, failure=failure)


__all__ = ["AuditEntry", "StoreEntry", "ExplanationStore", "utc_now_iso"]
