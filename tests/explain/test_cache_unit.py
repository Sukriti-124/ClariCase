"""Unit tests for src/explain/cache.py (Req 16.1 to 16.3)."""
from __future__ import annotations

import threading

import pytest

from src.explain.cache import ExplanationCache
from src.explain.narration import CitedFact, Explanation, Narration, ValidationSummary, print_narration


def _n(i: int = 0) -> Narration:
    return Narration(consumer_summary=f"Sent to team {i}.",
                     agent_summary=f"IM1 chose T{i:02d}.",
                     cited_facts=[CitedFact(path="outcome.chosen_team.id", value=f"T{i:02d}")])


def _h(i: int) -> str:
    return f"{i:064x}"


def test_miss_then_hit_returns_equal_narration():
    c = ExplanationCache()
    assert c.get(_h(1)) is None
    c.put(_h(1), _n(1))
    got = c.get(_h(1))
    assert got == _n(1)
    assert print_narration(got) == print_narration(_n(1))
    assert len(c) == 1 and _h(1) in c


def test_put_overwrites_same_hash():
    c = ExplanationCache()
    c.put(_h(1), _n(1))
    c.put(_h(1), _n(2))
    assert c.get(_h(1)) == _n(2) and len(c) == 1


def test_lru_evicts_least_recently_used():
    c = ExplanationCache(max_entries=2)
    c.put(_h(1), _n(1))
    c.put(_h(2), _n(2))
    assert c.get(_h(1)) is not None      # 1 is now most recent
    c.put(_h(3), _n(3))                  # evicts 2
    assert _h(2) not in c
    assert c.get(_h(1)) == _n(1) and c.get(_h(3)) == _n(3)
    assert len(c) == 2


def test_default_capacity_and_invalid_capacity():
    assert ExplanationCache().max_entries == 10_000
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ExplanationCache(max_entries=bad)


def test_rejects_non_narration_and_bad_hash():
    c = ExplanationCache()
    expl = Explanation(**_n(1).model_dump(), source="template",
                       validation=ValidationSummary(accepted=True),
                       record_hash=_h(1), created_at="2025-01-01T00:00:00.000Z")
    with pytest.raises(TypeError):
        c.put(_h(1), expl)               # template output is never cached (Req 16.3)
    with pytest.raises(TypeError):
        c.put(_h(1), {"consumer_summary": "x"})
    with pytest.raises(ValueError):
        c.put("", _n(1))
    assert len(c) == 0


def test_corrupt_entry_is_dropped_as_miss():
    c = ExplanationCache()
    c.put(_h(1), _n(1))
    c._data[_h(1)] = "not json"
    assert c.get(_h(1)) is None and _h(1) not in c


def test_concurrent_put_get_respects_capacity():
    c = ExplanationCache(max_entries=50)
    errors: list[BaseException] = []

    def worker(base: int) -> None:
        try:
            for i in range(200):
                k = base * 1000 + i
                c.put(_h(k), _n(k % 100))
                got = c.get(_h(k))
                assert got is None or got == _n(k % 100)
        except BaseException as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(c) == 50
