"""Explanation_Cache: an LRU from Record_Hash to printed Narration JSON (Req 16).

Only the Narrator writes to it, and only after a Narration has passed the
Faithfulness_Validator (Req 16.1, 16.3). Template output and rejected replies
are never stored. Entries are kept as canonical Narration JSON text
(``print_narration``) and parsed back on ``get``, so callers never share an
object with the cache.

The cache is thread-safe: ensemble-api narrates from background tasks.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

from src.explain.narration import Narration, parse_narration, print_narration

DEFAULT_MAX_ENTRIES = 10_000


class ExplanationCache:
    """In-memory LRU keyed by Record_Hash (design §18)."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._data: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, h: str) -> Narration | None:
        """The cached Narration for ``h``, or None on a miss (Req 16.2).

        A hit marks the entry as most recently used. An entry that no longer
        parses is dropped and reported as a miss.
        """
        with self._lock:
            text = self._data.get(h)
            if text is None:
                return None
            self._data.move_to_end(h)
        parsed = parse_narration(text)
        if parsed.ok:
            return parsed.narration
        with self._lock:
            if self._data.get(h) == text:
                del self._data[h]
        return None

    def put(self, h: str, n: Narration) -> None:
        """Store a validated Narration under ``h`` (Req 16.1).

        Inserting beyond ``max_entries`` evicts the least recently used entry.
        """
        if not isinstance(h, str) or not h:
            raise ValueError("Record_Hash must be a non-empty string")
        if not isinstance(n, Narration):
            raise TypeError("only a Narration can be cached")
        text = print_narration(n)
        with self._lock:
            self._data[h] = text
            self._data.move_to_end(h)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, h: object) -> bool:
        with self._lock:
            return h in self._data
