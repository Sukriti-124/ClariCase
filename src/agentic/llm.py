"""LLM backends with a shared prompt cache.

`choose(prompt, candidates)` returns the candidate id the model picked. The
model is told to answer with the id only; the first candidate id found in
its reply wins. If none is found the first candidate (the ML's own top
choice) is returned with parsed=False, so an unparseable answer can never
make things worse than not asking.

Backends:
  openai_compat  any OpenAI-compatible chat endpoint (Groq, OpenAI, vLLM,
                 Ollama). Plain httpx, temperature 0, a handful of output
                 tokens, no reasoning budget.
  mock           deterministic, for tests and dry runs: picks the candidate
                 whose label words overlap the complaint most.

Every (model, prompt) pair is cached in SQLite, so an evaluation re-run or a
prompt tweak that touches only some rows costs only the new calls.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from src.agentic import config as C


@dataclass
class Choice:
    candidate: str
    parsed: bool
    raw: str
    cached: bool
    latency_ms: int


class PromptCache:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        self._con = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._con = sqlite3.connect(str(path), check_same_thread=False)
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, "
                "model TEXT, raw TEXT, ts REAL)")
            self._con.commit()

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        if self._con is None:
            return None
        with self._lock:
            row = self._con.execute(
                "SELECT raw FROM answers WHERE key=?",
                (self.key(model, prompt),)).fetchone()
        return None if row is None else row[0]

    def put(self, model: str, prompt: str, raw: str) -> None:
        if self._con is None:
            return
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO answers VALUES (?,?,?,?)",
                (self.key(model, prompt), model, raw, time.time()))
            self._con.commit()

    def __len__(self) -> int:
        if self._con is None:
            return 0
        return self._con.execute("SELECT COUNT(*) FROM answers").fetchone()[0]


def parse_choice(raw: str, candidates: Sequence[str]) -> tuple[str, bool]:
    text = raw or ""
    hits = []
    for c in candidates:
        m = re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", text)
        if m:
            hits.append((m.start(), c))
    if hits:
        return min(hits)[1], True
    return candidates[0], False


class Backend(Protocol):
    model: str

    def complete(self, prompt: str) -> str: ...


class MockBackend:
    """Deterministic stand-in: scores each candidate by how many words of its
    label (the text after `<id>:` on the candidate line) appear in the
    complaint. Ties go to the first candidate."""
    model = "mock"

    def complete(self, prompt: str) -> str:
        body = prompt.lower()
        complaint = body.split("complaint:")[-1] if "complaint:" in body else body
        words = set(re.findall(r"[a-z]{4,}", complaint))
        best, best_score = None, -1
        for line in prompt.splitlines():
            m = re.match(r"\s*-\s*([A-Z]\d{2}(?:_I\d{3})?)\s*:\s*(.*)", line)
            if not m:
                continue
            cid, label = m.group(1), m.group(2).lower()
            score = len(words & set(re.findall(r"[a-z]{4,}", label)))
            if score > best_score:
                best, best_score = cid, score
        return best or ""


class OpenAICompatBackend:
    def __init__(self, base_url: str = C.BASE_URL, api_key: str = C.API_KEY,
                 model: str = C.MODEL, timeout_s: float = C.TIMEOUT_S,
                 max_retries: int = C.MAX_RETRIES):
        import httpx
        if not api_key:
            raise RuntimeError(
                "No API key: set AGENTIC_API_KEY (or GROQ_API_KEY / "
                "OPENAI_API_KEY), or use AGENTIC_BACKEND=mock for a dry run")
        self.model = model
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}",
                     "content-type": "application/json"})

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": 16,
            "messages": [{"role": "user", "content": prompt}],
        }
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            resp = self._client.post("/chat/completions", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return (data["choices"][0]["message"].get("content") or "").strip()
            if resp.status_code in (408, 409, 429) or resp.status_code >= 500:
                if attempt == self.max_retries:
                    resp.raise_for_status()
                retry_after = resp.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else delay)
                delay = min(delay * 2, 30.0)
                continue
            resp.raise_for_status()
        raise RuntimeError("unreachable")


def get_backend(name: str | None = None, **kw) -> Backend:
    """Backend named by `name`, else by AGENTIC_BACKEND at call time (so a
    service reads the env it was started with, not the import-time value)."""
    import os
    name = name or os.environ.get("AGENTIC_BACKEND", C.BACKEND)
    if name == "mock":
        return MockBackend()
    if name == "openai_compat":
        return OpenAICompatBackend(**kw)
    raise ValueError(f"unknown AGENTIC_BACKEND {name!r}")


class Chooser:
    def __init__(self, backend: Backend, cache: PromptCache | None = None):
        self.backend = backend
        self.cache = cache if cache is not None else PromptCache(None)

    def choose(self, prompt: str, candidates: Sequence[str]) -> Choice:
        if not candidates:
            raise ValueError("no candidates")
        t0 = time.perf_counter()
        raw = self.cache.get(self.backend.model, prompt)
        cached = raw is not None
        if raw is None:
            raw = self.backend.complete(prompt)
            self.cache.put(self.backend.model, prompt, raw)
        picked, parsed = parse_choice(raw, candidates)
        return Choice(candidate=picked, parsed=parsed, raw=raw, cached=cached,
                      latency_ms=int((time.perf_counter() - t0) * 1000))

    def choose_many(self, prompts: Sequence[str],
                    candidates: Sequence[Sequence[str]],
                    concurrency: int = C.CONCURRENCY,
                    progress: bool = False) -> list[Choice]:
        from concurrent.futures import ThreadPoolExecutor
        out: list[Choice | None] = [None] * len(prompts)

        def work(i):
            out[i] = self.choose(prompts[i], candidates[i])
            return i

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            it = ex.map(work, range(len(prompts)))
            if progress:
                try:
                    from tqdm import tqdm
                    it = tqdm(it, total=len(prompts), unit="row")
                except ImportError:
                    pass
            for _ in it:
                pass
        return out  # type: ignore[return-value]
