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
from typing import Callable, Protocol, Sequence

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

    def delete(self, model: str, prompt: str) -> None:
        """Remove the entry for (model, prompt); a no-op when it is absent."""
        if self._con is None:
            return
        with self._lock:
            self._con.execute("DELETE FROM answers WHERE key=?",
                              (self.key(model, prompt),))
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

    def complete_text(self, prompt: str, *, max_tokens: int,
                      deadline: float | None = None) -> str: ...


MOCK_MODES = ("choice", "faithful", "unfaithful", "failure_parse", "failure_timeout")
MOCK_PARSE_FAILURE_REPLY = "Sorry, I cannot help with that."
_RECORD_RE = re.compile(r"<<<DECISION_RECORD\n(.*?)\nDECISION_RECORD>>>", re.DOTALL)


def _record_from_prompt(prompt: str) -> dict:
    """The Decision_Record JSON between the narrator prompt markers."""
    m = _RECORD_RE.search(prompt)
    if m is None:
        raise ValueError("mock backend: no <<<DECISION_RECORD block in the prompt")
    return json.loads(m.group(1))


def _foreign_team(record_text: str) -> str:
    """A team id (Tnn) that does not occur anywhere in the record text;
    known team ids first, in sorted order, so the choice is deterministic."""
    known: list[str] = []
    try:
        from src.explain.labels import load_tables
        known = sorted(load_tables().teams)
    except Exception:
        pass
    for tid in [*known, *(f"T{i:02d}" for i in range(1, 100))]:
        if not re.search(rf"\b{re.escape(tid)}\b", record_text):
            return tid
    raise ValueError("mock backend: no team id outside the record")


class MockBackend:
    """Deterministic stand-in, with no network access.

    `complete` (the chooser path) scores each candidate by how many words of
    its label (the text after `<id>:` on the candidate line) appear in the
    complaint. Ties go to the first candidate.

    `complete_text` depends on `mode` (design §13, Req 11.13, 21.2):
      choice           same reply as `complete`
      faithful         the template narration of the prompt's Decision_Record
      unfaithful       faithful, with the first numeric Cited_Fact set to
                       value + 0.5 and the Chosen_Team id in agent_summary
                       replaced by a team id not in the record
      failure_parse    a reply with no JSON object
      failure_timeout  raises TimeoutError
    `delay_s` sleeps before every reply. The chooser mode keeps the model
    name "mock"; narration modes use "mock-<mode>" so a PromptCache never
    serves one mode's reply to another."""

    def __init__(self, mode: str = "choice", delay_s: float = 0.0, *,
                 sleep: Callable[[float], None] = time.sleep):
        if mode not in MOCK_MODES:
            raise ValueError(f"unknown mock mode {mode!r}")
        if not isinstance(delay_s, (int, float)) or isinstance(delay_s, bool) \
                or not delay_s >= 0:
            raise ValueError("mock delay_s must be a number >= 0")
        self.mode = mode
        self.delay_s = float(delay_s)
        self._sleep = sleep
        self.model = "mock" if mode == "choice" else f"mock-{mode}"

    def _wait(self) -> None:
        if self.delay_s > 0:
            self._sleep(self.delay_s)

    def complete(self, prompt: str) -> str:
        self._wait()
        return self._choose(prompt)

    def _choose(self, prompt: str) -> str:
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

    def complete_text(self, prompt: str, *, max_tokens: int,
                      deadline: float | None = None) -> str:
        self._wait()
        if self.mode == "choice":
            return self._choose(prompt)
        if self.mode == "failure_parse":
            return MOCK_PARSE_FAILURE_REPLY
        if self.mode == "failure_timeout":
            raise TimeoutError("mock backend timeout")
        # Lazy import: src.agentic must not depend on src.explain at import time.
        from src.explain.narration import print_narration
        from src.explain.template import template_narration
        record = _record_from_prompt(prompt)
        narration = template_narration(record)
        if self.mode == "unfaithful":
            narration = self._unfaithful(narration, record)
        return print_narration(narration)

    @staticmethod
    def _unfaithful(narration, record: dict):
        from src.explain.narration import CitedFact
        from src.explain.record import canonical_json
        foreign = _foreign_team(canonical_json(record))
        facts = list(narration.cited_facts)
        idx = next((i for i, f in enumerate(facts)
                    if isinstance(f.value, (int, float))
                    and not isinstance(f.value, bool)), None)
        if idx is not None:
            facts[idx] = CitedFact(path=facts[idx].path, value=facts[idx].value + 0.5)
        elif facts:   # no numeric fact: give the first fact the foreign team id
            facts[0] = CitedFact(path=facts[0].path, value=foreign)
        chosen = (record.get("outcome") or {}).get("chosen_team", {}).get("id")
        agent = narration.agent_summary
        if isinstance(chosen, str) and chosen:
            agent = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(chosen)}(?![A-Za-z0-9_])",
                           foreign, agent)
        return type(narration)(consumer_summary=narration.consumer_summary,
                               agent_summary=agent, cited_facts=facts)


class OpenAICompatBackend:
    def __init__(self, base_url: str = C.BASE_URL, api_key: str = C.API_KEY,
                 model: str = C.MODEL, timeout_s: float = C.TIMEOUT_S,
                 max_retries: int = C.MAX_RETRIES, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 transport=None):
        import httpx
        if not api_key:
            raise RuntimeError(
                "No API key: set AGENTIC_API_KEY (or GROQ_API_KEY / "
                "OPENAI_API_KEY), or use AGENTIC_BACKEND=mock for a dry run")
        self.model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._clock = clock
        self._sleep = sleep
        client_kw = {} if transport is None else {"transport": transport}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}",
                     "content-type": "application/json"}, **client_kw)

    def _chat(self, payload: dict, deadline: float | None) -> str:
        """POST a chat completion with retries on retryable statuses.

        With a `deadline` (absolute, on `clock`), no attempt or back-off is
        started that would end after it: each attempt's HTTP timeout is
        capped at the time remaining, and `TimeoutError` is raised once the
        deadline is reached or a back-off would cross it. With
        `deadline=None` the behaviour is the original chooser loop."""
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            post_kw = {}
            if deadline is not None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise TimeoutError("narrator deadline reached")
                post_kw["timeout"] = min(self.timeout_s, remaining)
            resp = self._client.post("/chat/completions", json=payload, **post_kw)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"].get("content") or ""
            if resp.status_code in (408, 409, 429) or resp.status_code >= 500:
                if attempt == self.max_retries:
                    resp.raise_for_status()
                retry_after = resp.headers.get("retry-after")
                backoff = float(retry_after) if retry_after else delay
                if deadline is not None and self._clock() + backoff >= deadline:
                    raise TimeoutError("narrator deadline reached before retry")
                self._sleep(backoff)
                delay = min(delay * 2, 30.0)
                continue
            resp.raise_for_status()
        raise RuntimeError("unreachable")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": 16,
            "messages": [{"role": "user", "content": prompt}],
        }
        return self._chat(payload, None).strip()

    def complete_text(self, prompt: str, *, max_tokens: int,
                      deadline: float | None = None) -> str:
        """Whole reply text, unparsed, at temperature 0. An HTTP timeout
        (per attempt or capped by the deadline) is raised as TimeoutError."""
        import httpx
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            return self._chat(payload, deadline)
        except httpx.TimeoutException as e:
            raise TimeoutError(str(e) or "request timed out") from e


def get_backend(name: str | None = None, **kw) -> Backend:
    """Backend named by `name`, else by AGENTIC_BACKEND at call time (so a
    service reads the env it was started with, not the import-time value)."""
    import os
    name = name or os.environ.get("AGENTIC_BACKEND", C.BACKEND)
    if name == "mock":
        mock_kw = {k: kw[k] for k in ("mode", "delay_s", "sleep") if k in kw}
        return MockBackend(**mock_kw)
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
