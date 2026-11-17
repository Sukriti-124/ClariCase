"""Shared fixtures for the explainability test suite.

Three guarantees hold for every test under ``tests/explain/``:

* No network. ``socket.socket.connect`` (plus ``connect_ex``) and
  ``socket.create_connection`` raise ``NetworkBlocked``. A test that triggers
  the guard is reported as failed even if it catches the exception (Req 21.7).
  ``socket.socketpair`` stays usable because on Windows it is implemented with
  a loopback ``connect`` and asyncio event loops need it for their self-pipe.
* A clean environment. Every ``EXPLAIN_*``, ``AGENTIC_*`` and ``AIC_*``
  variable plus ``GROQ_API_KEY`` and ``OPENAI_API_KEY`` is removed and
  ``EXPLAIN_BACKEND=mock`` is set, so no test can reach a paid backend.
* A ``fake_clock`` fixture with injectable ``monotonic()`` and ``sleep()``.
"""

from __future__ import annotations

import os
import socket
import threading
from dataclasses import dataclass, field

import pytest

_ENV_PREFIXES = ("EXPLAIN_", "AGENTIC_", "AIC_")
_ENV_NAMES = ("GROQ_API_KEY", "OPENAI_API_KEY")

_NETWORK_ATTEMPTS_KEY = pytest.StashKey[list]()

_bypass = threading.local()


class NetworkBlocked(RuntimeError):
    """Raised when a test tries to open a network connection."""


def _bypassed() -> bool:
    return getattr(_bypass, "depth", 0) > 0


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    attempts: list[str] = []
    request.node.stash[_NETWORK_ATTEMPTS_KEY] = attempts

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_socketpair = socket.socketpair

    def _block(what: str, address: object) -> None:
        msg = f"{what} to {address!r} blocked in tests/explain"
        attempts.append(msg)
        raise NetworkBlocked(msg)

    def guarded_connect(self, address):
        if _bypassed():
            return real_connect(self, address)
        _block("socket.connect", address)

    def guarded_connect_ex(self, address):
        if _bypassed():
            return real_connect_ex(self, address)
        _block("socket.connect_ex", address)

    def guarded_create_connection(address, *args, **kwargs):
        _block("socket.create_connection", address)

    def guarded_socketpair(*args, **kwargs):
        _bypass.depth = getattr(_bypass, "depth", 0) + 1
        try:
            return real_socketpair(*args, **kwargs)
        finally:
            _bypass.depth -= 1

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "socketpair", guarded_socketpair)
    yield attempts


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or report.failed:
        return
    attempts = item.stash.get(_NETWORK_ATTEMPTS_KEY, None)
    if attempts:
        report.outcome = "failed"
        report.longrepr = "Network access attempted during test:\n" + "\n".join(
            f"  - {a}" for a in attempts
        )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(_ENV_PREFIXES) or name in _ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EXPLAIN_BACKEND", "mock")


@dataclass
class FakeClock:
    """Deterministic clock. ``sleep(s)`` advances time instead of blocking."""

    now: float = 1000.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep length must be non-negative")
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        """Move time forward without recording a sleep (e.g. attempt duration)."""
        if seconds < 0:
            raise ValueError("cannot move the clock backwards")
        self.now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()
