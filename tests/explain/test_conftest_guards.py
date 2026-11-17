"""Checks for the shared guards in tests/explain/conftest.py (Req 21.3, 21.7)."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.explain.conftest import NetworkBlocked

_CONFTEST = Path(__file__).with_name("conftest.py")


def test_env_is_cleared_and_backend_is_mock():
    assert os.environ["EXPLAIN_BACKEND"] == "mock"
    for name in os.environ:
        if name == "EXPLAIN_BACKEND":
            continue
        assert not name.startswith(("EXPLAIN_", "AGENTIC_", "AIC_")), name
    assert "GROQ_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_fake_clock_sleep_advances_without_blocking(fake_clock):
    start = fake_clock.monotonic()
    fake_clock.sleep(30.0)
    fake_clock.advance(2.5)
    assert fake_clock.monotonic() == start + 32.5
    assert fake_clock.sleeps == [30.0]


def test_socketpair_and_event_loop_still_work():
    a, b = socket.socketpair()
    a.close()
    b.close()
    assert asyncio.run(asyncio.sleep(0, result=7)) == 7


def test_guard_fails_a_test_that_swallows_network_errors(tmp_path):
    pkg = tmp_path / "guarded"
    pkg.mkdir()
    shutil.copy(_CONFTEST, pkg / "conftest.py")
    (pkg / "test_net.py").write_text(
        "import os, socket\n"
        "def test_swallowed():\n"
        "    try:\n"
        "        socket.create_connection(('example.com', 443))\n"
        "    except Exception:\n"
        "        pass\n"
        "def test_connect_raises():\n"
        "    s = socket.socket()\n"
        "    try:\n"
        "        s.connect(('127.0.0.1', 9))\n"
        "    finally:\n"
        "        s.close()\n"
        "def test_offline_passes():\n"
        "    assert os.environ['EXPLAIN_BACKEND'] == 'mock'\n",
        encoding="utf-8",
    )
    env = dict(os.environ, GROQ_API_KEY="leak", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(pkg)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert "2 failed, 1 passed" in out, out
    assert "Network access attempted" in out, out
    assert "NetworkBlocked" in out, out


def test_direct_connect_raises_network_blocked(_no_network):
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 80))
    # Clear the record so this intentional attempt does not fail the test.
    _no_network.clear()
