import io
import json
import logging

import pytest

from claricase_platform_shared.logging import (
    bind_request_id,
    configure_logging,
    get_request_id,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    logging.getLogger().handlers.clear()
    bind_request_id(None)
    yield
    logging.getLogger().handlers.clear()
    bind_request_id(None)


def test_configure_logging_writes_json_lines(capsys):
    configure_logging(service_name="svm-api")
    logger = logging.getLogger("svm-api")
    logger.info("hello", extra={"k": "v"})
    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 1
    row = json.loads(captured[0])
    assert row["event"] == "hello"
    assert row["service"] == "svm-api"
    assert row["k"] == "v"


def test_bind_request_id_propagates_through_contextvar(capsys):
    configure_logging(service_name="router")
    bind_request_id("req-42")
    assert get_request_id() == "req-42"
    logging.getLogger("router").info("routed")
    row = json.loads(capsys.readouterr().out.strip().splitlines()[0])
    assert row["request_id"] == "req-42"


def test_request_id_absent_when_unbound(capsys):
    configure_logging(service_name="router")
    logging.getLogger("router").info("no-ctx")
    row = json.loads(capsys.readouterr().out.strip().splitlines()[0])
    assert "request_id" not in row or row["request_id"] is None
