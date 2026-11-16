"""JSON logging with per-request context via contextvar.

Services call `configure_logging(service_name)` once at startup. Any
HTTP middleware (or manual code) that generates a request_id calls
`bind_request_id(rid)` so log lines downstream in the same async task
include it.
"""
from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar

import structlog

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def bind_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


def _add_request_id(_logger, _method_name, event_dict):
    rid = _request_id.get()
    if rid is not None:
        event_dict["request_id"] = rid
    return event_dict


def _add_service(service_name: str):
    def processor(_logger, _method_name, event_dict):
        event_dict["service"] = service_name
        return event_dict
    return processor


def configure_logging(service_name: str) -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_service(service_name),
            _add_request_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Mirror stdlib logging through structlog so `logging.getLogger(...).info()`
    # also lands in JSON.
    class _StructlogHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self._log = structlog.get_logger()

        def emit(self, record: logging.LogRecord) -> None:
            extras = {
                k: v for k, v in record.__dict__.items()
                if k not in logging.LogRecord(
                    "", 0, "", 0, "", None, None
                ).__dict__ and k != "message"
            }
            self._log.log(record.levelno, record.getMessage(), **extras)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(_StructlogHandler())
    root.setLevel(level)
