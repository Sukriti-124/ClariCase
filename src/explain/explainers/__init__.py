"""Shared section helpers for the explainers (design §1).

* ``SECTION_NAMES`` (re-exported from ``record``) and ``NOMINAL``, the
  ``(method, technique)`` pair every section carries, even when skipped or
  errored.
* ``skipped_section`` / ``error_section`` build non-``ok`` sections
  (Req 1.2, 1.4); ``run_explainer`` turns any exception into an ``error``
  section and never re-raises.
* ``select_drivers``: the shared top-N rule (Req 3.2, 3.3, 4.2).
* ``validate_n``: the per-explainer N range, checked at configuration time.

Explainer modules are imported explicitly (``from src.explain.explainers
import im1``); importing this package pulls in no model code.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Sequence

from src.explain.record import SECTION_MODELS, SECTION_NAMES, Driver, Section, make_section

__all__ = ["SECTION_NAMES", "NOMINAL", "N_RANGE", "DEFAULT_N", "validate_n",
           "skipped_section", "error_section", "error_reason", "run_explainer",
           "select_drivers"]

log = logging.getLogger(__name__)

NOMINAL: dict[str, tuple[str, str]] = {
    "improved_model_1": ("exact", "linear"),
    "baseline_model_2": ("exact", "nb_llr"),
    "improved_model_2": ("approximate", "occlusion"),
    "knn_embed": ("exact", "neighbours"),
    "stacker": ("exact", "linear"),
    "gate": ("approximate", "ablation"),
    "routing": ("exact", "rules"),
    "agentic": ("exact", "rules"),
}
assert tuple(NOMINAL) == SECTION_NAMES

# Allowed Driver count per sign, inclusive (Req 3.2, 4.2).
N_RANGE: dict[str, tuple[int, int]] = {
    "improved_model_1": (1, 100),
    "baseline_model_2": (1, 50),
}
DEFAULT_N = 10
REASON_MSG_MAX = 500


def validate_n(name: str, n: Any) -> int:
    """Return ``n`` if it is an int in the explainer's range, else ValueError.

    Called once when an explainer is configured, not per request.
    """
    if name not in N_RANGE:
        raise ValueError(f"explainer {name!r} has no configurable N")
    lo, hi = N_RANGE[name]
    if isinstance(n, bool) or not isinstance(n, int) or not lo <= n <= hi:
        raise ValueError(f"N for {name} must be an integer from {lo} to {hi}, got {n!r}")
    return n


def _section(name: str, status: str, reason: str, details: dict) -> Section:
    method, technique = NOMINAL[name]
    return make_section(name, status=status, reason=reason, method=method,
                        technique=technique, details=details or {})


def skipped_section(name: str, reason: str, **details: Any) -> Section:
    """A ``skipped`` section with zero Drivers and optional partial details."""
    if name not in NOMINAL:
        raise ValueError(f"unknown section name {name!r}")
    if not (reason and reason.strip()):
        raise ValueError("a skipped section needs a non-empty reason")
    return _section(name, "skipped", reason, details)


def error_reason(exc: BaseException) -> str:
    """``"<Type>: <message[:500]>"`` (Req 1.4)."""
    try:
        msg = str(exc)
    except Exception:                    # a broken __str__ must not escape
        msg = "<unprintable message>"
    return f"{type(exc).__name__}: {msg[:REASON_MSG_MAX]}"


def error_section(name: str, exc: BaseException, **details: Any) -> Section:
    """An ``error`` section for ``exc``. Details that fail validation are
    dropped so the section itself is always produced."""
    if name not in NOMINAL:
        raise ValueError(f"unknown section name {name!r}")
    reason = error_reason(exc)
    if details:
        try:
            return _section(name, "error", reason, details)
        except Exception:
            log.warning("dropping invalid details for %s error section", name)
    return _section(name, "error", reason, {})


def run_explainer(name: str, fn: Callable[..., Section], *args: Any, **kw: Any) -> Section:
    """Call ``fn(*args, **kw)``; any Exception becomes an ``error`` section.

    A return value that is not this name's section model is also an error.
    Never re-raises.
    """
    try:
        result = fn(*args, **kw)
        expected = SECTION_MODELS[name]
        if not isinstance(result, expected):
            raise TypeError(f"{name} explainer returned {type(result).__name__}, "
                            f"expected {expected.__name__}")
        return result
    except Exception as exc:
        log.warning("explainer %s failed: %s", name, type(exc).__name__)
        return error_section(name, exc)


def select_drivers(contribs: Sequence[float], labels: Sequence[str],
                   values: Sequence[float | None] | None, n: int,
                   *, kind: str = "ngram") -> list[Driver]:
    """Up to ``n`` positive and up to ``n`` negative Drivers.

    Exact zeros are dropped; there is no padding (Req 3.3). The result is
    ordered by ``(-abs(contribution), label)``, which within one sign is also
    the BM2 order of Req 4.2. A positive and a negative Driver with the same
    ``abs`` and label are ordered positive first. ``kind="sentence"`` stores the label as the
    Driver's ``text``; other kinds store it as ``label``.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    if len(labels) != len(contribs) or (values is not None and len(values) != len(contribs)):
        raise ValueError("contribs, labels and values must have the same length")
    items = []
    for i, c in enumerate(contribs):
        c = float(c)
        if not math.isfinite(c):
            raise ValueError(f"non-finite contribution for {labels[i]!r}")
        if c != 0.0:
            items.append((-abs(c), str(labels[i]), i, c))
    items.sort(key=lambda t: (t[0], t[1]))
    pos = [t for t in items if t[3] > 0][:n]
    neg = [t for t in items if t[3] < 0][:n]
    out = []
    # Equal |c| and label on both signs: positive first, so the order is total.
    for _, label, i, c in sorted(pos + neg, key=lambda t: (t[0], t[1], -t[3])):
        v = None if values is None or values[i] is None else float(values[i])
        if kind == "sentence":
            out.append(Driver(kind=kind, text=label, value=v, contribution=c))
        else:
            out.append(Driver(kind=kind, label=label, value=v, contribution=c))
    return out
