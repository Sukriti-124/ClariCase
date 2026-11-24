"""Bulk complaint routing for the intake app.

Reads an uploaded CSV (or Excel, when openpyxl is installed), routes every
row the same way a single complaint is routed (mask, predict, build the
stored record), and summarises and exports the results. Streamlit-free so
it can be tested directly.
"""

from __future__ import annotations

import io
from collections import Counter
from typing import Callable, Iterable

import pandas as pd

from src.app.pii import mask
from src.app.storage import build_record

MAX_ROWS = 2000
MIN_WORDS = 5
EXCERPT_CHARS = 140

# Checked in order, against lower-cased column names.
TEXT_HINTS = ("consumer_complaint_narrative", "complaint_text", "narrative",
              "complaint", "description", "text", "details", "message")
REF_HINTS = ("case_ref", "reference", "ref", "case_id", "ticket_id",
             "complaint_id", "id")

TEMPLATE_CSV = (
    "case_ref,narrative\n"
    'CRM-1001,"I was charged an overdraft fee even though my paycheck had '
    'already posted. I would like the fee refunded."\n'
    'CRM-1002,"There is an account on my credit report that I never opened '
    'and the bureau will not remove it."\n'
)


class BulkError(ValueError):
    """A problem with the uploaded file, worded for the user."""


def excel_supported() -> bool:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return False
    return True


def read_upload(name: str, data: bytes) -> pd.DataFrame:
    """Parse an uploaded file into a DataFrame of strings."""
    lower = name.lower()
    try:
        if lower.endswith((".xlsx", ".xls")):
            if not excel_supported():
                raise BulkError("Excel files aren't supported here yet. "
                                "Please save the sheet as CSV.")
            df = pd.read_excel(io.BytesIO(data), dtype=str)
        else:
            df = pd.read_csv(io.BytesIO(data), dtype=str,
                             keep_default_na=False, encoding_errors="replace")
    except BulkError:
        raise
    except Exception as exc:
        raise BulkError("We couldn't read this file. Please upload a CSV "
                        "with one complaint per row.") from exc
    df = df.fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty or not len(df.columns):
        raise BulkError("This file has no rows.")
    if len(df) > MAX_ROWS:
        raise BulkError(f"This file has {len(df):,} rows. Please upload at "
                        f"most {MAX_ROWS:,} at a time.")
    return df


def _match(columns: list[str], hints: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    return None


def guess_text_column(df: pd.DataFrame) -> str:
    """The column most likely to hold the complaint text."""
    named = _match(list(df.columns), TEXT_HINTS)
    if named is not None:
        return named
    lengths = {c: df[c].astype(str).str.len().mean() for c in df.columns}
    return max(lengths, key=lengths.get)


def guess_ref_column(df: pd.DataFrame, text_column: str) -> str | None:
    """A reference/ID column, if one is recognisable."""
    named = _match([c for c in df.columns if c != text_column], REF_HINTS)
    return named


def issue_name(label: str) -> str:
    # "Debt collection :: Communication tactics" -> "Communication tactics"
    return label.split(" :: ", 1)[-1] if label else ""


def _excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EXCERPT_CHARS else \
        text[:EXCERPT_CHARS - 1].rstrip() + "…"


def route_rows(texts: list[str], refs: list[str] | None,
               predict_fn: Callable[[str], dict],
               on_progress: Callable[[int, int], None] | None = None
               ) -> tuple[list[dict], list[dict], list[dict]]:
    """Route each text. Returns (records to store, result rows, skipped rows).

    Texts shorter than MIN_WORDS are skipped, as in the single form. Text is
    masked before prediction, so nothing downstream sees raw personal details.
    """
    records, results, skipped = [], [], []
    total = len(texts)
    for i, raw in enumerate(texts):
        ref = (refs[i] if refs else "") or ""
        text = str(raw).strip()
        if len(text.split()) < MIN_WORDS:
            skipped.append({"row": i + 1, "reference": ref})
        else:
            masked = mask(text)
            prediction = predict_fn(masked)
            record = build_record(masked, prediction)
            records.append(record)
            results.append({
                "row": i + 1,
                "reference": ref,
                "tracking_id": record["tracking_id"],
                "team": record["team_name"],
                "issue": issue_name(record["issue_label"] or ""),
                "confidence": record["confidence"],
                "status": record["status"],
                "complaint": _excerpt(masked),
            })
        if on_progress is not None:
            on_progress(i + 1, total)
    return records, results, skipped


def summarise(results: list[dict]) -> dict:
    teams = Counter(r["team"] for r in results)
    auto = sum(r["status"] == "Routed" for r in results)
    return {
        "routed": len(results),
        "auto": auto,
        "review": len(results) - auto,
        "teams": teams.most_common(),
    }


def export_csv(results: list[dict]) -> bytes:
    cols = ["row", "reference", "tracking_id", "team", "issue", "confidence",
            "status", "complaint"]
    df = pd.DataFrame(results, columns=cols)
    return df.to_csv(index=False).encode("utf-8")
