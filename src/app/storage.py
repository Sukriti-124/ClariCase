"""Complaint storage for the intake app.

Supabase when deployed; a local SQLite file for development when no
Supabase secrets are configured. Both expose add(), add_many(), get(),
update(), assign() and fetch_recent().
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.app.pii import mask

TABLE = "complaints"
LOCAL_DB = Path(__file__).resolve().parents[2] / "data" / "app" / "complaints.db"

# No 0/O or 1/I, so IDs are easy to read back.
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
COLUMNS = ["tracking_id", "submitted_at", "complaint_text", "team_id",
           "team_name", "confidence", "status", "issue_id", "issue_label",
           "route"]
# Added after the first release; nullable so older rows stay valid.
ADDED_COLUMNS = {"issue_id": "TEXT", "issue_label": "TEXT", "route": "TEXT",
                 "decision_record": "TEXT", "explanation": "TEXT",
                 "explanation_audit": "TEXT",
                 # Set when an admin assigns a "needs review" complaint.
                 "model_team_id": "TEXT", "model_team_name": "TEXT",
                 "reviewed_at": "TEXT", "review_note": "TEXT"}
# Columns update() may write. JSON text in SQLite, jsonb in Supabase.
JSON_COLUMNS = ("decision_record", "explanation", "explanation_audit")
UPDATABLE_COLUMNS = frozenset(JSON_COLUMNS)
STATUS_BY_ROUTE = {"auto": "Routed", "review": "Under review"}
ASSIGNED = "Assigned"
REVIEW_NOTE_MAX = 500
# Columns the admin dashboard reads; the large decision record is left out.
DASHBOARD_COLUMNS = ["tracking_id", "submitted_at", "complaint_text", "team_id",
                     "team_name", "confidence", "status", "issue_id",
                     "issue_label", "route", "explanation"]
REVIEW_COLUMNS = ["model_team_id", "model_team_name", "reviewed_at",
                  "review_note"]
DASHBOARD_MAX_ROWS = 50_000
PAGE_SIZE = 1000   # Supabase returns at most 1,000 rows per request.


def new_tracking_id() -> str:
    # Random, not sequential, so one user cannot guess another's ID.
    return "CC-" + "".join(secrets.choice(ALPHABET) for _ in range(6))


def normalise_tracking_id(raw: str) -> str:
    tid = raw.strip().upper()
    return tid if tid.startswith("CC-") else f"CC-{tid}"


def assignment_fields(row: dict, team_id: str, team_name: str,
                      note: str | None = None,
                      now: datetime | None = None) -> dict:
    """Columns to write when an admin assigns a complaint to a team.

    The model's original choice is kept in model_team_*; if the complaint
    was already assigned once, the first recorded model choice is preserved.
    """
    if not team_id or not team_name:
        raise ValueError("a team is required")
    now = now or datetime.now(timezone.utc)
    note = (note or "").strip()[:REVIEW_NOTE_MAX] or None
    return {
        "team_id": team_id,
        "team_name": team_name,
        "status": ASSIGNED,
        "model_team_id": row.get("model_team_id") or row["team_id"],
        "model_team_name": row.get("model_team_name") or row["team_name"],
        "reviewed_at": now.isoformat(timespec="seconds"),
        "review_note": note,
    }


def build_record(text: str, prediction: dict) -> dict:
    top_issue = (prediction.get("issue_top_k") or [{}])[0]
    route = prediction.get("route")
    return {
        "tracking_id": new_tracking_id(),
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Masked here as well as at intake, so no caller can write raw text.
        # mask() is idempotent, so the second pass is a no-op.
        "complaint_text": mask(text),
        "team_id": prediction["predicted_team_id"],
        "team_name": prediction["predicted_team_name"],
        "confidence": round(prediction["confidence"], 4),
        "status": STATUS_BY_ROUTE.get(route, "Received"),
        "issue_id": top_issue.get("issue_id"),
        "issue_label": top_issue.get("issue_label"),
        "route": route,
    }


class SupabaseStore:
    persistent = True

    def __init__(self, url: str, key: str):
        from supabase import create_client
        self.client = create_client(url, key)

    def add(self, record: dict) -> None:
        self.client.table(TABLE).insert(record).execute()

    def add_many(self, records: list[dict]) -> None:
        # One request, so the batch is saved whole or not at all.
        if records:
            self.client.table(TABLE).insert(records).execute()

    def get(self, tracking_id: str) -> dict | None:
        rows = (self.client.table(TABLE).select("*")
                .eq("tracking_id", tracking_id).limit(1).execute().data)
        return rows[0] if rows else None

    def update(self, tracking_id: str, fields: dict) -> None:
        _check_fields(fields)
        (self.client.table(TABLE).update(fields)
         .eq("tracking_id", tracking_id).execute())

    def assign(self, tracking_id: str, team_id: str, team_name: str,
               note: str | None = None) -> dict:
        row = self.get(tracking_id)
        if row is None:
            raise KeyError(tracking_id)
        fields = assignment_fields(row, team_id, team_name, note)
        (self.client.table(TABLE).update(fields)
         .eq("tracking_id", tracking_id).execute())
        return {**row, **fields}

    def fetch_recent(self, limit: int = DASHBOARD_MAX_ROWS) -> list[dict]:
        """Newest complaints first, for the admin dashboard."""
        try:
            return self._fetch(DASHBOARD_COLUMNS + REVIEW_COLUMNS, limit)
        except Exception:
            # Review columns not migrated yet (see schema.sql).
            return self._fetch(DASHBOARD_COLUMNS, limit)

    def _fetch(self, columns: list[str], limit: int) -> list[dict]:
        rows: list[dict] = []
        cols = ", ".join(columns)
        while len(rows) < limit:
            start = len(rows)
            end = min(start + PAGE_SIZE, limit) - 1      # inclusive
            page = (self.client.table(TABLE).select(cols)
                    .order("submitted_at", desc=True)
                    .range(start, end).execute().data)
            rows.extend(page)
            if len(page) < end - start + 1:               # last page
                break
        return rows


def _check_fields(fields: dict) -> None:
    unknown = set(fields) - UPDATABLE_COLUMNS
    if unknown:
        raise ValueError(f"cannot update columns: {sorted(unknown)}")


class LocalStore:
    persistent = False

    def __init__(self, path: Path = LOCAL_DB):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                tracking_id TEXT PRIMARY KEY,
                submitted_at TEXT NOT NULL,
                complaint_text TEXT NOT NULL,
                team_id TEXT NOT NULL,
                team_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'Received')""")
            # Bring databases created before ADDED_COLUMNS up to date.
            have = {r["name"] for r in con.execute(
                f"PRAGMA table_info({TABLE})")}
            for name, sql_type in ADDED_COLUMNS.items():
                if name not in have:
                    con.execute(f"ALTER TABLE {TABLE} ADD COLUMN {name} {sql_type}")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def add(self, record: dict) -> None:
        with self._connect() as con:
            con.execute(f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) "
                        f"VALUES ({', '.join('?' * len(COLUMNS))})",
                        [record[c] for c in COLUMNS])

    def add_many(self, records: list[dict]) -> None:
        # One transaction, so the batch is saved whole or not at all.
        with self._connect() as con:
            con.executemany(f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) "
                            f"VALUES ({', '.join('?' * len(COLUMNS))})",
                            [[r[c] for c in COLUMNS] for r in records])

    def get(self, tracking_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute(f"SELECT * FROM {TABLE} WHERE tracking_id = ?",
                              (tracking_id,)).fetchone()
        if not row:
            return None
        record = dict(row)
        # Decode JSON text so both stores return the same shapes.
        for name in JSON_COLUMNS:
            if record.get(name) is not None:
                record[name] = json.loads(record[name])
        return record

    def update(self, tracking_id: str, fields: dict) -> None:
        _check_fields(fields)
        if not fields:
            return
        # Column names come from the allow-list, never from the caller.
        names = [c for c in JSON_COLUMNS if c in fields]
        values = [None if fields[c] is None else json.dumps(fields[c])
                  for c in names]
        with self._connect() as con:
            con.execute(f"UPDATE {TABLE} SET "
                        f"{', '.join(f'{c} = ?' for c in names)} "
                        f"WHERE tracking_id = ?", [*values, tracking_id])

    def assign(self, tracking_id: str, team_id: str, team_name: str,
               note: str | None = None) -> dict:
        row = self.get(tracking_id)
        if row is None:
            raise KeyError(tracking_id)
        fields = assignment_fields(row, team_id, team_name, note)
        with self._connect() as con:
            con.execute(f"UPDATE {TABLE} SET "
                        f"{', '.join(f'{c} = ?' for c in fields)} "
                        f"WHERE tracking_id = ?",
                        [*fields.values(), tracking_id])
        return {**row, **fields}

    def fetch_recent(self, limit: int = DASHBOARD_MAX_ROWS) -> list[dict]:
        """Newest complaints first, for the admin dashboard."""
        cols = DASHBOARD_COLUMNS + REVIEW_COLUMNS
        with self._connect() as con:
            rows = con.execute(
                f"SELECT {', '.join(cols)} FROM {TABLE} "
                f"ORDER BY submitted_at DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for row in rows:
            record = dict(row)
            if record.get("explanation") is not None:
                record["explanation"] = json.loads(record["explanation"])
            out.append(record)
        return out


def make_store(secrets) -> SupabaseStore | LocalStore:
    """Supabase when its secrets are configured, else the local SQLite file."""
    try:
        cfg = secrets["supabase"]
        return SupabaseStore(cfg["url"], cfg["key"])
    except (KeyError, FileNotFoundError, TypeError):
        return LocalStore()
