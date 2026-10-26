"""Complaint storage for the intake app.

Supabase when deployed; a local SQLite file for development when no
Supabase secrets are configured. Both expose add() and get().
"""

from __future__ import annotations

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
ADDED_COLUMNS = {"issue_id": "TEXT", "issue_label": "TEXT", "route": "TEXT"}
STATUS_BY_ROUTE = {"auto": "Routed", "review": "Under review"}


def new_tracking_id() -> str:
    # Random, not sequential, so one user cannot guess another's ID.
    return "CC-" + "".join(secrets.choice(ALPHABET) for _ in range(6))


def normalise_tracking_id(raw: str) -> str:
    tid = raw.strip().upper()
    return tid if tid.startswith("CC-") else f"CC-{tid}"


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

    def get(self, tracking_id: str) -> dict | None:
        rows = (self.client.table(TABLE).select("*")
                .eq("tracking_id", tracking_id).limit(1).execute().data)
        return rows[0] if rows else None


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

    def get(self, tracking_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute(f"SELECT * FROM {TABLE} WHERE tracking_id = ?",
                              (tracking_id,)).fetchone()
        return dict(row) if row else None
