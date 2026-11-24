"""Aggregations for the admin dashboard (pages/admin.py).

Everything works on the rows the store returns, so it can be tested without
Streamlit. Complaint text is already masked when it is stored.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.app.bulk import issue_name

ROOT = Path(__file__).resolve().parents[2]
SPLIT_REPORT = ROOT / "docs" / "metrics" / "improved_model_1" / "split_report.json"
RUN_INFO = ROOT / "docs" / "metrics" / "improved_model_1" / "run_info.json"
LABEL_TABLES = ROOT / "src" / "explain" / "label_tables.json"

ASSIGNED = "Assigned"
ROUTE_LABELS = {"auto": "Auto-routed", "review": "Needs review"}


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """One row per complaint, with derived columns the charts use."""
    cols = ["tracking_id", "submitted_at", "complaint_text", "team_id",
            "team_name", "confidence", "status", "issue_id", "issue_label",
            "route", "explanation", "model_team_id", "model_team_name",
            "reviewed_at", "review_note"]
    df = pd.DataFrame(rows, columns=cols)
    df["reviewed_at"] = pd.to_datetime(df["reviewed_at"], utc=True,
                                       format="ISO8601")
    df["submitted_at"] = pd.to_datetime(df["submitted_at"], utc=True,
                                        format="ISO8601")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    # Rows saved before `route` existed: infer it from the status.
    inferred = df["status"].map({"Routed": "auto", "Under review": "review"})
    df["route"] = df["route"].fillna(inferred).fillna("unknown")
    df["routing"] = df["route"].map(ROUTE_LABELS).fillna("Unknown")
    df["issue"] = df["issue_label"].fillna("").map(issue_name)
    df["words"] = df["complaint_text"].fillna("").str.split().str.len()
    df["explanation_source"] = df["explanation"].map(
        lambda e: e.get("source") if isinstance(e, dict) else None)
    return df


def kpis(df: pd.DataFrame, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    total = len(df)
    auto = int((df["route"] == "auto").sum())
    review = int((df["route"] == "review").sum())
    pending = int(((df["route"] == "review")
                   & (df["status"] != ASSIGNED)).sum())
    last7 = int((df["submitted_at"] >= now - timedelta(days=7)).sum())
    return {
        "total": total,
        "last_7_days": last7,
        "auto": auto,
        "review": review,
        "pending": pending,
        "auto_rate": auto / total if total else 0.0,
        "avg_confidence": float(df["confidence"].mean()) if total else 0.0,
    }


def volume(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Complaints per period and routing outcome."""
    if df.empty:
        return pd.DataFrame(columns=["period", "routing", "complaints"])
    period = df["submitted_at"].dt.tz_convert(None).dt.to_period(freq) \
        .dt.start_time
    out = (df.assign(period=period).groupby(["period", "routing"])
           .size().rename("complaints").reset_index())
    return out


def by_team(df: pd.DataFrame) -> pd.DataFrame:
    """Complaints per team and routing outcome, largest team first."""
    out = (df.groupby(["team_name", "routing"]).size().rename("complaints")
           .reset_index())
    order = out.groupby("team_name")["complaints"].sum() \
        .sort_values(ascending=False)
    out["team_total"] = out["team_name"].map(order)
    return out.sort_values(["team_total", "team_name"], ascending=[False, True])


def top_issues(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    s = df.loc[df["issue"] != "", "issue"].value_counts().head(n)
    return s.rename_axis("issue").rename("complaints").reset_index()


def training_team_counts(path: Path = SPLIT_REPORT) -> dict[str, int]:
    """Complaints per team id in the training split."""
    try:
        return json.loads(path.read_text())["per_team"]["train_full"]
    except (OSError, KeyError, ValueError):
        return {}


def team_names(path: Path = LABEL_TABLES) -> dict[str, str]:
    try:
        teams = json.loads(path.read_text())["teams"]
        return {tid: t["name"] for tid, t in teams.items()}
    except (OSError, KeyError, ValueError):
        return {}


def routing_threshold(path: Path = RUN_INFO) -> float | None:
    try:
        return float(json.loads(path.read_text())["global_threshold"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def team_mix(df: pd.DataFrame, train_counts: dict[str, int],
             names: dict[str, str]) -> pd.DataFrame:
    """Each team's share of live complaints against its share in training."""
    live = df["team_id"].value_counts()
    train_total = sum(train_counts.values()) or 1
    live_total = int(live.sum()) or 1
    ids = sorted(set(train_counts) | set(live.index))
    rows = []
    for tid in ids:
        name = names.get(tid, tid)
        rows.append({"team_name": name, "source": "Live complaints",
                     "share": live.get(tid, 0) / live_total})
        rows.append({"team_name": name, "source": "Training data",
                     "share": train_counts.get(tid, 0) / train_total})
    return pd.DataFrame(rows)


_MASK = re.compile(r"\bX{2,}\b|\{\$?[\d.,]+\}|\d+", re.IGNORECASE)


def top_phrases(texts: pd.Series, n: int = 15) -> list[tuple[str, int]]:
    """Most frequent two- and three-word phrases, ignoring masked tokens."""
    from sklearn.feature_extraction.text import CountVectorizer
    docs = [_MASK.sub(" ", t) for t in texts.fillna("") if t.strip()]
    if not docs:
        return []
    vec = CountVectorizer(ngram_range=(2, 3), stop_words="english",
                          min_df=1, max_features=5000)
    try:
        counts = vec.fit_transform(docs).sum(axis=0).A1
    except ValueError:              # every document was only stop words
        return []
    vocab = vec.get_feature_names_out()
    order = counts.argsort()[::-1][:n]
    return [(str(vocab[i]), int(counts[i])) for i in order]


def explanation_coverage(df: pd.DataFrame) -> dict:
    total = len(df)
    src = df["explanation_source"]
    return {
        "with_explanation": int(src.notna().sum()),
        "ai_written": int((src == "llm").sum()),
        "standard": int((src == "template").sum()),
        "total": total,
    }


def pending_review(df: pd.DataFrame) -> pd.DataFrame:
    """Every column of the complaints still waiting for a team, least
    confident first."""
    waiting = (df["route"] == "review") & (df["status"] != ASSIGNED)
    return df[waiting].sort_values("confidence", kind="stable")


def review_queue(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Complaints waiting for review, least confident first."""
    q = pending_review(df).head(n)
    return q[["tracking_id", "submitted_at", "team_name", "issue",
              "confidence", "complaint_text"]]


def export_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[["tracking_id", "submitted_at", "team_id", "team_name",
               "issue", "confidence", "routing", "status", "words",
               "explanation_source", "complaint_text"]]
