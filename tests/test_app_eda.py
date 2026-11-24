"""Admin dashboard aggregations and the store's fetch_recent()."""

from datetime import datetime, timezone

import pytest

from src.app import eda
from src.app.storage import LocalStore

NOW = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)


def row(tid, day, team_id, team, route, conf, issue="Bank :: Managing an account",
        text="My checking account was charged an overdraft fee XXXX",
        explanation=None):
    return {"tracking_id": tid, "submitted_at": f"2026-11-{day:02d}T10:00:00+00:00",
            "complaint_text": text, "team_id": team_id, "team_name": team,
            "confidence": conf,
            "status": {"auto": "Routed", "review": "Under review"}.get(route, "Received"),
            "issue_id": "I1", "issue_label": issue, "route": route,
            "explanation": explanation}


@pytest.fixture
def df():
    return eda.to_frame([
        row("CC-1", 19, "T04", "Banking Accounts Team", "auto", 0.95,
            explanation={"source": "llm"}),
        row("CC-2", 19, "T04", "Banking Accounts Team", "review", 0.60,
            explanation={"source": "template"}),
        row("CC-3", 10, "T01", "Credit Reporting Team", "auto", 0.90,
            issue="Credit reporting :: Incorrect information on your report",
            text="Incorrect information on my credit report account XXXX"),
        row("CC-4", 1, "T01", "Credit Reporting Team", None, 0.40),
    ])


def test_to_frame_infers_route_and_issue(df):
    old = df[df["tracking_id"] == "CC-4"].iloc[0]
    assert old["route"] == "unknown" and old["routing"] == "Unknown"
    assert df.iloc[0]["issue"] == "Managing an account"
    assert df.iloc[0]["words"] == 9


def test_kpis(df):
    k = eda.kpis(df, now=NOW)
    assert (k["total"], k["auto"], k["review"]) == (4, 2, 1)
    assert k["last_7_days"] == 2
    assert k["auto_rate"] == pytest.approx(0.5)
    assert k["avg_confidence"] == pytest.approx((0.95 + 0.6 + 0.9 + 0.4) / 4)


def test_volume_and_by_team(df):
    v = eda.volume(df, "D")
    assert v["complaints"].sum() == 4
    day = v[v["period"] == "2026-11-19"]
    assert set(day["routing"]) == {"Auto-routed", "Needs review"}

    t = eda.by_team(df)
    assert t.groupby("team_name")["complaints"].sum().to_dict() == {
        "Banking Accounts Team": 2, "Credit Reporting Team": 2}


def test_top_issues(df):
    issues = eda.top_issues(df, n=1)
    assert issues.iloc[0].to_dict() == {"issue": "Managing an account",
                                        "complaints": 3}


def test_team_mix_shares_sum_to_one(df):
    mix = eda.team_mix(df, {"T01": 75, "T04": 25},
                       {"T01": "Credit Reporting Team", "T04": "Banking Accounts Team"})
    for _, grp in mix.groupby("source"):
        assert grp["share"].sum() == pytest.approx(1.0)
    train = mix[(mix["source"] == "Training data")
                & (mix["team_name"] == "Credit Reporting Team")]
    assert train["share"].iloc[0] == pytest.approx(0.75)


def test_top_phrases_ignore_masked_tokens(df):
    phrases = dict(eda.top_phrases(df["complaint_text"], n=20))
    assert phrases["checking account"] == 3
    assert not any("xxxx" in p for p in phrases)
    assert eda.top_phrases(df["complaint_text"].iloc[:0]) == []


def test_explanation_coverage_and_queue(df):
    cov = eda.explanation_coverage(df)
    assert cov == {"with_explanation": 2, "ai_written": 1, "standard": 1,
                   "total": 4}
    q = eda.review_queue(df)
    assert list(q["tracking_id"]) == ["CC-2"]


def test_committed_metadata_loads():
    assert eda.routing_threshold() == pytest.approx(0.818)
    counts = eda.training_team_counts()
    assert set(counts) >= {"T01", "T11"}
    assert eda.team_names()["T04"] == "Banking Accounts Team"


def test_local_store_fetch_recent(tmp_path):
    store = LocalStore(tmp_path / "c.db")
    for r in [row("CC-A", 1, "T04", "Banking Accounts Team", "auto", 0.9),
              row("CC-B", 5, "T04", "Banking Accounts Team", "review", 0.5)]:
        store.add(r)
    store.update("CC-B", {"explanation": {"source": "template"}})
    rows = store.fetch_recent()
    assert [r["tracking_id"] for r in rows] == ["CC-B", "CC-A"]
    assert rows[0]["explanation"] == {"source": "template"}
    assert len(store.fetch_recent(limit=1)) == 1


def test_assigned_complaints_leave_the_queue():
    done = {**row("CC-5", 18, "T03", "Credit Card Team", "review", 0.30),
            "status": "Assigned", "model_team_id": "T04",
            "model_team_name": "Banking Accounts Team",
            "reviewed_at": "2026-11-19T09:00:00+00:00"}
    frame = eda.to_frame([
        row("CC-2", 19, "T04", "Banking Accounts Team", "review", 0.60), done])
    k = eda.kpis(frame, now=NOW)
    assert (k["review"], k["pending"]) == (2, 1)
    assert list(eda.review_queue(frame)["tracking_id"]) == ["CC-2"]
    assert frame.loc[1, "reviewed_at"].day == 19
