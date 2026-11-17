"""LocalStore round trip and explanation update (Req 18.5)."""

import pytest

from src.app.storage import LocalStore, build_record

PREDICTION = {
    "predicted_team_id": "T01",
    "predicted_team_name": "Credit Reporting",
    "confidence": 0.91234,
    "route": "auto",
    "issue_top_k": [{"issue_id": "I1", "issue_label": "Incorrect info"}],
}


def test_update_writes_json_columns(tmp_path):
    store = LocalStore(tmp_path / "c.db")
    record = build_record("My report is wrong.", PREDICTION)
    store.add(record)
    tid = record["tracking_id"]

    assert store.get(tid)["explanation"] is None

    store.update(tid, {"explanation": {"consumer_summary": "Sent to X."},
                       "explanation_audit": [{"source": "template"}]})
    row = store.get(tid)
    assert row["explanation"] == {"consumer_summary": "Sent to X."}
    assert row["explanation_audit"] == [{"source": "template"}]
    assert row["decision_record"] is None
    assert row["team_id"] == "T01"


def test_update_rejects_unknown_columns(tmp_path):
    store = LocalStore(tmp_path / "c.db")
    record = build_record("text", PREDICTION)
    store.add(record)
    with pytest.raises(ValueError):
        store.update(record["tracking_id"], {"team_id": "T99"})
    assert store.get(record["tracking_id"])["team_id"] == "T01"
