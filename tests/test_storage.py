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


REVIEW_PREDICTION = {**PREDICTION, "route": "review", "confidence": 0.41}


def test_assign_moves_complaint_and_keeps_model_choice(tmp_path):
    store = LocalStore(tmp_path / "c.db")
    record = build_record("Card charged twice.", REVIEW_PREDICTION)
    store.add(record)
    tid = record["tracking_id"]

    store.assign(tid, "T03", "Credit Card Team", "  Card dispute.  ")
    row = store.get(tid)
    assert (row["team_id"], row["team_name"], row["status"]) == \
        ("T03", "Credit Card Team", "Assigned")
    assert (row["model_team_id"], row["model_team_name"]) == \
        ("T01", "Credit Reporting")
    assert row["review_note"] == "Card dispute."
    assert row["reviewed_at"] and row["route"] == "review"

    # Reassigning keeps the model's original choice, not the first admin's.
    store.assign(tid, "T04", "Banking Accounts Team")
    row = store.get(tid)
    assert row["team_id"] == "T04" and row["model_team_id"] == "T01"
    assert row["review_note"] is None

    recent = {r["tracking_id"]: r for r in store.fetch_recent()}
    assert recent[tid]["model_team_name"] == "Credit Reporting"


def test_assign_requires_a_known_complaint_and_team(tmp_path):
    store = LocalStore(tmp_path / "c.db")
    record = build_record("text", REVIEW_PREDICTION)
    store.add(record)
    with pytest.raises(KeyError):
        store.assign("CC-NOPE00", "T03", "Credit Card Team")
    with pytest.raises(ValueError):
        store.assign(record["tracking_id"], "", "")
    assert store.get(record["tracking_id"])["status"] == "Under review"


def test_assignment_note_is_trimmed_and_capped():
    from src.app.storage import REVIEW_NOTE_MAX, assignment_fields
    row = {"team_id": "T01", "team_name": "A"}
    assert assignment_fields(row, "T02", "B", "   ")["review_note"] is None
    long = assignment_fields(row, "T02", "B", "x" * 900)["review_note"]
    assert len(long) == REVIEW_NOTE_MAX
