"""Bulk upload: parsing, routing, summary, export and batch saving."""

import pandas as pd
import pytest

from src.app import bulk
from src.app.storage import LocalStore


def fake_predict(text: str) -> dict:
    card = "card" in text.lower()
    return {
        "predicted_team_id": "T03" if card else "T04",
        "predicted_team_name": "Credit Card Team" if card
        else "Banking Accounts Team",
        "confidence": 0.95 if card else 0.55,
        "route": "auto" if card else "review",
        "issue_top_k": [{"issue_id": "I1",
                         "issue_label": "Credit card :: Fees or interest"}],
    }


def test_read_upload_parses_csv():
    df = bulk.read_upload("x.csv", bulk.TEMPLATE_CSV.encode())
    assert list(df.columns) == ["case_ref", "narrative"]
    assert len(df) == 2


def test_read_upload_rejects_too_many_rows():
    data = "narrative\n" + "a b c d e\n" * (bulk.MAX_ROWS + 1)
    with pytest.raises(bulk.BulkError, match="at most"):
        bulk.read_upload("x.csv", data.encode())


def test_read_upload_rejects_empty():
    with pytest.raises(bulk.BulkError):
        bulk.read_upload("x.csv", b"narrative\n")


def test_guess_columns_by_name_and_length():
    df = pd.DataFrame({"case_ref": ["A"], "narrative": ["long text here"]})
    assert bulk.guess_text_column(df) == "narrative"
    assert bulk.guess_ref_column(df, "narrative") == "case_ref"

    df = pd.DataFrame({"a": ["x"], "b": ["a much longer value"]})
    assert bulk.guess_text_column(df) == "b"
    assert bulk.guess_ref_column(df, "b") is None


def test_route_rows_masks_skips_and_reports_progress():
    texts = ["My credit card 4111 1111 1111 1111 was charged twice today",
             "too short",
             "My checking account was closed without any notice at all"]
    seen = []
    records, results, skipped = bulk.route_rows(
        texts, ["R1", "R2", "R3"], fake_predict,
        on_progress=lambda done, total: seen.append((done, total)))

    assert skipped == [{"row": 2, "reference": "R2"}]
    assert [r["reference"] for r in results] == ["R1", "R3"]
    assert "4111" not in records[0]["complaint_text"]
    assert results[0]["status"] == "Routed"
    assert results[1]["status"] == "Under review"
    assert results[0]["issue"] == "Fees or interest"
    assert seen[-1] == (3, 3)


def test_summarise_and_export():
    _, results, _ = bulk.route_rows(
        ["my credit card fee was wrong again", "my bank account is locked now",
         "another card charge I did not make"], None, fake_predict)
    s = bulk.summarise(results)
    assert (s["routed"], s["auto"], s["review"]) == (3, 2, 1)
    assert s["teams"][0] == ("Credit Card Team", 2)

    csv = bulk.export_csv(results).decode()
    assert csv.splitlines()[0].startswith("row,reference,tracking_id,team")
    assert len(csv.splitlines()) == 4


def test_local_store_add_many(tmp_path):
    store = LocalStore(tmp_path / "c.db")
    records, _, _ = bulk.route_rows(
        ["my credit card fee was wrong again", "my bank account is locked now"],
        None, fake_predict)
    store.add_many(records)
    for r in records:
        assert store.get(r["tracking_id"])["team_name"] == r["team_name"]
