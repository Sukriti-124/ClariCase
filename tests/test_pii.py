"""Tests for src/app/pii.py."""

import pytest

from src.app.pii import DATE_REDACTION, REDACTION, mask, scan

# A Luhn-valid test card number published by payment processors.
VALID_CARD = "4111111111111111"


@pytest.mark.parametrize("raw, expected", [
    ("Reach me at john.doe@example.com please.",
     f"Reach me at {REDACTION} please."),
    ("See https://bank.example.com/acct?id=9 for details.",
     f"See {REDACTION} for details."),
    ("My SSN is 123-45-6789.", f"My SSN is {REDACTION}."),
    ("Call 555-123-4567 any time.", f"Call {REDACTION} any time."),
    ("Account 1234567890 was debited.", f"Account {REDACTION} was debited."),
    (f"Card {VALID_CARD} was declined.", f"Card {REDACTION} was declined."),
    ("I live at 742 Evergreen Terrace.", f"I live at {REDACTION}."),
    ("Mail it to Apt 4B.", f"Mail it to {REDACTION}."),
    ("I am in TX 75201 now.", f"I am in {REDACTION} now."),
    ("zip code: 90210", REDACTION),
    ("My name is Jane Smith and I object.",
     f"My name is {REDACTION} and I object."),
])
def test_masks_each_kind(raw, expected):
    assert mask(raw) == expected


@pytest.mark.parametrize("raw", [
    "It happened on 03/14/2025.",
    "It happened on March 14, 2025.",
])
def test_dates_use_the_cfpb_marker(raw):
    assert DATE_REDACTION in mask(raw)
    assert "2025" not in mask(raw)


def test_masking_is_idempotent():
    once = mask("SSN 123-45-6789, card 4111 1111 1111 1111, on 03/14/2025.")
    assert mask(once) == once


def test_leaves_ordinary_complaint_text_alone():
    raw = ("Wells Fargo charged me an overdraft fee twice in one week and "
           "the branch refused to reverse it.")
    assert mask(raw) == raw


def test_keeps_short_numbers_that_carry_meaning():
    # Amounts and counts are not identifiers and the model routes on them.
    raw = "They charged me 35 dollars three times, totalling 105."
    assert mask(raw) == raw


def test_rejects_long_digit_runs_that_are_not_cards():
    # Fails Luhn, so it is not reported as a card...
    raw = "Reference 1234567812345678 on the letter."
    assert "card" not in {f.kind for f in scan(raw)}
    # ...but is still masked, because it is long enough to identify an account.
    assert mask(raw) == f"Reference {REDACTION} on the letter."


def test_scan_reports_spans_without_changing_text():
    raw = "Email a@b.com or call 555-123-4567."
    found = scan(raw)
    assert {f.kind for f in found} == {"email", "phone"}
    assert all(raw[f.start:f.end] == f.text for f in found)


def test_overlapping_matches_resolve_to_one_span():
    # The phone number sits inside no other span; the SSN must not also be
    # claimed by the generic number rule.
    found = scan("SSN 123-45-6789 here.")
    assert len(found) == 1
    assert found[0].kind == "ssn"


def test_handles_empty_and_none():
    assert mask("") == ""
    assert mask(None) is None
    assert scan("") == []


def test_masks_every_identifier_in_a_realistic_complaint():
    raw = ("My name is Robert Chen. On 03/14/2025 Chase debited my account "
           "4111111111111111 for an unauthorised charge. I called "
           "555-867-5309 and emailed support@chase.example.com with no "
           "reply. I live at 1600 Pennsylvania Avenue, Apt 2C, DC 20500.")
    out = mask(raw)
    for leaked in ("Robert Chen", "03/14/2025", "4111111111111111",
                   "555-867-5309", "support@chase.example.com",
                   "1600 Pennsylvania Avenue", "20500"):
        assert leaked not in out, f"{leaked!r} survived masking"
    # The routable content must survive, or the classifier loses its signal.
    assert "Chase" in out and "unauthorised charge" in out
