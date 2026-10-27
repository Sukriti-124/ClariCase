"""Redact personal information from a complaint before it is stored or shown.

The app accepts free text typed by a consumer, so it can contain anything:
account numbers, Social Security numbers, phone numbers, addresses. The
training corpus never did, because the CFPB redacts narratives before
publication. `mask()` closes that gap.

Replacements follow the CFPB's own convention rather than typed markers like
`[SSN]`:

    identifiers   ->  XXXX
    dates         ->  XX/XX/XXXX

That matters for accuracy as well as privacy. `clean_text` drops `x{2,}` runs
when it builds features, which is how the model learned to read redacted
narratives. A marker the corpus never contained would survive that step and
reach the classifier as an unseen token.

`mask()` is idempotent: masking already-masked text changes nothing.

    >>> mask("Call me at 555-123-4567 about account 1234567890.")
    'Call me at XXXX about account XXXX.'

Names are deliberately left in place; see NAMES below.
"""

from __future__ import annotations

import re
from typing import Iterable, NamedTuple

__all__ = ["mask", "scan", "Finding", "REDACTION", "DATE_REDACTION"]

REDACTION = "XXXX"
DATE_REDACTION = "XX/XX/XXXX"

# Suffixes that end a street address. Ordered longest-first inside the group so
# "Street" wins over "St" on the same span.
_STREET = (r"Street|Avenue|Boulevard|Drive|Circle|Court|Place|Lane|Road|Way"
           r"|Terrace|Parkway|Highway|Blvd|Pkwy|Hwy|Ave|Rd|Ln|Dr|Ct|Cir|Pl|St")

# Words that introduce a reference code ("ref", "claim no.", "loan #").
_LABEL = (r"\b(?i:ref|reference|case|claim|confirmation|ticket|tracking|order"
          r"|acct|account|loan)\b")

_MONTH = (r"January|February|March|April|May|June|July|August|September"
          r"|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept"
          r"|Oct|Nov|Dec")

# Rules are tried in this order and earlier rules win an overlap, so the most
# specific patterns must come first. A bare run of digits is last: by then the
# card, SSN and phone rules have already claimed the spans they recognise.
_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), REDACTION),
    ("url", re.compile(r"\b(?:https?://|www\.)\S+", re.I), REDACTION),
    ("ssn", re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"), REDACTION),
    # 13-19 digits with optional spaces or dashes; Luhn-checked below.
    ("card", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), REDACTION),
    ("phone", re.compile(
        r"(?:\+?1[-.\s])?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), REDACTION),
    ("date", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), DATE_REDACTION),
    ("date", re.compile(
        rf"\b(?:{_MONTH})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I),
     DATE_REDACTION),
    # No trailing `\.?`: it cannot tell "Ave." from a sentence ending in
    # "Terrace." and swallowing the full stop runs two sentences together.
    # Case-sensitive on purpose: the street name must be one to four
    # capitalised words ("742 Evergreen Terrace"), otherwise ordinary prose
    # such as "charged 3 times along the way" or "500 dollars before going to
    # court" reads as an address. Only the suffix itself ignores case.
    ("address", re.compile(
        rf"\b\d{{1,6}}\s+(?:[A-Z0-9][\w.'-]*\s+){{1,4}}(?i:{_STREET})\b"),
     REDACTION),
    # "Apt 4B", "Suite 200", "Unit 12", "Ste. #5". The keyword must be a whole
    # word followed by a space or `#`, and the unit must contain a digit;
    # without both, "flagged", "flat", "stealing", "steps" and "units" match.
    ("address", re.compile(
        r"\b(?:apt|apartment|suite|ste|unit|floor|fl)\b\.?(?:\s*#\s*|\s+)"
        r"(?=[A-Za-z-]*\d)[\w-]{1,6}\b",
        re.I), REDACTION),
    ("zip", re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"), REDACTION),
    ("zip", re.compile(r"\bzip(?:\s*code)?\s*[:#]?\s*\d{5}(?:-\d{4})?\b",
                       re.I), REDACTION),
    # Reference codes after a label such as "ref", "claim" or "loan".
    # A code that mixes letters and digits ("ref #A1B2C3D4E5", "claim no.
    # 4XK29PL") is masked on its own. A digits-only code is masked only when
    # it is explicitly marked ("ref #12345", "case no. 48213"); otherwise
    # "a loan 25000 dollars" or "loan 2024-07" would lose the amount or date.
    # Longer digit runs are caught by the `number` rule below either way.
    ("reference", re.compile(
        rf"{_LABEL}\.?\s*(?:(?i:no\.?|number|id)\s*[:#]?\s*|[:#]\s*)?"
        r"((?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{5,})\b"),
     REDACTION),
    ("reference", re.compile(
        rf"{_LABEL}\.?\s*(?:(?i:no\.?|number|id)\s*[:#]?|[:#])\s*(\d{{5,6}})\b"),
     REDACTION),
    # Anything else long enough to identify an account, claim or case.
    ("number", re.compile(r"\b\d{7,}\b"), REDACTION),
)


class Finding(NamedTuple):
    """One span of personal information located in the text."""
    kind: str
    start: int
    end: int
    text: str


def _luhn(digits: str) -> bool:
    """True if `digits` passes the Luhn checksum used by payment cards."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _accept(kind: str, matched: str) -> bool:
    """Reject matches a pattern alone cannot distinguish from ordinary text."""
    if kind == "card":
        digits = re.sub(r"\D", "", matched)
        # Without a checksum, "I paid 1234 5678 9012 3456 dollars" and a real
        # card number look the same. The `number` rule still catches long runs,
        # so a rejected candidate is not left exposed.
        return 13 <= len(digits) <= 19 and _luhn(digits)
    return True


def _spans(text: str) -> list[Finding]:
    """Every accepted match, earlier rules winning any overlap."""
    taken: list[Finding] = []
    for kind, pattern, _ in _RULES:
        for m in pattern.finditer(text):
            # A rule with a group (reference codes) masks only the group,
            # not the label in front of it.
            start, end = (m.span(1) if pattern.groups else m.span())
            if not _accept(kind, text[start:end]):
                continue
            if any(start < f.end and f.start < end for f in taken):
                continue
            taken.append(Finding(kind, start, end, text[start:end]))
    return sorted(taken, key=lambda f: f.start)


def scan(text: str) -> list[Finding]:
    """Locate personal information without altering the text.

    Useful for auditing what a masking pass would remove, and for tests.
    """
    return _spans(text or "")


def mask(text: str) -> str:
    """Return `text` with personal information replaced by CFPB-style markers.

    Idempotent: the markers contain no digits, so a second pass finds nothing.
    """
    if not text:
        return text
    replacement = {kind: rep for kind, _, rep in _RULES}
    out = text
    # Right to left, so earlier offsets stay valid as the string changes.
    for f in reversed(_spans(text)):
        out = out[:f.start] + replacement[f.kind] + out[f.end:]
    return out


def mask_all(texts: Iterable[str]) -> list[str]:
    """Convenience wrapper for masking a batch, e.g. a backfill."""
    return [mask(t) for t in texts]


# NAMES
# -----
# Personal names are intentionally not masked: the team chose to keep them so
# complaints stay readable for the people handling them. Stored complaints can
# therefore contain names, and should be treated as personal data, not as
# anonymised text. Masking names reliably would also need a named-entity model
# to tell people from banks, products and places.
