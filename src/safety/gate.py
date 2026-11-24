"""Content safety gate: decide whether a complaint may enter the pipeline.

The gate runs before anything else touches a complaint. A complaint that
fails it is not classified, routed or stored; the caller shows
`result.message` to the consumer instead.

    from src.safety import SafetyGate

    result = SafetyGate().screen(text)
    if not result.allowed:
        return result.message          # do not process the complaint
    ...                                # hand `text` to the normal pipeline

Checks, in order:

    threat          threats of violence ("I will kill the manager")
    harassment      slurs and insults aimed at a person
    profanity       vulgar language, including disguised forms such as
                    "f*ck", "sh1t", "a$$hole" and "f u c k"
    misinformation  demonstrably false claims, via an LLM judge
                    (only when a judge is supplied, see `judge.py`)

The first three are deterministic word and pattern checks with no network
calls. The LLM judge is skipped when those checks have already blocked the
complaint, so blocked text costs nothing.

If the judge fails (timeout, bad reply), `fail_closed` decides: False (the
default) lets the complaint through and records `judge_error`; True blocks it
with the category `safety_check_unavailable`.

The gate is not wired into the app or the platform services yet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

from src.safety import lexicon as L
from src.safety.judge import MisinformationJudge

__all__ = ["SafetyGate", "ScreenResult", "Finding", "screen",
           "THREAT", "HARASSMENT", "PROFANITY", "MISINFORMATION",
           "CHECK_UNAVAILABLE", "MESSAGES"]

THREAT = "threat"
HARASSMENT = "harassment"
PROFANITY = "profanity"
MISINFORMATION = "misinformation"
CHECK_UNAVAILABLE = "safety_check_unavailable"

# Shown to the consumer. Keyed by the most serious category found.
MESSAGES: dict[str, str] = {
    THREAT: ("We cannot process a complaint that contains threats of violence. "
             "Please remove that language and submit it again."),
    HARASSMENT: ("We cannot process a complaint that contains slurs or personal "
                 "insults. Please remove that language and submit it again."),
    PROFANITY: ("We cannot process a complaint that contains vulgar language. "
                "Please remove it and submit your complaint again."),
    MISINFORMATION: ("Your complaint appears to contain information that is not "
                     "accurate. Please describe only what happened to you and "
                     "submit it again."),
    CHECK_UNAVAILABLE: ("We could not complete a safety check on your complaint. "
                        "Please try again in a few minutes."),
}

# Most serious first; decides which message is shown.
_SEVERITY = (THREAT, HARASSMENT, PROFANITY, MISINFORMATION, CHECK_UNAVAILABLE)


class Finding(NamedTuple):
    """One reason the complaint was blocked.

    `term` is the canonical word or rule that matched ("fuck", "threat:1"),
    not the consumer's spelling of it, so audit logs carry no raw text.
    """
    category: str
    term: str


@dataclass(frozen=True)
class ScreenResult:
    allowed: bool
    categories: tuple[str, ...] = ()
    findings: tuple[Finding, ...] = ()
    message: str | None = None
    misinformation_checked: bool = False
    judge_reason: str = ""
    judge_error: str | None = None

    @property
    def decision(self) -> str:
        return "allow" if self.allowed else "block"


# Character substitutions used to disguise words: "sh1t", "a$$", "@ss".
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                       "7": "t", "@": "a", "$": "s"})
# Letters split by single spaces, dots, dashes or underscores: "f u c k",
# "s.h.i.t". Joined before tokenising.
_SPACED = re.compile(
    r"(?<![\w*@$])(?:[a-z0-9*@$][ .\-_]){2,}[a-z0-9*@$](?![\w*@$])")
_SPACED_SEP = re.compile(r"[ .\-_]")
_TOKEN = re.compile(r"[a-z0-9*@$]+")
_CURRENCY = re.compile(r"\$(?=\d)")


def _normalise(token: str) -> str:
    """Undo disguises in one token. Tokens with no letter ("500") are dropped.

    Stars at either end are stripped. Complaints pasted from formatted
    letters are full of leftover Markdown ("**a** formal demand"), and "a**"
    must not read as a masked "ass". The cost is that a word masked to its
    first letter ("f***") is not caught; "f*ck" and "a**hole" still are.
    """
    if not any(c.isalpha() for c in token):
        return ""
    return token.translate(_LEET).strip("*")


def _tokens(text: str) -> list[str]:
    low = text.lower()
    # "$" before a digit is a currency sign, not a disguised "s": "a$5 fee".
    low = _CURRENCY.sub(" ", low)
    low = _SPACED.sub(lambda m: _SPACED_SEP.sub("", m.group(0)), low)
    return [t for t in (_normalise(raw) for raw in _TOKEN.findall(low)) if t]


def _matches(token: str, term: str, prefix: bool) -> bool:
    if "*" not in token:
        return token.startswith(term) if prefix else token == term
    # Masked spelling such as "f*ck" or "sh**ty": each star stands for one
    # letter. Stars are only ever inside the token (`_normalise` strips the
    # ends), so the first and last letters are always visible.
    if prefix:
        if len(token) < len(term):
            return False
        head = token[:len(term)]
    else:
        if len(token) != len(term):
            return False
        head = token
    return all(a == "*" or a == b for a, b in zip(head, term))


def _longest_first(terms: frozenset[str]) -> tuple[str, ...]:
    # Longest first, so "motherfucker" reports "motherfuck" and not a
    # shorter term; ties sorted so the reported term is deterministic.
    return tuple(sorted(terms, key=lambda t: (-len(t), t)))


# (category, terms, prefix-match?) checked in this order for every token.
_WORD_LISTS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    (HARASSMENT, _longest_first(L.SLUR_STEMS), True),
    (HARASSMENT, _longest_first(L.SLUR_WORDS), False),
    (PROFANITY, _longest_first(L.PROFANITY_STEMS), True),
    (PROFANITY, _longest_first(L.PROFANITY_WORDS), False),
)


def _word_findings(text: str) -> list[Finding]:
    found: list[Finding] = []
    for token in _tokens(text):
        for category, terms, prefix in _WORD_LISTS:
            term = next((t for t in terms if _matches(token, t, prefix)), None)
            if term is not None:
                found.append(Finding(category, term))
                break
    return found


def _pattern_findings(text: str) -> list[Finding]:
    low = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    low = re.sub(r"\s+", " ", low)
    found: list[Finding] = []
    for category, patterns in ((THREAT, L.THREAT_PATTERNS),
                               (HARASSMENT, L.INSULT_PATTERNS)):
        for i, pattern in enumerate(patterns):
            if pattern.search(low):
                found.append(Finding(category, f"{category}:{i}"))
    return found


def _dedupe(findings: list[Finding]) -> tuple[Finding, ...]:
    return tuple(dict.fromkeys(findings))


def _categories(findings: tuple[Finding, ...]) -> tuple[str, ...]:
    present = {f.category for f in findings}
    return tuple(c for c in _SEVERITY if c in present)


@dataclass(frozen=True)
class SafetyGate:
    """Screens complaints. Holds only configuration, so one instance can be
    shared across requests and threads, provided the judge's backend is
    thread-safe."""
    judge: MisinformationJudge | None = None
    fail_closed: bool = False

    def screen(self, text: str | None) -> ScreenResult:
        text = text or ""
        if not text.strip():
            return ScreenResult(allowed=True)

        findings = _dedupe(_pattern_findings(text) + _word_findings(text))
        if findings:
            return self._block(findings)

        if self.judge is None:
            return ScreenResult(allowed=True)

        verdict = self.judge.judge(text)
        if verdict.error is not None:
            if self.fail_closed:
                return self._block((Finding(CHECK_UNAVAILABLE, verdict.error),),
                                   checked=True, error=verdict.error)
            return ScreenResult(allowed=True, misinformation_checked=True,
                                judge_error=verdict.error)
        if verdict.flagged:
            return self._block((Finding(MISINFORMATION, "llm_judge"),),
                               checked=True, reason=verdict.reason)
        return ScreenResult(allowed=True, misinformation_checked=True,
                            judge_reason=verdict.reason)

    @staticmethod
    def _block(findings: tuple[Finding, ...], *, checked: bool = False,
               reason: str = "", error: str | None = None) -> ScreenResult:
        categories = _categories(findings)
        return ScreenResult(allowed=False, categories=categories,
                            findings=findings, message=MESSAGES[categories[0]],
                            misinformation_checked=checked,
                            judge_reason=reason, judge_error=error)


_DEFAULT = SafetyGate()


def screen(text: str | None) -> ScreenResult:
    """Screen with the word and pattern checks only (no LLM call)."""
    return _DEFAULT.screen(text)
