"""Word lists and patterns used by the content safety gate.

Terms are compared against normalised tokens (lower case, common character
substitutions undone, see `gate._normalise`). Two kinds of list:

    *_STEMS   match the start of a token, so "fuck" also catches "fucking",
              "fucked" and "fucker".
    *_WORDS   match a whole token only, because as a prefix they would start
              ordinary words ("ass" in "assessment", "retard" in "retardant").

Words that are vulgar but routine in angry complaints and harmless on their
own ("damn", "hell", "crap", "sucks") are left out on purpose. So are names
and words with a common innocent meaning ("Dick", "cock", "chink in the
armour"). Add to these sets rather than editing `gate.py`.
"""
from __future__ import annotations

import re

PROFANITY_STEMS: frozenset[str] = frozenset({
    "fuck", "motherfuck", "shit", "bullshit", "dipshit", "bitch", "asshole",
    "cunt", "bastard", "dumbass", "jackass", "wanker",
})

PROFANITY_WORDS: frozenset[str] = frozenset({
    "ass", "asses", "piss", "pissed", "pissing", "wtf", "stfu",
    "fck", "fcking", "fckin", "fuk", "fuking", "fukin", "biatch",
    "twat", "twats", "prick", "pricks", "douche", "douchebag", "douchebags",
})

# Slurs against protected groups. Reported as harassment, not profanity.
SLUR_STEMS: frozenset[str] = frozenset({"nigger", "faggot"})

SLUR_WORDS: frozenset[str] = frozenset({
    "nigga", "niggas", "fag", "fags", "retard", "retards", "retarded",
    "spic", "spics", "kike", "kikes", "tranny", "trannies",
    "wetback", "wetbacks",
})

# Insults aimed at a person ("you idiot", "you are a stupid moron").
INSULT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\byou(?:\s+are|\s*'re|\s+r)?\s+(?:an?\s+|such\s+an?\s+)?(?:\w+\s+)?"
        r"(?:idiot|moron|imbecile|scumbag)s?\b"),
)

# People a threat can be aimed at. Kept to people so that "I will kill the
# autopay" or "shoot me an email" are not read as threats.
_TARGET = (r"(?:you|him|her|them|everyone|everybody|someone|somebody"
           r"|all\s+of\s+you|(?:the|your|that|this|those|these)\s+"
           r"(?:agent|manager|teller|rep|representative|ceo|staff|employee"
           r"|people|owner|banker|collector|supervisor|lady|guy|man|woman)s?)")

_FIRST_PERSON_INTENT = (r"(?:i|we)\s*(?:will|'ll|am\s+going\s+to|'m\s+going\s+to"
                        r"|are\s+going\s+to|'re\s+going\s+to|am\s+gonna"
                        r"|'m\s+gonna|gonna|want\s+to|wanna)")

THREAT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "I will kill the manager", "we're going to stab you".
    re.compile(rf"\b{_FIRST_PERSON_INTENT}\s+(?:\w+\s+)?"
               rf"(?:kill|murder|stab|strangle|shoot)\s+{_TARGET}\b"),
    # "I'm going to shoot up the branch".
    re.compile(rf"\b{_FIRST_PERSON_INTENT}\s+(?:\w+\s+)?shoot\s+up\b"),
    # "bomb the bank", "burn down your office".
    re.compile(r"\b(?:bomb|blow\s+up|burn\s+down|shoot\s+up)\s+"
               r"(?:the|your|this|that|their|every)\s+"
               r"(?:bank|branch|office|building|store|headquarters|hq)s?\b"),
    re.compile(r"\b(?:kill|hang)\s+yo(?:u|ur)sel(?:f|ves)\b"),
    re.compile(r"\byou\s+(?:should|deserve\s+to|will|are\s+going\s+to)\s+die\b"),
)
