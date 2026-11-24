"""Mark the phrases that drove a routing decision inside the complaint text.

Phrases are the model's n-grams (lower-case words joined by spaces); in the
text the same words may differ in case or be separated by punctuation.
Returns HTML: the text is escaped, matches are wrapped in <mark>, and masked
tokens (XXXX, XX/XX/XXXX) are wrapped in <span class="cc-x">.
"""

from __future__ import annotations

import html
import re

_MASKED = re.compile(r"\b(?:X{2,}(?:/X{2,})*)\b")


def _pattern(phrase: str) -> re.Pattern | None:
    words = [re.escape(w) for w in phrase.split() if w]
    if not words:
        return None
    return re.compile(r"\b" + r"\W+".join(words) + r"\b", re.IGNORECASE)


def _spans(text: str, phrases: list[str]) -> list[tuple[int, int]]:
    found = []
    for phrase in phrases:
        pat = _pattern(phrase)
        if pat is not None:
            found.extend(m.span() for m in pat.finditer(text))
    found.sort()
    merged: list[list[int]] = []
    for start, end in found:            # merge overlapping matches
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _plain(segment: str) -> str:
    return _MASKED.sub(lambda m: f'<span class="cc-x">{m.group(0)}</span>',
                       html.escape(segment))


def highlight(text: str, phrases: list[str]) -> str:
    out, pos = [], 0
    for start, end in _spans(text, phrases):
        out.append(_plain(text[pos:start]))
        out.append(f"<mark>{html.escape(text[start:end])}</mark>")
        pos = end
    out.append(_plain(text[pos:]))
    return "".join(out)
