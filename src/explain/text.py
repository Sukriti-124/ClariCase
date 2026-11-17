"""Text helpers shared by the explainers (design §1, Req 1.6, 1.12, 5.6).

* ``norm``: whitespace-collapsing, case-folding normalisation.
* ``segment``: deterministic sentence offsets into the Masked_Text.
* ``derivable_ngram`` / ``derivable_sentence``: is a Driver derivable from
  the Masked_Text (Req 1.6, F6)?
* ``fold_redacted``: folds every non-derivable text Driver into one
  text-free Redacted_Driver (Req 1.12, F10).
"""
from __future__ import annotations

import math
import re
import weakref
from typing import Callable, Iterable

from src.app.pii import mask
from src.explain.record import Driver

__all__ = ["norm", "segment", "strip_prefix", "derivable_ngram",
           "derivable_sentence", "driver_text", "fold_redacted"]

_WS = re.compile(r"\s+")
# A sentence ends at ., ! or ? followed by whitespace; any newline also splits.
_SEP = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
TEXT_KINDS = frozenset({"ngram", "sentence"})


def norm(s: str) -> str:
    return _WS.sub(" ", s).strip().casefold()


def segment(masked_text: str) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of each sentence, whitespace trimmed, empty
    pieces dropped. ``masked_text[start:end]`` is the sentence."""
    text = masked_text or ""
    bounds: list[tuple[int, int]] = []
    pos = 0
    for m in _SEP.finditer(text):
        bounds.append((pos, m.start()))
        pos = m.end()
    bounds.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    for start, end in bounds:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            out.append((start, end))
    return out


def _parts(vectorizer) -> dict[str, object]:
    """Named sub-vectorizers of a FeatureUnion, else ``{}``."""
    return {name: tr for name, tr in getattr(vectorizer, "transformer_list", ())}


def _resolve(name: str, vectorizer) -> tuple[object | None, str]:
    """The sub-vectorizer that produced feature ``name`` and the bare n-gram.

    A FeatureUnion names features ``<part>__<ngram>`` (``word__``/``char__``
    for IM1). A plain TfidfVectorizer (BM2, or IM1 built without char
    features) has no prefix.
    """
    parts = _parts(vectorizer)
    if parts:
        prefix, sep, rest = name.partition("__")
        if sep and prefix in parts:
            return parts[prefix], rest
        return None, name
    if hasattr(vectorizer, "build_analyzer"):
        return vectorizer, name
    return None, name


def strip_prefix(name: str, vectorizer=None) -> str:
    """The n-gram without its ``word__``/``char__`` (or FeatureUnion) prefix."""
    if vectorizer is not None:
        return _resolve(name, vectorizer)[1]
    for p in ("word__", "char__"):
        if name.startswith(p):
            return name[len(p):]
    return name


# Analyzer output per (sub-vectorizer, masked_text); explainers test every
# non-zero feature of one text, so the analyzer runs once per text.
_CACHE: "weakref.WeakKeyDictionary[object, dict[str, frozenset[str]]]" = \
    weakref.WeakKeyDictionary()
_CACHE_PER_VEC = 8


def _ngrams(sub, masked_text: str) -> frozenset[str]:
    try:
        per = _CACHE.setdefault(sub, {})
    except TypeError:                     # not weak-referenceable
        per = {}
    got = per.get(masked_text)
    if got is None:
        got = frozenset(g.lower() for g in sub.build_analyzer()(masked_text))
        if len(per) >= _CACHE_PER_VEC:
            per.pop(next(iter(per)))
        per[masked_text] = got
    return got


def derivable_ngram(name: str, masked_text: str, vectorizer) -> bool:
    """True when feature ``name`` is produced by its vectorizer's analyzer on
    the Masked_Text, compared lower-cased (Req 1.6, F6)."""
    sub, gram = _resolve(name, vectorizer)
    if sub is None or not gram:
        return False
    return gram.lower() in _ngrams(sub, masked_text or "")


def derivable_sentence(text: str, masked_text: str) -> bool:
    """True when the sentence Driver text is a non-empty Masked_Text substring."""
    return bool(text) and text in (masked_text or "")


def driver_text(d: Driver) -> str | None:
    """The text a text Driver carries: the n-gram label or the sentence."""
    if d.kind == "ngram":
        return d.label
    if d.kind == "sentence":
        return d.text
    return None


def _mask_invariant(s: str) -> bool:
    if mask(s) != s:
        return False
    bare = strip_prefix(s)                # "word__1234567" hides the \b
    return bare == s or mask(bare) == bare


def fold_redacted(drivers: Iterable[Driver],
                  is_derivable: Callable[[str], bool]
                  ) -> tuple[list[Driver], Driver | None]:
    """Split ``drivers`` into kept Drivers and one optional Redacted_Driver.

    A text Driver (``ngram``/``sentence``) is kept only when its text passes
    ``is_derivable`` and ``mask(text) == text`` (also for the n-gram without
    its prefix). Non-text Drivers pass through. Folded contributions (and any
    incoming ``redacted`` Driver) are summed; the Redacted_Driver is returned
    only when that sum is non-zero. Input order is preserved.
    """
    kept: list[Driver] = []
    folded: list[float] = []
    for d in drivers:
        if d.kind == "redacted":
            folded.append(d.contribution)
            continue
        if d.kind in TEXT_KINDS:
            t = driver_text(d)
            if not t or not is_derivable(t) or not _mask_invariant(t):
                folded.append(d.contribution)
                continue
        kept.append(d)
    total = math.fsum(folded)
    redacted = Driver(kind="redacted", contribution=total) if total != 0 else None
    return kept, redacted
