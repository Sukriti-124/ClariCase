"""IM2_Explainer: sentence occlusion for improved_model_2 (design §4, Req 5).

For each sentence (or merged span) of the Masked_Text, the Driver is
``P(chosen | masked_text) - P(chosen | masked_text without that span)``.
All occlusion variants, plus the Masked_Text baseline when the served
embedding was not computed on the Masked_Text, are encoded in one batched
``encoder.encode`` call (F7). The section is ``approximate``/``occlusion``.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from src.ensemble.probs import TEAMS, reorder
from src.explain.explainers import NOMINAL, error_reason, skipped_section
from src.explain.record import Driver, Section, Span, make_section
from src.explain.text import derivable_sentence, fold_redacted, segment

__all__ = ["NAME", "K_DEFAULT", "K_MIN", "K_MAX", "resolve_k", "merge_spans",
           "explain_im2"]

NAME = "improved_model_2"
K_DEFAULT = 20
K_MIN, K_MAX = 1, 200
_INT = re.compile(r"[+-]?\d+")


def resolve_k(k: Any) -> tuple[int, bool]:
    """``(k_eff, k_default_applied)`` for a raw configured K (Req 5.3).

    An int, or a string holding an integer, in 1..200 is used as is. Anything
    else (missing, bool, float, non-integer text, out of range) gives 20.
    """
    value: int | None = None
    if isinstance(k, bool):
        value = None
    elif isinstance(k, (int, np.integer)):
        value = int(k)
    elif isinstance(k, str) and _INT.fullmatch(k.strip()):
        value = int(k.strip())
    if value is not None and K_MIN <= value <= K_MAX:
        return value, False
    return K_DEFAULT, True


def merge_spans(sentences: Sequence[tuple[int, int]], k_eff: int
                ) -> list[tuple[int, int]]:
    """Merge sentence offsets into at most ``k_eff`` contiguous spans (Req 5.4).

    With ``q, r = divmod(n, k_eff)`` the first ``r`` spans hold ``q + 1``
    sentences and the rest ``q``. With ``n <= k_eff`` each sentence is a span.
    """
    n = len(sentences)
    if n <= k_eff:
        return list(sentences)
    q, r = divmod(n, k_eff)
    out, pos = [], 0
    for i in range(k_eff):
        size = q + 1 if i < r else q
        group = sentences[pos:pos + size]
        out.append((group[0][0], group[-1][1]))
        pos += size
    return out


def _raw_k(k: Any) -> int | str | None:
    if k is None or isinstance(k, str):
        return k
    if isinstance(k, (int, np.integer)) and not isinstance(k, bool):
        return int(k)
    return str(k)


def _p_chosen(classifier, team_classes, X: np.ndarray, chosen_team: str) -> np.ndarray:
    P = reorder(classifier.predict_proba(X), team_classes, TEAMS)
    p = P[:, TEAMS.index(chosen_team)]
    return np.clip(p, 0.0, 1.0)


def explain_im2(classifier, team_classes, encoder, masked_text: str,
                chosen_team: str, served_probs: dict[str, float],
                enabled: bool, k: int | str | None,
                served_embedding_text_is_masked: bool,
                served_embedding: np.ndarray | None) -> Section:
    """The ``improved_model_2`` section (Req 5.1-5.9).

    ``served_probs`` are improved_model_2's served team probabilities and are
    recorded as ``team_probs`` in every outcome. ``served_embedding`` is used
    for ``p_full`` only when ``served_embedding_text_is_masked`` is true;
    otherwise the Masked_Text is encoded first in the same batch.
    """
    team_probs = {str(t): float(p) for t, p in served_probs.items()}
    k_eff, k_default = resolve_k(k)
    base = {"chosen_team": chosen_team, "k": _raw_k(k), "k_eff": k_eff,
            "k_default_applied": k_default, "team_probs": team_probs}

    if not enabled:
        return skipped_section(NAME, "occlusion disabled", **base)

    masked = masked_text or ""
    sentences = segment(masked)
    if not sentences:
        return skipped_section(NAME, "no sentences", n_sentences=0, **base)

    spans = merge_spans(sentences, k_eff)
    details = {**base, "n_sentences": len(sentences), "n_spans": len(spans),
               "spans_merged": len(spans) < len(sentences)}

    variants = [(masked[:s] + " " + masked[e:]).strip() for s, e in spans]
    need_baseline = not served_embedding_text_is_masked or served_embedding is None
    batch = ([masked] if need_baseline else []) + variants
    details["n_encoded_texts"] = len(batch)

    method, technique = NOMINAL[NAME]
    try:
        X = np.asarray(encoder.encode(batch))
        if X.ndim != 2 or X.shape[0] != len(batch):
            raise ValueError(f"encoder returned shape {X.shape} for {len(batch)} texts")
    except Exception as exc:
        return make_section(NAME, status="error",
                            reason=f"encoder failure: {error_reason(exc)}",
                            method=method, technique=technique, details=details)

    if need_baseline:
        X_full, X_var = X[:1], X[1:]
    else:
        X_full = np.asarray(served_embedding, dtype=X.dtype).reshape(1, -1)
        X_var = X
    p_full = float(_p_chosen(classifier, team_classes, X_full, chosen_team)[0])
    p_var = _p_chosen(classifier, team_classes, X_var, chosen_team)
    details["p_full"] = p_full

    drivers = [Driver(kind="sentence", text=masked[s:e],
                      span=Span(index=i, start=s, end=e),
                      contribution=p_full - float(p_var[i]))
               for i, (s, e) in enumerate(spans)]
    # Defensive guard: every Driver is a Masked_Text substring by
    # construction, so nothing is expected to fold (design §1).
    kept, redacted = fold_redacted(drivers, lambda t: derivable_sentence(t, masked))
    if redacted is not None:
        kept.append(redacted)
    return make_section(NAME, status="ok", method=method, technique=technique,
                        drivers=kept, details=details)
