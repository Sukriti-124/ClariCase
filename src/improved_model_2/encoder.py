"""Text -> embedding. sentence-transformers is imported lazily so everything
else in this package (and its tests) works without torch installed."""
from __future__ import annotations

import re
from typing import Protocol, Sequence

import numpy as np

_REDACTION = re.compile(r"\b[xX]{2,}\b")
_SPACE = re.compile(r"\s+")


def prepare_text(text: str) -> str:
    """Light cleaning only: drop CFPB redaction tokens (XXXX) and collapse
    whitespace. Casing and punctuation are left for the encoder's tokenizer."""
    text = _REDACTION.sub(" ", text or "")
    return _SPACE.sub(" ", text).strip()


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, name: str, max_seq_length: int = 256,
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self._model = SentenceTransformer(name, device=device)
        self._model.max_seq_length = max_seq_length

    def encode(self, texts: Sequence[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        out = self._model.encode(
            [prepare_text(t) for t in texts], batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(out, dtype=np.float32)


def load_encoder(name: str, max_seq_length: int = 256,
                 device: str | None = None) -> Encoder:
    return SentenceTransformerEncoder(name, max_seq_length, device)
