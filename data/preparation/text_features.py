"""Deterministic text preparation shared by training and prediction."""
import re
import unicodedata


def prepare_text(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", " ", text)
    text = re.sub(r"\bx{2,}\b", " ", text)
    return " ".join(text.split())
