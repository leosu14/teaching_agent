"""Deterministic answer normalisation. Surface only: unicode form, case, punctuation, whitespace and (for configured
languages) accents. It never maps one word to another, so "no" and "sí" can never become the same answer."""

from __future__ import annotations

import re
import unicodedata

from app.schemas.assessment import NormalizationConfig

_TILDE = "̃"  # kept on n: "año" and "ano" are different words


def fold_accents(text: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and not (ch == _TILDE and out and out[-1] in "nN"):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize_answer(text: str, config: NormalizationConfig | None = None) -> str:
    c = config or NormalizationConfig()
    text = unicodedata.normalize(c.unicode_form, text)
    if not c.case_sensitive:
        text = text.lower()
    if c.strip_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    if c.fold_accents:
        text = fold_accents(text)
    if c.collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    return text.strip()
