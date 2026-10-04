"""Text normalisation shared by answer grading and answer-leak checks."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Case-folded, punctuation-free, whitespace-collapsed text (accents are kept: they can be part of an answer)."""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def reveals(text: str, answers: set[str] | list[str]) -> bool:
    """Does the text state one of the answers, as whole words?"""
    haystack = f" {normalize(text)} "
    return any(f" {normalize(a)} " in haystack for a in answers if normalize(a))
