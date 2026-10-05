"""Deterministic matching: the first grading step, and the only one for closed questions.

The normalised answer is compared with the normalised expected and acceptable answers. A multiple-choice answer may
give the choice itself or its letter / number; a true/false answer may say it in a few languages. A known wrong answer
(an item's `known_errors`) is recognised as such. Nothing here interprets meaning: an answer that matches nothing is
"no match", and only free text may go on to rubric or semantic grading.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.assessment.normalization import fold_accents, normalize_answer
from app.schemas.assessment import KnownError, NormalizationConfig, ResponseType

MatchKind = Literal["accepted", "known_error", "invalid", "no_match", "empty"]

# True/false words by language (accent-folded, lower case). English is always accepted.
TRUE_WORDS = {"en": {"true", "t", "yes", "y", "1"}, "es": {"verdadero", "cierto", "si"}, "pt": {"verdadeiro", "sim"},
              "fr": {"vrai", "oui"}, "de": {"wahr", "richtig", "ja"}, "it": {"vero", "si"}}
FALSE_WORDS = {"en": {"false", "f", "no", "n", "0"}, "es": {"falso", "no"}, "pt": {"falso", "nao"},
               "fr": {"faux", "non"}, "de": {"falsch", "nein"}, "it": {"falso", "no"}}


@dataclass(frozen=True)
class Match:
    kind: MatchKind
    normalized: str
    matched: str | None = None  # the expected / acceptable answer, choice or known error that matched
    known_error: KnownError | None = None


def choice_labels(choices: Sequence[str]) -> dict[str, str]:
    """"a"/"b"/... and "1"/"2"/... for each choice."""
    return {**{chr(ord("a") + i): c for i, c in enumerate(choices)}, **{str(i + 1): c for i, c in enumerate(choices)}}


def parse_boolean(normalized: str, language: str) -> bool | None:
    word = fold_accents(normalized)
    lang = language.split("-")[0].lower()
    if word in TRUE_WORDS["en"] | TRUE_WORDS.get(lang, set()):
        return True
    if word in FALSE_WORDS["en"] | FALSE_WORDS.get(lang, set()):
        return False
    return None


def match_answer(answer: str, *, expected: str, acceptable: Sequence[str] = (), response_type: ResponseType,
                 choices: Sequence[str] = (), known_errors: Sequence[KnownError] = (), language: str = "en",
                 config: NormalizationConfig | None = None) -> Match:
    def norm(text: str) -> str:
        return normalize_answer(text, config)

    given = norm(answer)
    if not given:
        return Match("empty", given)
    accepted = {n: original for original in (expected, *acceptable) if (n := norm(original))}
    if response_type == ResponseType.MULTIPLE_CHOICE:
        by_norm = {norm(c): c for c in choices}
        if given not in by_norm:
            labels = choice_labels(choices)
            if given not in labels:
                return Match("invalid", given)
            given = norm(labels[given])
    elif response_type == ResponseType.TRUE_FALSE:
        value = parse_boolean(given, language)
        if value is None:
            return Match("invalid", given)
        expected_value = parse_boolean(norm(expected), "en")
        return Match("accepted" if value == expected_value else "no_match", given,
                     matched=expected if value == expected_value else None)
    if given in accepted:
        return Match("accepted", given, matched=accepted[given])
    for error in known_errors:
        if norm(error.answer) == given:
            return Match("known_error", given, matched=error.answer, known_error=error)
    return Match("no_match", given)
