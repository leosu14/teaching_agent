"""Source deduplication: the same document must not appear twice however it was found."""

from __future__ import annotations

import re

from pydantic import Field

from app.schemas.common import Schema
from app.schemas.research import RankCandidate


class DuplicateRecord(Schema):
    kept_source_id: str
    dropped_source_id: str
    url: str
    reason: str


class Deduplicated(Schema):
    unique: list[RankCandidate]
    duplicates: list[DuplicateRecord] = Field(default_factory=list)


def _norm(text: str | None) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def deduplicate(candidates: list[RankCandidate]) -> Deduplicated:
    """Keep the first occurrence of each source (input order) and merge the queries that found it.

    Two candidates are the same source when they share a source id (the id is derived from the canonical
    URL, so http/https, `www.`, trailing slashes, fragments and tracking parameters do not matter), when
    their canonical URLs match, or when they have identical content, or identical title and publisher.
    """
    kept: list[RankCandidate] = []
    by_key: dict[tuple[str, str], int] = {}
    duplicates: list[DuplicateRecord] = []
    for cand in candidates:
        src = cand.result.source
        keys = [("id", src.source_id), ("url", src.canonical_url)]
        if cand.result.content:
            keys.append(("content", _norm(cand.result.content)))
        if src.publisher:
            keys.append(("title", f"{_norm(src.title)}|{_norm(src.publisher)}"))
        match = next(((kind, by_key[(kind, value)]) for kind, value in keys if (kind, value) in by_key), None)
        if match is None:
            index = len(kept)
            kept.append(cand.model_copy(deep=True))
            for key in keys:
                by_key.setdefault(key, index)
            continue
        kind, index = match
        original = kept[index]
        merged = list(dict.fromkeys([*original.matched_queries, *cand.matched_queries]))
        kept[index] = original.model_copy(update={"matched_queries": merged})
        reason = {"id": "same source id", "url": "same canonical URL", "content": "identical content",
                  "title": "same title and publisher"}[kind]
        duplicates.append(DuplicateRecord(kept_source_id=original.result.source_id, dropped_source_id=src.source_id,
                                          url=src.url, reason=reason))
    return Deduplicated(unique=kept, duplicates=duplicates)
