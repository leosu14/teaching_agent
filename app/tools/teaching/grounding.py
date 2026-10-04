"""`teaching.ground`: the material a learner's question may be answered from.

The candidates are the current lesson's sections and its research bundle's findings (passed in by the caller), plus
the knowledge base's passages for the subject (retrieved here). An item qualifies when it shares a content word with
the question; the best few are returned, deterministically ordered. An empty pack means the material cannot answer
the question: the teacher then states that limitation instead of answering.
"""

from __future__ import annotations

import re
import unicodedata

from app.observability.scope import ExecutionScope
from app.providers.retrieval.base import Retriever
from app.schemas.teaching import GroundingItem, GroundingPack, GroundingRequest
from app.tools.base import Tool, ToolTransientError

STOPWORDS = frozenset(
    "a an and are as at be because but by can could did do does for from had has have how i if in into is it its "
    "here me my not of on or our should so than that the their them then there these they this to us use used using was "
    "we were what when where which who why will with would you your yo tu el la los las de del que y en un una por "
    "para con es se lo al".split())
KIND_ORDER = {"lesson": 0, "research": 1, "knowledge_base": 2}
KB_RESULTS = 8


def terms(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return {t for t in re.findall(r"\w+", text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())}


def select(question: str, candidates: list[GroundingItem], max_items: int) -> list[GroundingItem]:
    wanted = terms(question)
    scored = []
    for item in candidates:
        overlap = len(wanted & terms(f"{item.title} {item.text}"))
        if overlap:
            scored.append((-overlap, KIND_ORDER[item.kind], item.ref, item))
    scored.sort(key=lambda x: x[:3])
    seen: set[str] = set()
    picked = []
    for *_, item in scored:
        if item.ref not in seen:
            seen.add(item.ref)
            picked.append(item)
    return picked[:max_items]


class TeachingGroundingTool(Tool[GroundingRequest, GroundingPack]):
    name = "teaching.ground"
    description = ("Select the lesson, research and knowledge-base material a learner's question can be answered "
                   "from (empty when the material does not cover it).")
    input_model = GroundingRequest
    output_model = GroundingPack
    permissions = frozenset({"knowledge:read"})

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def run(self, data: GroundingRequest, scope: ExecutionScope) -> GroundingPack:
        try:
            passages = await self._retriever.retrieve(data.question, KB_RESULTS, {"subject": data.subject})
        except (ConnectionError, OSError) as exc:
            raise ToolTransientError(f"retriever '{self._retriever.name}' unavailable: {exc}") from exc
        scope.usage.record_service(service=f"retrieval:{self._retriever.name}", results=len(passages))
        kb = [GroundingItem(ref=f"kb:{p.doc_id}", kind="knowledge_base", title=p.title, text=p.text)
              for p in passages]
        return GroundingPack(items=select(data.question, [*data.candidates, *kb], data.max_items))
