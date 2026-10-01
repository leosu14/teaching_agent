"""Research schemas. Domain- and provider-independent.

Traceability chain: a lesson section cites Citation ids -> each Citation points at one Evidence item
-> each Evidence item quotes one Source at a recorded location. Citations are built by code from
evidence and sources, never written by a model, so they cannot be invented.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import Schema

RetrievedVia = Literal["web", "knowledge_base"]

# --- Search ------------------------------------------------------------------------------


class SearchQuery(Schema):
    query_id: str = ""  # assigned by whoever plans the queries; not part of the cache key
    text: str = Field(min_length=1)
    language: str | None = None
    subject: str | None = None
    domain: str | None = None  # topic or domain hint, e.g. "football"
    max_results: int = Field(default=5, ge=1, le=50)
    target_ids: list[str] = Field(default_factory=list)  # what the query researches; never sent to providers

    def cache_key(self) -> str:
        """Equal for queries that would return the same results, whatever their id or purpose."""
        return json.dumps({
            "text": re.sub(r"\s+", " ", self.text.strip().lower()),
            "language": self.language, "subject": self.subject, "domain": self.domain,
            "max_results": self.max_results,
        }, sort_keys=True)


class SourceReliability(Schema):
    """An assessment of a source, never provider metadata. `score` is None when nothing supports one."""

    score: float | None = Field(default=None, ge=0, le=1)
    basis: str
    assessed_by: str


class Source(Schema):
    """A document as its provider described it. Optional fields stay None unless the provider supplied them."""

    source_id: str
    url: str
    canonical_url: str
    title: str
    publisher: str | None = None
    author: str | None = None
    published_at: date | None = None
    retrieved_at: datetime
    language: str | None = None
    source_type: str = "unknown"
    retrieved_via: RetrievedVia
    provider: str
    reliability: SourceReliability | None = None
    metadata: dict = Field(default_factory=dict)  # any other provider-supplied fields, verbatim


class SearchResult(Schema):
    source_id: str
    title: str
    url: str
    snippet: str
    content: str | None = None  # full text when the provider returns it
    rank: int = Field(ge=1)  # position in the provider's result list
    provider_score: float | None = None
    source: Source

    @model_validator(mode="after")
    def _matches_source(self) -> SearchResult:
        if (self.source_id, self.title, self.url) != (self.source.source_id, self.source.title, self.source.url):
            raise ValueError("search result fields must match its source")
        return self


# --- Ranking -----------------------------------------------------------------------------


class RankingWeights(Schema):
    relevance: float = Field(default=0.45, ge=0)
    quality: float = Field(default=0.3, ge=0)
    freshness: float = Field(default=0.1, ge=0)
    language: float = Field(default=0.1, ge=0)
    novelty: float = Field(default=0.05, ge=0)


class RankCandidate(Schema):
    result: SearchResult
    matched_queries: list[str] = Field(min_length=1)  # texts of the queries that returned this result


class RankingSignals(Schema):
    """Each signal is 0..1. `quality`, `freshness` and `language` are 0.5 (neutral) when unknown."""

    relevance: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    language: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


# --- Evidence, findings, citations -------------------------------------------------------


class EvidenceLocation(Schema):
    field: Literal["content", "snippet"]
    start: int = Field(ge=0)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> EvidenceLocation:
        if self.end <= self.start:
            raise ValueError("evidence location end must be after start")
        return self


class ExtractionMetadata(Schema):
    method: str
    extractor: str
    extracted_at: datetime
    query_ids: list[str] = Field(default_factory=list)  # the queries that surfaced the source


class Evidence(Schema):
    evidence_id: str
    source_id: str
    target_id: str
    text: str = Field(min_length=1)
    relevance: float = Field(ge=0, le=1)
    location: EvidenceLocation | None = None
    extraction: ExtractionMetadata


class KeyFinding(Schema):
    finding_id: str
    target_id: str
    statement: str = Field(min_length=1)
    example: str | None = None
    practice_prompt: str | None = None
    practice_answer: str | None = None
    evidence_ids: list[str] = Field(min_length=1)


class Citation(Schema):
    citation_id: str
    evidence_id: str
    source_id: str
    title: str
    url: str
    publisher: str | None = None
    retrieved_at: datetime
    locator: str | None = None

    def reference(self) -> str:
        parts = [self.title, self.publisher, self.url, f"retrieved {self.retrieved_at.date().isoformat()}"]
        return ". ".join(p for p in parts if p)


# --- Research objective and bundle ---------------------------------------------------------


class ResearchTarget(Schema):
    """One thing to research, e.g. a lesson concept. `priority` orders the work."""

    target_id: str
    name: str
    description: str = ""
    priority: Literal["gap", "learn", "review"] = "learn"


class ResearchObjective(Schema):
    description: str
    subject: str
    topic: str
    level: str | None = None
    language: str | None = None
    targets: list[ResearchTarget] = Field(min_length=1)


class ResearchError(Schema):
    stage: Literal["search", "retrieve", "rank", "select", "extract"]
    message: str
    query_id: str | None = None
    tool: str | None = None


class RejectedSource(Schema):
    source: Source
    reason: str


class ResearchBundle(Schema):
    research_id: str
    objective: ResearchObjective
    status: Literal["complete", "partial", "failed"]
    queries: list[SearchQuery] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)  # selected sources only
    rejected_sources: list[RejectedSource] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    key_findings: list[KeyFinding] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[ResearchError] = Field(default_factory=list)
    generated_at: datetime

    @model_validator(mode="after")
    def _traceable(self) -> ResearchBundle:
        ids = [s.source_id for s in self.sources]
        urls = [s.canonical_url for s in self.sources]
        if len(ids) != len(set(ids)) or len(urls) != len(set(urls)):
            raise ValueError("sources must be deduplicated")
        sources = set(ids)
        evidence = {}
        for ev in self.evidence:
            if ev.source_id not in sources:
                raise ValueError(f"evidence {ev.evidence_id} cites unknown or unselected source {ev.source_id}")
            if ev.evidence_id in evidence:
                raise ValueError(f"duplicate evidence id {ev.evidence_id}")
            evidence[ev.evidence_id] = ev
        for finding in self.key_findings:
            missing = [e for e in finding.evidence_ids if e not in evidence]
            if missing:
                raise ValueError(f"finding {finding.finding_id} refers to unknown evidence {missing}")
        cited = set()
        for citation in self.citations:
            ev = evidence.get(citation.evidence_id)
            if ev is None or ev.source_id != citation.source_id:
                raise ValueError(f"citation {citation.citation_id} does not match its evidence and source")
            cited.add(citation.evidence_id)
        if cited != set(evidence):
            raise ValueError("every evidence item needs exactly one citation")
        if len({c.citation_id for c in self.citations}) != len(self.citations):
            raise ValueError("duplicate citation ids")
        if self.status == "failed" and (self.sources or self.evidence):
            raise ValueError("a failed research bundle carries no sources or evidence")
        if self.status == "failed" and not (self.errors or self.warnings):
            raise ValueError("a failed research bundle must say why")
        if self.status == "complete" and not self.evidence:
            raise ValueError("a complete research bundle needs evidence")
        return self

    # --- lookups used downstream ----------------------------------------------------------

    def source(self, source_id: str) -> Source:
        return next(s for s in self.sources if s.source_id == source_id)

    def citation_for(self, evidence_id: str) -> Citation:
        return next(c for c in self.citations if c.evidence_id == evidence_id)

    def citation_ids_for(self, finding: KeyFinding) -> list[str]:
        return [self.citation_for(e).citation_id for e in finding.evidence_ids]

    def resolve(self, citation_id: str) -> tuple[Citation, Evidence, Source]:
        """Citation -> Evidence -> Source. Raises LookupError for an unknown citation."""
        citation = next((c for c in self.citations if c.citation_id == citation_id), None)
        if citation is None:
            raise LookupError(f"unknown citation {citation_id}")
        ev = next(e for e in self.evidence if e.evidence_id == citation.evidence_id)
        return citation, ev, self.source(ev.source_id)

    def findings_for(self, target_id: str) -> list[KeyFinding]:
        return [f for f in self.key_findings if f.target_id == target_id]

    def focused(self, target_ids: list[str]) -> ResearchBundle:
        """The part of the bundle that supports the given targets, with citations intact."""
        wanted = set(target_ids)
        findings = [f for f in self.key_findings if f.target_id in wanted]
        evidence_ids = {e for f in findings for e in f.evidence_ids}
        evidence = [e for e in self.evidence if e.evidence_id in evidence_ids]
        source_ids = {e.source_id for e in evidence}
        if not evidence and self.status != "failed":
            return self.model_copy(update={
                "status": "partial", "sources": [], "evidence": [], "key_findings": [], "citations": [],
                "warnings": [*self.warnings, "The research has no evidence for these targets."],
            })
        return self.model_copy(update={
            "sources": [s for s in self.sources if s.source_id in source_ids],
            "evidence": evidence,
            "key_findings": findings,
            "citations": [c for c in self.citations if c.evidence_id in evidence_ids],
        })


# --- Evidence extraction (the model's part of research) ------------------------------------


class ExtractionSource(Schema):
    source_id: str
    title: str
    publisher: str | None = None
    snippet: str
    content: str | None = None
    relevance: float = Field(ge=0, le=1)


class EvidenceExtractionInput(Schema):
    objective: ResearchObjective
    sources: list[ExtractionSource] = Field(min_length=1)
    max_findings_per_target: int = Field(default=3, ge=1)


class ExtractedEvidence(Schema):
    ref: str
    source_id: str
    target_id: str
    text: str = Field(min_length=1)
    field: Literal["content", "snippet"]
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    relevance: float = Field(ge=0, le=1)


class ProposedFinding(Schema):
    target_id: str
    statement: str = Field(min_length=1)
    example: str | None = None
    practice_prompt: str | None = None
    practice_answer: str | None = None
    evidence_refs: list[str] = Field(min_length=1)


class EvidenceExtraction(Schema):
    evidence: list[ExtractedEvidence] = Field(default_factory=list)
    findings: list[ProposedFinding] = Field(default_factory=list)
    summary: str = ""
