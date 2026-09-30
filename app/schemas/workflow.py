"""Persisted workflow state: the checkpoint that makes every task resumable."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from app.schemas.common import Schema
from app.schemas.lesson import ReviewIssue, ReviewResult, Verdict


class NodeStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class NodeState(Schema):
    status: NodeStatus = NodeStatus.PENDING
    attempts: int = 0
    output: dict | None = None
    output_type: str | None = None
    error: str | None = None
    progress: dict = Field(default_factory=dict)
    human_input: dict | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None


class WorkflowState(Schema):
    workflow_id: str
    node_states: dict[str, NodeState] = Field(default_factory=dict)
    execution_order: list[str] = Field(default_factory=list)


class RevisionPolicy(Schema):
    max_revisions: int = Field(default=2, ge=0)
    on_exhausted: Literal["fail", "accept_with_warnings"] = "fail"


class RevisionRequest(Schema):
    """Generic hand-back from a ReviewNode to its generator: what to fix, and the previous candidate."""

    revision_number: int = Field(ge=1)
    issues: list[ReviewIssue]
    previous: dict


class ReviewRound(Schema):
    revision_number: int
    verdict: Verdict
    scores: dict[str, float]
    issues: list[ReviewIssue]


class ReviewOutcome(Schema):
    status: Literal["approved", "accepted_with_warnings"]
    candidate: dict
    candidate_type: str
    revisions: int
    rounds: list[ReviewRound]
    final_review: ReviewResult


class BranchDecision(Schema):
    branch: bool
    skipped: list[str]


class ParallelOutcome(Schema):
    children: list[str]
