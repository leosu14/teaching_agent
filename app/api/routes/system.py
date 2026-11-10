from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.auth import Principal
from app.api.authorization import principal
from app.api.deps import container
from app.schemas.catalog import AgentInfo, ProviderInfo, ToolInfo
from app.schemas.common import Schema
from app.schemas.course import CourseSummary
from app.services.container import Container

public = APIRouter(tags=["system"])  # the only routes served without authentication
router = APIRouter(tags=["system"])


@public.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/providers", response_model=ProviderInfo)
def providers(c: Container = Depends(container)) -> ProviderInfo:
    return c.catalog.providers()


@router.get("/agents", response_model=list[AgentInfo])
def agents(c: Container = Depends(container)) -> list[AgentInfo]:
    return c.catalog.agents()


@router.get("/tools", response_model=list[ToolInfo])
def tools(c: Container = Depends(container)) -> list[ToolInfo]:
    return c.catalog.tools()


class Identity(Schema):
    """Who the caller is, as the authenticator established it: the student app starts here (which learners it may
    act for, and whether the caller also teaches)."""

    user_id: str
    learner_ids: list[str]
    roles: list[str]


@router.get("/me", response_model=Identity)
def me(p: Principal = Depends(principal)) -> Identity:
    return Identity(user_id=p.user_id, learner_ids=sorted(p.learner_ids), roles=sorted(p.roles))


@router.get("/courses", response_model=list[CourseSummary])
def courses(c: Container = Depends(container)) -> list[CourseSummary]:
    """The courses the knowledge base is built from (TA_COURSES): what a goal can target. Empty when the
    knowledge base is the fixture file. Course descriptions only: no concept, probe or answer."""
    return c.courses
