from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import container
from app.schemas.catalog import AgentInfo, ProviderInfo, ToolInfo
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
