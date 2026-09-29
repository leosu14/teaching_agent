"""Introspection schemas for agents, tools and providers."""

from __future__ import annotations

from app.schemas.common import ModelTier, Schema


class AgentInfo(Schema):
    id: str
    name: str
    description: str
    tier: ModelTier
    tools: list[str]
    permissions: list[str]
    timeout_seconds: float
    validation_retries: int
    input_schema: dict
    output_schema: dict


class ToolInfo(Schema):
    name: str
    description: str
    permissions: list[str]
    timeout_seconds: float
    max_attempts: int
    input_schema: dict
    output_schema: dict


class ProviderInfo(Schema):
    llm_providers: list[str]
    tiers: dict[str, list[str]]
    level_frameworks: list[str]
    workflows: list[str]
