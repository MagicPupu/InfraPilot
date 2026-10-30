"""Typed graph state and the Pydantic models exchanged between nodes.

The state only holds JSON-serialisable values (``model_dump()`` output) so any
LangGraph checkpointer can persist it.
"""

import hashlib
import json
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["question", "action"]


class PlannedCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    rationale: str
    calls: list[PlannedCall] = Field(min_length=1)


class PolicyDecision(BaseModel):
    outcome: Literal["approve", "needs_approval", "refuse"]
    reasons: list[str] = Field(default_factory=list)
    destructive: bool = False
    plan_hash: str = ""


class Approval(BaseModel):
    approved: bool
    plan_hash: str


class Chunk(BaseModel):
    id: str
    source: str
    heading_path: str
    text: str


class RelevanceGrade(BaseModel):
    relevant: bool


class RewrittenQuery(BaseModel):
    query: str


class CitedAnswer(BaseModel):
    answer: str
    cited_ids: list[str]


class InfraState(TypedDict, total=False):
    request: str
    route: str
    # action path
    plan: dict[str, Any]
    policy: dict[str, Any]
    approval: dict[str, Any]
    results: list[dict[str, Any]]
    verification: dict[str, Any]
    # question path
    search_query: str
    rewrites: int
    retrieved: list[dict[str, Any]]
    relevant: bool
    answer: str
    cited_ids: list[str]
    # shared
    error: str
    final: str


def plan_hash(calls: list[PlannedCall]) -> str:
    """Stable digest binding an approval to the exact calls that were reviewed."""
    canonical = json.dumps([{"tool": c.tool, "args": c.args} for c in calls], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
