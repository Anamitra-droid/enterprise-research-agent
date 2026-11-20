"""Typed LangGraph state.

State persists per thread (via the SQLite checkpointer). Conversation-level fields
(`messages`, `summary`) accumulate across turns; turn-level fields are reset by the
service at the start of every new user message (see `fresh_turn_input`).
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from research_agent.schemas import (
    Conflict,
    Evidence,
    GroundednessReport,
    QueryClassification,
    Route,
    SynthesizedAnswer,
)

RESET = "__reset__"


def append_or_reset(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer: appends lists (safe for parallel branches); a list starting with RESET clears."""
    left = left or []
    right = right or []
    if right and right[0] == RESET:
        return list(right[1:])
    return left + right


class ResearchState(BaseModel):
    # --- conversation-level (persist across turns) ---
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    summary: str = ""
    user_id: str = "anonymous"

    # --- turn-level ---
    turn_id: str = ""
    query: str = ""
    standalone_query: str = ""
    classification: QueryClassification | None = None
    route: Route | None = None
    internal_query: str = ""
    web_query: str = ""
    clarification_count: int = 0
    clarification_question: str = ""

    web_allowed: bool = True
    internal_evidence: list[Evidence] = Field(default_factory=list)
    web_evidence: list[Evidence] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    retrieval_attempts: int = 0

    conflicts: list[Conflict] = Field(default_factory=list)
    draft: SynthesizedAnswer | None = None
    synthesis_attempts: int = 0
    synthesis_feedback: str = ""
    groundedness: GroundednessReport | None = None

    approvals: Annotated[list[str], append_or_reset] = Field(default_factory=list)  # audit trail "type:decision"
    status: str = "running"  # running | answered | rejected
    final_answer: str = ""

    warnings: Annotated[list[str], append_or_reset] = Field(default_factory=list)
    trace: Annotated[list[str], append_or_reset] = Field(default_factory=list)


def fresh_turn_input(message: Any, query: str, user_id: str, turn_id: str) -> dict[str, Any]:
    """Input for a new user turn: appends the message and resets all turn-level fields."""
    return {
        "messages": [message],
        "user_id": user_id,
        "turn_id": turn_id,
        "query": query,
        "standalone_query": "",
        "classification": None,
        "route": None,
        "internal_query": "",
        "web_query": "",
        "clarification_count": 0,
        "clarification_question": "",
        "web_allowed": True,
        "internal_evidence": [],
        "web_evidence": [],
        "evidence": [],
        "retrieval_attempts": 0,
        "conflicts": [],
        "draft": None,
        "synthesis_attempts": 0,
        "synthesis_feedback": "",
        "groundedness": None,
        "approvals": [RESET],
        "status": "running",
        "final_answer": "",
        "warnings": [RESET],
        "trace": [RESET],
    }
