"""Pydantic models: LLM structured outputs, domain objects and API contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------


class Route(StrEnum):
    INTERNAL = "internal"
    WEB = "web"
    BOTH = "both"
    CLARIFY = "clarify"
    DIRECT = "direct"


class Evidence(BaseModel):
    """A single retrieved passage (internal chunk or web result)."""

    source_id: str = ""  # "S1", "S2"... assigned after retrieval, used for citations
    kind: Literal["internal", "web"]
    title: str
    content: str
    doc_id: str | None = None
    url: str | None = None
    classification: Literal["public", "internal", "confidential"] = "public"
    last_updated: str | None = None
    score: float = 0.0

    def citation_label(self) -> str:
        where = self.url if self.kind == "web" else f"internal:{self.doc_id}"
        date = f", {self.last_updated}" if self.last_updated else ""
        return f"[{self.source_id}] {self.title} ({where}{date})"


# --------------------------------------------------------------------------
# LLM structured outputs (one model per agent)
# --------------------------------------------------------------------------


class StandaloneQuery(BaseModel):
    """Contextualizer output: follow-up rewritten as a self-contained question."""

    standalone_query: str = Field(description="The user's latest message rewritten so it is understandable without the chat history.")
    is_follow_up: bool = Field(description="True if the latest message depended on earlier turns.")


class ConversationSummary(BaseModel):
    summary: str = Field(description="Concise running summary of the conversation so far: topics, entities, decisions.")


class QueryClassification(BaseModel):
    route: Route = Field(description="internal = company knowledge base only; web = public/current info only; both = needs both; clarify = too ambiguous to answer; direct = greeting/small talk/meta question needing no retrieval.")
    reasoning: str = Field(description="One or two sentences explaining the routing decision.")
    needs_current_info: bool = Field(description="True if the answer depends on recent/real-time external facts.")
    internal_query: str = Field(default="", description="Optimised search query for the internal knowledge base (empty if not needed).")
    web_query: str = Field(default="", description="Optimised web search query. MUST NOT contain confidential internal names (empty if not needed).")
    clarification_question: str = Field(default="", description="If route=clarify: a single specific question to ask the user, ideally offering options.")
    direct_answer: str = Field(default="", description="If route=direct: the short reply to give.")


class RelevanceGrade(BaseModel):
    source_id: str
    relevant: bool
    reason: str = ""


class RelevanceGrades(BaseModel):
    grades: list[RelevanceGrade]


class ReformulatedQuery(BaseModel):
    internal_query: str = ""
    web_query: str = ""


class Conflict(BaseModel):
    topic: str = Field(description="What the disagreement is about, e.g. 'PayNova card processing fee'.")
    internal_claim: str
    internal_source_ids: list[str]
    external_claim: str
    external_source_ids: list[str]
    likely_resolution: str = Field(description="Which is more likely correct/current and why (e.g. dates), or 'unresolved'.")
    severity: Literal["low", "medium", "high"] = "medium"


class ConflictReport(BaseModel):
    conflicts: list[Conflict] = Field(default_factory=list)


class Claim(BaseModel):
    text: str = Field(description="One atomic factual statement made in the answer.")
    source_ids: list[str] = Field(description="Evidence ids (e.g. ['S1','S3']) that directly support this claim.")


class SynthesizedAnswer(BaseModel):
    answer: str = Field(description="Markdown answer. Every factual sentence ends with citation markers like [S1] or [S2][S4].")
    claims: list[Claim] = Field(description="Every factual claim in the answer with its supporting sources.")
    confidence: float = Field(ge=0, le=1, description="Self-assessed confidence that the answer is complete and correct given the evidence.")
    gaps: list[str] = Field(default_factory=list, description="Aspects of the question the evidence did not cover.")


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    reason: str = ""


class GroundednessReport(BaseModel):
    verdicts: list[ClaimVerdict]
    score: float = Field(ge=0, le=1, description="Fraction of claims fully supported by their cited evidence.")


class JudgeScores(BaseModel):
    """LLM-as-judge evaluation output."""

    faithfulness: float = Field(ge=0, le=1)
    answer_relevance: float = Field(ge=0, le=1)
    citation_quality: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    rationale: str = ""


# --------------------------------------------------------------------------
# Human in the loop
# --------------------------------------------------------------------------


class ApprovalType(StrEnum):
    CONFIDENTIAL_DATA = "confidential_data"
    LOW_CONFIDENCE = "low_confidence"
    EXTERNAL_SEARCH_PRIVACY = "external_search_privacy"
    KB_WRITE = "kb_write"
    KB_DELETE = "kb_delete"


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = "anonymous"
    comment: str = ""


# --------------------------------------------------------------------------
# API contracts
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, description="Omit to start a new conversation.")
    user_id: str = "anonymous"


class ResumeRequest(BaseModel):
    """Resume an interrupted thread: either answer a clarification or decide an approval."""

    answer: str | None = Field(default=None, description="Reply to a clarification question.")
    decision: Literal["approve", "reject"] | None = None
    reviewer: str = "anonymous"
    comment: str = ""


class Citation(BaseModel):
    id: str
    kind: Literal["internal", "web"]
    title: str
    url: str | None = None
    doc_id: str | None = None
    classification: str = "public"
    last_updated: str | None = None
    snippet: str = ""


class PendingApproval(BaseModel):
    approval_id: str
    approval_type: ApprovalType
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["answered", "needs_clarification", "pending_approval", "rejected", "error"]
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    route: Route | None = None
    confidence: float | None = None
    groundedness: float | None = None
    clarification_question: str | None = None
    approval: PendingApproval | None = None
    warnings: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list, description="Agent nodes executed this turn, in order.")
    latency_ms: int | None = None


class DocumentIn(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]{2,80}$")
    title: str
    content: str = Field(min_length=20)
    classification: Literal["public", "internal", "confidential"] = "internal"
    department: str = "general"
    last_updated: str | None = None
    requested_by: str = "anonymous"
