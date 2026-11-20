"""Human-in-the-loop approval gate and final answer assembly."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from research_agent.agents.deps import AgentDeps
from research_agent.agents.synthesizer import cited_ids
from research_agent.graph.state import ResearchState
from research_agent.observability.tracing import traced_node
from research_agent.schemas import ApprovalType

REJECTED_ANSWER = (
    "This answer was withheld after human review{reason}. "
    "Please contact the document owner or your manager if you need access to this information."
)


def used_source_ids(state: ResearchState) -> set[str]:
    if not state.draft:
        return set()
    ids = cited_ids(state.draft.answer)
    for c in state.draft.claims:
        ids.update(c.source_ids)
    return ids


def approval_reasons(state: ResearchState, deps: AgentDeps) -> tuple[ApprovalType | None, list[str], list[str]]:
    s = deps.settings
    if not state.draft or not state.evidence or not state.draft.claims:
        return None, [], []  # nothing asserted (abstention) -> nothing to review
    used = used_source_ids(state)
    confidential = [e for e in state.evidence if e.classification == "confidential" and e.source_id in used]
    reasons: list[str] = []
    kind: ApprovalType | None = None
    if s.hitl_confidential and confidential:
        kind = ApprovalType.CONFIDENTIAL_DATA
        reasons.append("Answer uses CONFIDENTIAL documents: " + ", ".join(sorted({e.doc_id or e.title for e in confidential})))
    g = state.groundedness.score if state.groundedness else 1.0
    if s.hitl_low_confidence and (state.draft.confidence < s.low_confidence_threshold or g < s.groundedness_threshold):
        kind = kind or ApprovalType.LOW_CONFIDENCE
        reasons.append(f"Low confidence answer (confidence={state.draft.confidence:.2f}, groundedness={g:.2f})")
    return kind, reasons, sorted({e.doc_id for e in confidential if e.doc_id})


def make_approval_gate(deps: AgentDeps):
    @traced_node("approval_gate")
    def approval_gate(state: ResearchState) -> dict:
        kind, reasons, conf_docs = approval_reasons(state, deps)
        if kind is None:
            return {}
        decision = interrupt({
            "kind": "approval",
            "approval_type": kind.value,
            "reason": "; ".join(reasons),
            "details": {
                "question": state.standalone_query or state.query,
                "draft_answer": state.draft.answer if state.draft else "",
                "confidence": state.draft.confidence if state.draft else None,
                "groundedness": state.groundedness.score if state.groundedness else None,
                "confidential_documents": conf_docs,
                "user_id": state.user_id,
            },
        })
        approved = isinstance(decision, dict) and decision.get("decision") == "approve"
        tag = f"{kind.value}:{'approved' if approved else 'rejected'}"
        if approved:
            return {"approvals": [tag]}
        comment = decision.get("comment", "") if isinstance(decision, dict) else ""
        text = REJECTED_ANSWER.format(reason=f" ({comment})" if comment else "")
        return {"approvals": [tag], "status": "rejected", "final_answer": text}

    return approval_gate


def render_final(state: ResearchState) -> str:
    draft = state.draft
    parts = [draft.answer.strip() if draft else ""]
    if state.conflicts:
        parts.append("\n### ⚠️ Conflicting information between internal and external sources")
        for c in state.conflicts:
            parts.append(
                f"- **{c.topic}** — Internal: “{c.internal_claim}” {''.join(f'[{s}]' for s in c.internal_source_ids)} "
                f"vs. External: “{c.external_claim}” {''.join(f'[{s}]' for s in c.external_source_ids)}. "
                f"_Assessment:_ {c.likely_resolution}"
            )
    if state.groundedness and state.groundedness.verdicts:
        bad = [v.claim for v in state.groundedness.verdicts if not v.supported]
        if bad:
            parts.append("\n> ⚠️ The following statements could not be fully verified against the cited sources:\n"
                         + "\n".join(f"> - {b}" for b in bad))
    if draft and draft.gaps and draft.claims:
        parts.append("\n**Not covered by the available sources:** " + "; ".join(draft.gaps))
    used = used_source_ids(state)
    srcs = [e for e in state.evidence if e.source_id in used]
    if srcs:
        parts.append("\n**Sources**")
        parts.extend(f"- {e.citation_label()}" for e in srcs)
    return "\n".join(p for p in parts if p is not None).strip()


def make_finalize(deps: AgentDeps):
    @traced_node("finalize")
    def finalize(state: ResearchState) -> dict:
        if state.status == "rejected":
            text = state.final_answer
            return {"messages": [AIMessage(content=text)]}
        text = render_final(state)
        return {"final_answer": text, "status": "answered", "messages": [AIMessage(content=text)]}

    return finalize
