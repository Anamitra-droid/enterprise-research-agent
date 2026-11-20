"""Privacy guard: human approval before confidential internal terms are sent to an external search engine."""

from __future__ import annotations

import re

from langgraph.types import interrupt

from research_agent.agents.deps import AgentDeps
from research_agent.graph.state import ResearchState
from research_agent.observability.tracing import traced_node
from research_agent.schemas import ApprovalType, Route


def find_sensitive_terms(text: str, terms: set[str]) -> list[str]:
    return sorted({t for t in terms if t and re.search(rf"\b{re.escape(t)}\b", text, re.I)})


def make_privacy_guard(deps: AgentDeps):
    s = deps.settings

    @traced_node("privacy_guard")
    def privacy_guard(state: ResearchState) -> dict:
        if state.route not in (Route.WEB, Route.BOTH) or not state.web_allowed:
            return {}
        if any(a.startswith(f"{ApprovalType.EXTERNAL_SEARCH_PRIVACY.value}:") for a in state.approvals):
            return {}  # already decided this turn (e.g. after a query reformulation)
        hits = find_sensitive_terms(state.web_query, deps.kb.sensitive_terms)
        if not hits:
            return {}
        if not s.hitl_web_privacy:
            # Policy without a human: strip the terms automatically.
            cleaned = state.web_query
            for t in hits:
                cleaned = re.sub(re.escape(t), "", cleaned, flags=re.I)
            return {"web_query": " ".join(cleaned.split()), "warnings": [f"Removed confidential terms from web query: {hits}"]}

        decision = interrupt({
            "kind": "approval",
            "approval_type": ApprovalType.EXTERNAL_SEARCH_PRIVACY.value,
            "reason": f"The external web search query contains confidential internal terms: {', '.join(hits)}.",
            "details": {"web_query": state.web_query, "sensitive_terms": hits},
        })
        approved = isinstance(decision, dict) and decision.get("decision") == "approve"
        tag = f"{ApprovalType.EXTERNAL_SEARCH_PRIVACY.value}:{'approved' if approved else 'rejected'}"
        if approved:
            return {"approvals": [tag]}
        update: dict = {"approvals": [tag], "web_allowed": False,
                        "warnings": ["External web search was skipped: reviewer did not approve sending confidential terms externally."]}
        if state.route == Route.WEB:
            update["internal_query"] = state.internal_query or state.standalone_query or state.query
        return update

    return privacy_guard
