"""LangGraph wiring: nodes, conditional routing, parallel fan-out and loops."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from research_agent.agents.approval import make_approval_gate, make_finalize
from research_agent.agents.conflicts import make_detect_conflicts
from research_agent.agents.conversation import make_contextualize
from research_agent.agents.deps import AgentDeps
from research_agent.agents.privacy import make_privacy_guard
from research_agent.agents.retrievers import make_retrieve_internal, make_search_web
from research_agent.agents.router import make_clarify, make_classify, make_respond_direct
from research_agent.agents.synthesizer import make_check_groundedness, make_synthesize
from research_agent.agents.validator import make_reformulate, make_validate
from research_agent.graph.state import ResearchState
from research_agent.schemas import Route

# ----------------------------------------------------------------------------- routers


def route_after_classify(state: ResearchState) -> str:
    if state.route == Route.CLARIFY:
        return "clarify"
    if state.route == Route.DIRECT:
        return "respond_direct"
    return "privacy_guard"


def fan_out(state: ResearchState) -> list[str]:
    """Pick retrieval agents; runs them in parallel when both are needed."""
    targets: list[str] = []
    if state.route in (Route.INTERNAL, Route.BOTH) or (state.route == Route.WEB and not state.web_allowed):
        targets.append("retrieve_internal")
    if state.route in (Route.WEB, Route.BOTH) and state.web_allowed:
        targets.append("search_web")
    return targets or ["retrieve_internal"]


def make_route_after_validate(max_attempts: int):
    def route_after_validate(state: ResearchState) -> str:
        if not state.evidence and state.retrieval_attempts < max_attempts:
            return "reformulate"
        kinds = {e.kind for e in state.evidence}
        if {"internal", "web"} <= kinds:
            return "detect_conflicts"
        return "synthesize"

    return route_after_validate


def route_after_synthesize(state: ResearchState) -> str:
    return "check_groundedness" if state.draft and state.draft.claims else "approval_gate"


def make_route_after_groundedness(threshold: float, max_attempts: int):
    def route_after_groundedness(state: ResearchState) -> str:
        score = state.groundedness.score if state.groundedness else 1.0
        if score < threshold and state.synthesis_feedback and state.synthesis_attempts < max_attempts:
            return "synthesize"  # regenerate without the unsupported claims
        return "approval_gate"

    return route_after_groundedness


# ----------------------------------------------------------------------------- builder


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None):
    s = deps.settings
    g = StateGraph(ResearchState)

    g.add_node("contextualize", make_contextualize(deps))
    g.add_node("classify", make_classify(deps))
    g.add_node("clarify", make_clarify(deps))
    g.add_node("respond_direct", make_respond_direct(deps))
    g.add_node("privacy_guard", make_privacy_guard(deps))
    g.add_node("retrieve_internal", make_retrieve_internal(deps))
    g.add_node("search_web", make_search_web(deps))
    g.add_node("validate", make_validate(deps))
    g.add_node("reformulate", make_reformulate(deps))
    g.add_node("detect_conflicts", make_detect_conflicts(deps))
    g.add_node("synthesize", make_synthesize(deps))
    g.add_node("check_groundedness", make_check_groundedness(deps))
    g.add_node("approval_gate", make_approval_gate(deps))
    g.add_node("finalize", make_finalize(deps))

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "classify")
    g.add_conditional_edges("classify", route_after_classify, ["clarify", "respond_direct", "privacy_guard"])
    g.add_edge("clarify", "classify")
    g.add_edge("respond_direct", END)
    g.add_conditional_edges("privacy_guard", fan_out, ["retrieve_internal", "search_web"])
    g.add_edge("retrieve_internal", "validate")
    g.add_edge("search_web", "validate")
    g.add_conditional_edges("validate", make_route_after_validate(s.max_retrieval_attempts),
                            ["reformulate", "detect_conflicts", "synthesize"])
    g.add_edge("reformulate", "privacy_guard")
    g.add_edge("detect_conflicts", "synthesize")
    g.add_conditional_edges("synthesize", route_after_synthesize, ["check_groundedness", "approval_gate"])
    g.add_conditional_edges("check_groundedness",
                            make_route_after_groundedness(s.groundedness_threshold, s.max_synthesis_attempts),
                            ["synthesize", "approval_gate"])
    g.add_edge("approval_gate", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
