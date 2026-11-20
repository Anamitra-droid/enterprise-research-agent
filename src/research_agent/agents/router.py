"""Query classification / routing, clarification (interrupt) and direct replies."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from research_agent.agents import prompts
from research_agent.agents.deps import AgentDeps
from research_agent.graph.state import ResearchState
from research_agent.observability.logging import get_logger
from research_agent.observability.tracing import traced_node
from research_agent.schemas import QueryClassification, Route

log = get_logger("research_agent.agents.router")


def make_classify(deps: AgentDeps):
    s = deps.settings

    @traced_node("classify")
    def classify(state: ResearchState) -> dict:
        query = state.standalone_query or state.query
        can_clarify = state.clarification_count < s.max_clarifications
        try:
            c = deps.llm.structured(
                QueryClassification, prompts.CLASSIFY,
                f"Conversation summary: {state.summary or '(none)'}\n"
                f"Clarifications already asked this turn: {state.clarification_count} (max {s.max_clarifications})"
                f"{'' if can_clarify else ' - do NOT choose clarify; pick the most likely interpretation.'}\n\n"
                f"Query: {query}",
                purpose="classify", tier="fast",
                context={"query": query, "clarification_count": state.clarification_count, "max_clarifications": s.max_clarifications},
            )
        except Exception as exc:  # noqa: BLE001 - fail open to the broadest safe route
            log.warning("classify_failed", error=str(exc))
            c = QueryClassification(route=Route.BOTH, reasoning="classifier unavailable; searching all sources",
                                    needs_current_info=False, internal_query=query, web_query=query)
            return {"classification": c, "route": c.route, "internal_query": query, "web_query": query,
                    "warnings": ["Query classifier unavailable; searched both internal and web sources."]}

        if c.route == Route.CLARIFY and (not can_clarify or not c.clarification_question.strip()):
            c = c.model_copy(update={"route": Route.BOTH})
        internal_q = c.internal_query.strip() or query
        web_q = c.web_query.strip() or query
        log.info("classified", route=c.route.value, reasoning=c.reasoning)
        return {"classification": c, "route": c.route, "internal_query": internal_q, "web_query": web_q,
                "clarification_question": c.clarification_question}

    return classify


def make_clarify(deps: AgentDeps):
    @traced_node("clarify")
    def clarify(state: ResearchState) -> dict:
        question = state.clarification_question or "Could you clarify your question?"
        # Pauses the graph; the API returns status=needs_clarification. Resumed via Command(resume=...).
        reply = interrupt({"kind": "clarification", "question": question})
        answer = reply.get("answer", "") if isinstance(reply, dict) else str(reply)
        base = state.standalone_query or state.query
        return {
            "messages": [AIMessage(content=question), HumanMessage(content=answer)],
            "standalone_query": f"{base} — clarification: {answer}",
            "clarification_count": state.clarification_count + 1,
        }

    return clarify


def make_respond_direct(deps: AgentDeps):
    @traced_node("respond_direct")
    def respond_direct(state: ResearchState) -> dict:
        text = (state.classification.direct_answer if state.classification else "") or (
            "I'm the LedgerLeaf research assistant. I can answer questions using internal company documents and the web, with citations."
        )
        return {"final_answer": text, "status": "answered", "messages": [AIMessage(content=text)]}

    return respond_direct
