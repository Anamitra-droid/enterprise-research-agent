"""Retrieval agents: internal RAG over ChromaDB and external web search. Run in parallel."""

from __future__ import annotations

from research_agent.agents.deps import AgentDeps
from research_agent.graph.state import ResearchState
from research_agent.observability.logging import get_logger
from research_agent.observability.tracing import traced_node
from research_agent.resilience import ToolError, call_with_retry

log = get_logger("research_agent.agents.retrieval")


def make_retrieve_internal(deps: AgentDeps):
    s = deps.settings

    @traced_node("retrieve_internal")
    def retrieve_internal(state: ResearchState) -> dict:
        query = state.internal_query or state.standalone_query or state.query
        try:
            docs = call_with_retry("vector_search", lambda: deps.kb.search(query, k=s.retrieval_k),
                                   max_attempts=s.tool_max_retries, base_delay=s.tool_retry_base_delay_s)
        except ToolError as exc:
            log.error("internal_retrieval_failed", error=str(exc))
            return {"internal_evidence": [], "warnings": ["Internal knowledge base search failed; internal documents were not consulted."]}
        log.info("internal_retrieved", n=len(docs), query=query)
        return {"internal_evidence": docs}

    return retrieve_internal


def make_search_web(deps: AgentDeps):
    s = deps.settings

    @traced_node("search_web")
    def search_web(state: ResearchState) -> dict:
        query = state.web_query or state.standalone_query or state.query
        try:
            docs = call_with_retry("web_search", lambda: deps.web.search(query, max_results=s.web_max_results),
                                   max_attempts=s.tool_max_retries, base_delay=s.tool_retry_base_delay_s)
        except ToolError as exc:
            log.error("web_search_failed", error=str(exc))
            return {"web_evidence": [], "warnings": ["External web search is currently unavailable; the answer relies on internal sources only."]}
        log.info("web_retrieved", n=len(docs), query=query)
        return {"web_evidence": docs}

    return search_web
