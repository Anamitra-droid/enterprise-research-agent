"""Validation agent (relevance grading) and query reformulation on empty retrieval."""

from __future__ import annotations

import hashlib

from research_agent.agents import prompts
from research_agent.agents.deps import AgentDeps
from research_agent.graph.state import ResearchState
from research_agent.observability.logging import get_logger
from research_agent.observability.tracing import traced_node
from research_agent.schemas import Evidence, ReformulatedQuery, RelevanceGrades, Route

log = get_logger("research_agent.agents.validator")

MAX_EVIDENCE = 10
FALLBACK_MIN_SCORE = 0.35


def merge_and_number(internal: list[Evidence], web: list[Evidence]) -> list[Evidence]:
    """Dedupe passages and assign stable citation ids S1..Sn (internal first)."""
    seen: set[str] = set()
    merged: list[Evidence] = []
    for ev in [*sorted(internal, key=lambda e: -e.score), *sorted(web, key=lambda e: -e.score)]:
        key = hashlib.sha1(ev.content.strip().lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        merged.append(ev)
    return [ev.model_copy(update={"source_id": f"S{i + 1}"}) for i, ev in enumerate(merged[:MAX_EVIDENCE])]


def make_validate(deps: AgentDeps):
    @traced_node("validate")
    def validate(state: ResearchState) -> dict:
        candidates = merge_and_number(state.internal_evidence, state.web_evidence)
        attempts = state.retrieval_attempts + 1
        if not candidates:
            return {"evidence": [], "retrieval_attempts": attempts}
        query = state.standalone_query or state.query
        warnings: list[str] = []
        try:
            grades = deps.llm.structured(
                RelevanceGrades, prompts.GRADE,
                f"Question: {query}\n\nSources:\n{prompts.render_evidence(candidates, max_chars=1200)}",
                purpose="grade", tier="fast", context={"query": query, "evidence": candidates},
            )
            keep_ids = {g.source_id for g in grades.grades if g.relevant}
            kept = [e for e in candidates if e.source_id in keep_ids]
        except Exception as exc:  # noqa: BLE001 - degrade to score threshold
            log.warning("grading_failed", error=str(exc))
            kept = [e for e in candidates if e.score >= FALLBACK_MIN_SCORE]
            warnings.append("Relevance validation unavailable; used similarity-score filtering instead.")
        log.info("validated", candidates=len(candidates), kept=len(kept), attempt=attempts)
        # Renumber so citations are contiguous.
        kept = [e.model_copy(update={"source_id": f"S{i + 1}"}) for i, e in enumerate(kept)]
        return {"evidence": kept, "retrieval_attempts": attempts, "warnings": warnings}

    return validate


def make_reformulate(deps: AgentDeps):
    @traced_node("reformulate")
    def reformulate(state: ResearchState) -> dict:
        query = state.standalone_query or state.query
        needs_internal = state.route in (Route.INTERNAL, Route.BOTH) or not state.web_allowed
        needs_web = state.route in (Route.WEB, Route.BOTH) and state.web_allowed
        try:
            r = deps.llm.structured(
                ReformulatedQuery, prompts.REFORMULATE,
                f"Original question: {query}\nPrevious internal query: {state.internal_query}\nPrevious web query: {state.web_query}\n"
                f"Needs internal: {needs_internal}. Needs web: {needs_web}.",
                purpose="reformulate", tier="fast",
                context={"query": query, "needs_internal": needs_internal, "needs_web": needs_web},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("reformulate_failed", error=str(exc))
            r = ReformulatedQuery(internal_query=query, web_query=query)
        return {
            "internal_query": r.internal_query or state.internal_query,
            "web_query": r.web_query or state.web_query,
            "warnings": ["Initial retrieval found nothing relevant; retried with a reformulated query."],
        }

    return reformulate
