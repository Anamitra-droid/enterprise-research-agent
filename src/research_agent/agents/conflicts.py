"""Conflict-detection agent: finds disagreements between internal and external sources."""

from __future__ import annotations

from research_agent.agents import prompts
from research_agent.agents.deps import AgentDeps
from research_agent.graph.state import ResearchState
from research_agent.observability.logging import get_logger
from research_agent.observability.tracing import traced_node
from research_agent.schemas import ConflictReport

log = get_logger("research_agent.agents.conflicts")


def make_detect_conflicts(deps: AgentDeps):
    @traced_node("detect_conflicts")
    def detect_conflicts(state: ResearchState) -> dict:
        internal = [e for e in state.evidence if e.kind == "internal"]
        web = [e for e in state.evidence if e.kind == "web"]
        if not internal or not web:
            return {"conflicts": []}
        query = state.standalone_query or state.query
        try:
            report = deps.llm.structured(
                ConflictReport, prompts.CONFLICTS,
                f"Question: {query}\n\nINTERNAL sources:\n{prompts.render_evidence(internal)}\n\nEXTERNAL sources:\n{prompts.render_evidence(web)}",
                purpose="conflicts", tier="smart", context={"query": query, "internal": internal, "web": web},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("conflict_detection_failed", error=str(exc))
            return {"conflicts": [], "warnings": ["Conflict detection unavailable; internal/external disagreements were not checked."]}
        internal_ids = {e.source_id for e in internal}
        web_ids = {e.source_id for e in web}
        valid = [
            c for c in report.conflicts
            if set(c.internal_source_ids) & internal_ids and set(c.external_source_ids) & web_ids
        ]
        log.info("conflicts_detected", n=len(valid))
        return {"conflicts": valid}

    return detect_conflicts
