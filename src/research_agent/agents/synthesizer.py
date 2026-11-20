"""Synthesis agent (cited answer) and groundedness / hallucination checker."""

from __future__ import annotations

import re

from research_agent.agents import prompts
from research_agent.agents.deps import AgentDeps
from research_agent.graph.state import ResearchState
from research_agent.llm.mock_llm import MockLLM
from research_agent.observability.logging import get_logger
from research_agent.observability.tracing import traced_node
from research_agent.schemas import ClaimVerdict, GroundednessReport, SynthesizedAnswer

log = get_logger("research_agent.agents.synthesizer")
CITE = re.compile(r"\[(S\d+)\]")

NO_EVIDENCE_ANSWER = (
    "I couldn't find reliable information to answer this question in the internal knowledge base"
    "{web_part}. Rather than guess, I'm not providing an answer. You could rephrase the question, "
    "name the specific policy/product/document you mean, or ask the relevant team directly."
)


def cited_ids(text: str) -> set[str]:
    return set(CITE.findall(text))


def sanitize_citations(draft: SynthesizedAnswer, valid_ids: set[str]) -> tuple[SynthesizedAnswer, list[str]]:
    """Remove citation markers / claim sources that reference non-existent evidence."""
    bad = cited_ids(draft.answer) - valid_ids
    answer = draft.answer
    for b in bad:
        answer = answer.replace(f"[{b}]", "")
    claims = [c.model_copy(update={"source_ids": [s for s in c.source_ids if s in valid_ids]}) for c in draft.claims]
    warnings = [f"Removed citations to unknown sources: {sorted(bad)}"] if bad else []
    return draft.model_copy(update={"answer": answer, "claims": claims}), warnings


def make_synthesize(deps: AgentDeps):
    @traced_node("synthesize")
    def synthesize(state: ResearchState) -> dict:
        attempts = state.synthesis_attempts + 1
        query = state.standalone_query or state.query
        if not state.evidence:
            web_part = " or on the web" if state.route in ("web", "both") and state.web_allowed else ""
            draft = SynthesizedAnswer(answer=NO_EVIDENCE_ANSWER.format(web_part=web_part), claims=[], confidence=0.0, gaps=[query])
            return {"draft": draft, "synthesis_attempts": attempts}

        feedback = f"\n\nREVISION REQUIRED. A reviewer found these claims unsupported by their cited sources; remove or correct them:\n{state.synthesis_feedback}" if state.synthesis_feedback else ""
        context = {"query": query, "evidence": state.evidence, "conflicts": state.conflicts, "feedback": state.synthesis_feedback}
        prompt = (
            f"Question: {query}\n\nConversation summary: {state.summary or '(none)'}\n\n"
            f"Sources:\n{prompts.render_evidence(state.evidence)}\n\n"
            f"Detected internal-vs-external conflicts (mention them; a structured conflict section is appended automatically):\n"
            f"{prompts.render_conflicts(state.conflicts)}{feedback}"
        )
        warnings: list[str] = []
        try:
            draft = deps.llm.structured(SynthesizedAnswer, prompts.SYNTHESIZE, prompt, purpose="synthesize", tier="smart", context=context)
        except Exception as exc:  # noqa: BLE001 - degrade to extractive answer
            log.warning("synthesis_failed_using_extractive_fallback", error=str(exc))
            draft = MockLLM()._synthesize(context)
            draft = draft.model_copy(update={"confidence": min(draft.confidence, 0.4)})
            warnings.append("The answer model was unavailable; this is an extractive summary of the sources.")
        draft, w = sanitize_citations(draft, {e.source_id for e in state.evidence})
        return {"draft": draft, "synthesis_attempts": attempts, "warnings": warnings + w}

    return synthesize


def make_check_groundedness(deps: AgentDeps):
    @traced_node("check_groundedness")
    def check_groundedness(state: ResearchState) -> dict:
        draft = state.draft
        if draft is None or not draft.claims:
            return {"groundedness": GroundednessReport(verdicts=[], score=1.0 if draft and not state.evidence else 0.0)}

        # Deterministic check: every claim must cite at least one known source.
        uncited = [c for c in draft.claims if not c.source_ids]
        to_check = [c for c in draft.claims if c.source_ids]
        verdicts = [ClaimVerdict(claim=c.text, supported=False, reason="no citation") for c in uncited]
        try:
            if to_check:
                cited = {s for c in to_check for s in c.source_ids}
                ev = [e for e in state.evidence if e.source_id in cited]
                report = deps.llm.structured(
                    GroundednessReport, prompts.GROUNDEDNESS,
                    f"Claims:\n{prompts.render_claims(to_check)}\n\nSources:\n{prompts.render_evidence(ev)}",
                    purpose="groundedness", tier="fast", context={"claims": to_check, "evidence": ev},
                )
                verdicts += report.verdicts
        except Exception as exc:  # noqa: BLE001
            log.warning("groundedness_check_failed", error=str(exc))
            return {"groundedness": GroundednessReport(verdicts=verdicts, score=0.5),
                    "warnings": ["Hallucination check unavailable; answer could not be fully verified."]}

        score = sum(v.supported for v in verdicts) / len(verdicts) if verdicts else 1.0
        report = GroundednessReport(verdicts=verdicts, score=round(score, 3))
        unsupported = [v for v in verdicts if not v.supported]
        update: dict = {"groundedness": report,
                        "synthesis_feedback": "\n".join(f"- {v.claim} ({v.reason})" for v in unsupported)}
        log.info("groundedness", score=report.score, unsupported=len(unsupported))
        return update

    return check_groundedness
