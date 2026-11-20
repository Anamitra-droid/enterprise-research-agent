"""Conversation-memory agent: summarises long histories and rewrites follow-ups."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from research_agent.agents import prompts
from research_agent.agents.deps import AgentDeps
from research_agent.graph.state import ResearchState
from research_agent.observability.logging import get_logger
from research_agent.observability.tracing import traced_node
from research_agent.schemas import ConversationSummary, StandaloneQuery

log = get_logger("research_agent.agents.conversation")


def _role(m) -> str:
    return "user" if isinstance(m, HumanMessage) else "assistant" if isinstance(m, AIMessage) else "system"


def make_contextualize(deps: AgentDeps):
    s = deps.settings

    @traced_node("contextualize")
    def contextualize(state: ResearchState) -> dict:
        update: dict = {}
        history = state.messages[:-1]  # last message is the current user turn
        summary = state.summary

        # 1) Compress long histories into a running summary (keeps context windows bounded).
        if len(state.messages) > s.history_summarize_after:
            old = history[: -s.history_keep_last] if s.history_keep_last else history
            if old:
                try:
                    pairs = [(_role(m), str(m.content)) for m in old]
                    out = deps.llm.structured(
                        ConversationSummary, prompts.SUMMARIZE,
                        f"Previous summary:\n{summary or '(none)'}\n\nMessages:\n" + "\n".join(f"{r}: {t}" for r, t in pairs),
                        purpose="summarize", tier="fast", context={"history": pairs, "previous_summary": summary},
                    )
                    summary = out.summary
                    update["summary"] = summary
                    update["messages"] = [RemoveMessage(id=m.id) for m in old if m.id]
                    history = history[len(old):]
                except Exception as exc:  # noqa: BLE001 - memory compression is best-effort
                    log.warning("summarize_failed", error=str(exc))

        # 2) Rewrite follow-ups into standalone questions.
        if not history and not summary:
            update["standalone_query"] = state.query
            return update
        pairs = [(_role(m), str(m.content)) for m in history[-s.history_keep_last:]]
        try:
            out = deps.llm.structured(
                StandaloneQuery, prompts.CONTEXTUALIZE,
                f"Conversation summary: {summary or '(none)'}\n\nRecent history:\n"
                + "\n".join(f"{r}: {t[:800]}" for r, t in pairs)
                + f"\n\nLatest user message: {state.query}",
                purpose="contextualize", tier="fast", context={"query": state.query, "history": pairs},
            )
            update["standalone_query"] = out.standalone_query.strip() or state.query
        except Exception as exc:  # noqa: BLE001
            log.warning("contextualize_failed", error=str(exc))
            update["standalone_query"] = state.query
            update["warnings"] = ["Could not resolve conversation context; interpreted the message on its own."]
        return update

    return contextualize
