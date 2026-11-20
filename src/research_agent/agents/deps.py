"""Dependency bundle injected into every agent node (makes nodes trivially testable)."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.config import Settings
from research_agent.llm.base import LLMClient
from research_agent.rag.vectorstore import KnowledgeBase
from research_agent.tools.web_search import WebSearch


@dataclass
class AgentDeps:
    settings: Settings
    llm: LLMClient
    kb: KnowledgeBase
    web: WebSearch
