"""Shared fixtures. Tests run fully offline: MockLLM, mock web search and hashing embeddings."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from research_agent.config import PROJECT_ROOT, Settings
from research_agent.llm.base import LLMClient
from research_agent.llm.mock_llm import MockLLM
from research_agent.persistence.db import Database
from research_agent.rag.embeddings import HashingEmbeddings
from research_agent.rag.vectorstore import KnowledgeBase
from research_agent.service import ResearchService, build_service
from research_agent.tools.web_search import MockWebSearch

FIXTURE_WEB = PROJECT_ROOT / "data" / "mock_web" / "results.json"


def make_settings(tmp: Path, **overrides: Any) -> Settings:
    base = dict(
        llm_provider="mock", search_provider="mock", embedding_provider="hash",
        chroma_dir=tmp / "chroma", sqlite_path=tmp / "app.db", checkpoint_path=tmp / "cp.db",
        log_level="WARNING", tool_retry_base_delay_s=0.01, collection_name="test_kb",
    )
    base.update(overrides)
    s = Settings(_env_file=None, **base)
    s.ensure_dirs()
    return s


@pytest.fixture(scope="session")
def kb_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("kb")


@pytest.fixture(scope="session")
def shared_kb(kb_dir) -> KnowledgeBase:
    s = make_settings(kb_dir)
    kb = KnowledgeBase(s, HashingEmbeddings())
    kb.ingest_directory(s.docs_dir, reset=True)
    return kb


class ScriptedLLM(LLMClient):
    """MockLLM with per-purpose overrides. An override is a callable(context) -> model, or an
    exception instance to raise. Lists are consumed one item per call (last item repeats)."""

    name = "scripted"

    def __init__(self, overrides: dict[str, Any] | None = None):
        self.mock = MockLLM()
        self.overrides = overrides or {}
        self.calls: list[str] = []

    def structured(self, schema, system, prompt, *, purpose, tier="fast", context=None):
        self.calls.append(purpose)
        ov = self.overrides.get(purpose)
        if isinstance(ov, list):
            ov = ov.pop(0) if len(ov) > 1 else ov[0]
        if ov is None:
            return self.mock.structured(schema, system, prompt, purpose=purpose, tier=tier, context=context)
        if isinstance(ov, BaseException):
            raise ov
        if isinstance(ov, Callable):
            return ov(context or {})
        return ov


@pytest.fixture
def make_service(tmp_path, shared_kb) -> Callable[..., ResearchService]:
    def _make(llm: LLMClient | None = None, web=None, kb: KnowledgeBase | None = None, **settings_overrides) -> ResearchService:
        s = make_settings(tmp_path, **settings_overrides)
        return build_service(
            s, llm=llm or MockLLM(), web=web or MockWebSearch(FIXTURE_WEB), kb=kb or shared_kb,
            db=Database(s.sqlite_path), auto_ingest=False,
        )

    return _make


@pytest.fixture
def service(make_service) -> ResearchService:
    return make_service()
