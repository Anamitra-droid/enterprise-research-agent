"""External web search: Tavily (real) or an offline fixture corpus (mock)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from research_agent.config import Settings
from research_agent.schemas import Evidence
from research_agent.text_utils import content_tokens, overlap


class WebSearch(ABC):
    name = "base"

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[Evidence]: ...


class TavilyWebSearch(WebSearch):
    name = "tavily"

    def __init__(self, settings: Settings):
        from langchain_tavily import TavilySearch

        self._tool = TavilySearch(max_results=settings.web_max_results, search_depth="advanced", topic="general")

    def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        raw = self._tool.invoke({"query": query})
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict) and raw.get("error"):
            raise RuntimeError(f"Tavily error: {raw['error']}")
        results = raw.get("results", []) if isinstance(raw, dict) else []
        return [
            Evidence(
                kind="web",
                title=r.get("title") or r.get("url", "web result"),
                content=(r.get("content") or "")[:2000],
                url=r.get("url"),
                last_updated=(r.get("published_date") or None),
                score=float(r.get("score") or 0.0),
            )
            for r in results[:max_results]
            if r.get("content")
        ]


class MockWebSearch(WebSearch):
    """Keyword search over `data/mock_web/results.json` (clearly-labelled fictional pages)."""

    name = "mock"

    def __init__(self, fixture_path: Path, fail_times: int = 0):
        self._pages = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
        self._fail_times = fail_times  # for retry demos/tests

    def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ConnectionError("mock web search: simulated transient failure")
        q = content_tokens(query)
        scored = []
        for p in self._pages:
            s = overlap(q, p["title"] + " " + p["content"] + " " + " ".join(p.get("keywords", [])))
            if s > 0.2:
                scored.append((s, p))
        scored.sort(key=lambda x: -x[0])
        return [
            Evidence(kind="web", title=p["title"], content=p["content"], url=p["url"], last_updated=p.get("published_date"), score=round(s, 3))
            for s, p in scored[:max_results]
        ]


def build_web_search(settings: Settings) -> WebSearch:
    if settings.resolved_search_provider == "tavily":
        return TavilyWebSearch(settings)
    return MockWebSearch(settings.data_dir / "mock_web" / "results.json")
