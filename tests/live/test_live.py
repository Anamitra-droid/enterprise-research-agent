"""Live end-to-end tests against the real OpenRouter + Tavily APIs.

Skipped by default. Run with:  pytest -m live
Requires OPENROUTER_API_KEY and TAVILY_API_KEY (in env or .env).
"""

from __future__ import annotations

import os

import pytest

from research_agent.config import get_settings

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_service():
    s = get_settings()
    if s.resolved_llm_provider != "openrouter" or s.resolved_search_provider != "tavily":
        pytest.skip("needs OPENROUTER_API_KEY and TAVILY_API_KEY")
    from research_agent.service import build_service

    os.environ.setdefault("LOG_LEVEL", "WARNING")
    return build_service(s)


def test_live_internal(live_service):
    r = live_service.chat("How many weeks of parental leave do primary caregivers get?")
    assert r.status == "answered" and r.citations
    assert "20" in r.answer


def test_live_hybrid_conflict(live_service):
    r = live_service.chat("Is our treasury policy's Fed funds rate assumption still accurate today?")
    assert r.route in ("both", "web")
    assert any(c.kind == "web" for c in r.citations)


def test_live_clarification(live_service):
    r = live_service.chat("What's the limit?")
    assert r.status in ("needs_clarification", "answered")
