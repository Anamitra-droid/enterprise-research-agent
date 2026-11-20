"""Unit tests for the graph's conditional routing functions."""

from __future__ import annotations

from research_agent.graph.builder import (
    fan_out,
    make_route_after_groundedness,
    make_route_after_validate,
    route_after_classify,
    route_after_synthesize,
)
from research_agent.graph.state import ResearchState
from research_agent.schemas import Claim, Evidence, GroundednessReport, Route, SynthesizedAnswer


def st(**kw) -> ResearchState:
    return ResearchState(**kw)


def test_route_after_classify():
    assert route_after_classify(st(route=Route.CLARIFY)) == "clarify"
    assert route_after_classify(st(route=Route.DIRECT)) == "respond_direct"
    for r in (Route.INTERNAL, Route.WEB, Route.BOTH):
        assert route_after_classify(st(route=r)) == "privacy_guard"


def test_fan_out_parallel_and_privacy_fallback():
    assert fan_out(st(route=Route.BOTH)) == ["retrieve_internal", "search_web"]
    assert fan_out(st(route=Route.INTERNAL)) == ["retrieve_internal"]
    assert fan_out(st(route=Route.WEB)) == ["search_web"]
    # web search vetoed by reviewer -> fall back to internal only
    assert fan_out(st(route=Route.WEB, web_allowed=False)) == ["retrieve_internal"]
    assert fan_out(st(route=Route.BOTH, web_allowed=False)) == ["retrieve_internal"]


def test_route_after_validate():
    r = make_route_after_validate(max_attempts=2)
    assert r(st(evidence=[], retrieval_attempts=1)) == "reformulate"
    assert r(st(evidence=[], retrieval_attempts=2)) == "synthesize"  # give up -> abstain
    i = Evidence(kind="internal", title="a", content="x")
    w = Evidence(kind="web", title="b", content="y")
    assert r(st(evidence=[i], retrieval_attempts=1)) == "synthesize"
    assert r(st(evidence=[i, w], retrieval_attempts=1)) == "detect_conflicts"


def test_route_after_synthesize_and_groundedness():
    with_claims = SynthesizedAnswer(answer="a [S1]", claims=[Claim(text="a", source_ids=["S1"])], confidence=0.8)
    no_claims = SynthesizedAnswer(answer="no info", claims=[], confidence=0.0)
    assert route_after_synthesize(st(draft=with_claims)) == "check_groundedness"
    assert route_after_synthesize(st(draft=no_claims)) == "approval_gate"

    r = make_route_after_groundedness(threshold=0.7, max_attempts=2)
    low = GroundednessReport(verdicts=[], score=0.3)
    high = GroundednessReport(verdicts=[], score=0.95)
    assert r(st(groundedness=low, synthesis_feedback="- bad", synthesis_attempts=1)) == "synthesize"
    assert r(st(groundedness=low, synthesis_feedback="- bad", synthesis_attempts=2)) == "approval_gate"
    assert r(st(groundedness=high, synthesis_feedback="", synthesis_attempts=1)) == "approval_gate"
