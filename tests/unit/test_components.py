"""Unit tests for individual components (no graph execution)."""

from __future__ import annotations

import pytest

from research_agent.agents.privacy import find_sensitive_terms
from research_agent.agents.synthesizer import cited_ids, sanitize_citations
from research_agent.agents.validator import merge_and_number
from research_agent.config import PROJECT_ROOT
from research_agent.graph.state import RESET, append_or_reset
from research_agent.llm.mock_llm import MockLLM
from research_agent.persistence.db import Database
from research_agent.rag.embeddings import HashingEmbeddings
from research_agent.rag.loader import chunk_document, load_markdown
from research_agent.resilience import NonRetryableError, ToolError, call_with_retry
from research_agent.schemas import Claim, Evidence, QueryClassification, Route, SynthesizedAnswer
from research_agent.text_utils import content_tokens, numbers_in, overlap, sentences


def ev(i: int, kind="internal", content="text", score=0.5, **kw) -> Evidence:
    return Evidence(kind=kind, title=f"t{i}", content=content, doc_id=f"d{i}", score=score, **kw)


# ----------------------------------------------------------------------------- text utils / embeddings
def test_text_utils_basics():
    assert "paynova" in content_tokens("How do PayNova's fees compare?")
    assert numbers_in("PayNova charges 2.4% + $0.20") >= {"2.4%", "$0.20"}
    assert overlap("parental leave weeks", "20 weeks of parental leave") == 1.0
    assert all(not s.startswith("#") for s in sentences("# Heading\nThis is a sufficiently long sentence here."))


def test_hashing_embeddings_are_normalised_and_lexical():
    e = HashingEmbeddings()
    a, b, c = e.embed_query("parental leave policy"), e.embed_query("parental leave weeks"), e.embed_query("api rate limit")
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
    assert abs(dot(a, a) - 1) < 1e-6
    assert dot(a, b) > dot(a, c)


# ----------------------------------------------------------------------------- loader
def test_loader_parses_front_matter_and_chunks():
    doc = load_markdown(PROJECT_ROOT / "data" / "docs" / "project-falcon.md")
    assert doc.classification == "confidential" and "Project Falcon" in doc.sensitive_terms
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 3
    assert all(c.metadata["doc_id"] == "project-falcon" and c.metadata["classification"] == "confidential" for c in chunks)
    assert any("Status" in c.metadata["section"] for c in chunks)


# ----------------------------------------------------------------------------- state reducer
def test_append_or_reset_reducer():
    assert append_or_reset(["a"], ["b"]) == ["a", "b"]
    assert append_or_reset(["a", "b"], [RESET]) == []
    assert append_or_reset(["a"], [RESET, "c"]) == ["c"]


# ----------------------------------------------------------------------------- validator helpers
def test_merge_and_number_dedupes_and_orders_internal_first():
    merged = merge_and_number([ev(1, content="same", score=0.2), ev(2, content="other", score=0.9)],
                              [ev(3, kind="web", content="same"), ev(4, kind="web", content="web only")])
    assert [e.source_id for e in merged] == ["S1", "S2", "S3"]
    assert [e.kind for e in merged] == ["internal", "internal", "web"]
    assert merged[0].content == "other"  # higher score first


# ----------------------------------------------------------------------------- citations
def test_sanitize_citations_removes_unknown_sources():
    draft = SynthesizedAnswer(answer="Fact A [S1]. Fact B [S9].", claims=[Claim(text="B", source_ids=["S9", "S1"])], confidence=0.8)
    clean, warnings = sanitize_citations(draft, {"S1"})
    assert cited_ids(clean.answer) == {"S1"}
    assert clean.claims[0].source_ids == ["S1"]
    assert warnings


# ----------------------------------------------------------------------------- privacy
def test_find_sensitive_terms_is_case_insensitive_and_word_bounded():
    terms = {"Project Falcon", "Series D"}
    assert find_sensitive_terms("news about project falcon deal", terms) == ["Project Falcon"]
    assert find_sensitive_terms("series data analysis", terms) == []


# ----------------------------------------------------------------------------- retry
def test_call_with_retry_recovers_from_transient_errors():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert call_with_retry("t", flaky, max_attempts=3, base_delay=0.001) == "ok"
    assert calls["n"] == 3


def test_call_with_retry_gives_up_and_wraps_error():
    with pytest.raises(ToolError) as exc:
        call_with_retry("t", lambda: (_ for _ in ()).throw(TimeoutError("slow")), max_attempts=2, base_delay=0.001)
    assert exc.value.tool == "t"


def test_non_retryable_errors_are_not_retried():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise NonRetryableError("400")

    with pytest.raises(ToolError):
        call_with_retry("t", bad, max_attempts=5, base_delay=0.001)
    assert calls["n"] == 1


# ----------------------------------------------------------------------------- mock classifier
@pytest.mark.parametrize("query,route", [
    ("What is our parental leave policy?", Route.INTERNAL),
    ("What are the latest trends in real-time payments?", Route.WEB),
    ("How do PayNova's fees compare to ours?", Route.BOTH),
    ("What's the limit?", Route.CLARIFY),
    ("hello", Route.DIRECT),
])
def test_mock_classifier_routes(query, route):
    out = MockLLM().structured(QueryClassification, "", "", purpose="classify", context={"query": query})
    assert out.route == route


def test_mock_classifier_stops_clarifying_after_limit():
    out = MockLLM().structured(QueryClassification, "", "", purpose="classify",
                               context={"query": "What's the limit?", "clarification_count": 2, "max_clarifications": 2})
    assert out.route != Route.CLARIFY


# ----------------------------------------------------------------------------- database
def test_approval_lifecycle(tmp_path):
    db = Database(tmp_path / "x.db")
    aid = db.create_approval("kb_write", {"doc_id": "d"}, "reason", requested_by="me")
    assert db.get_approval(aid)["status"] == "pending"
    assert [a["approval_id"] for a in db.list_approvals("pending")] == [aid]
    db.decide_approval(aid, approved=False, reviewer="boss", comment="no")
    a = db.get_approval(aid)
    assert a["status"] == "rejected" and a["reviewer"] == "boss"
    db.decide_approval(aid, approved=True, reviewer="boss")  # already decided -> unchanged
    assert db.get_approval(aid)["status"] == "rejected"


def test_metrics_summary_aggregates(tmp_path):
    db = Database(tmp_path / "x.db")
    db.record_node("t", "turn1", "classify", 12.0, "ok")
    db.record_node("t", "turn1", "classify", 8.0, "error", "x")
    db.record_llm_call("classify", "m", 10, 5, 100.0, "ok")
    m = db.metrics_summary()
    assert m["turns"] == 1
    assert m["nodes"][0]["runs"] == 2 and m["nodes"][0]["errors"] == 1
    assert m["llm"][0]["input_tokens"] == 10
