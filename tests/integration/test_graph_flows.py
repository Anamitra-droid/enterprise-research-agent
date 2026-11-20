"""Integration tests: full graph runs through the service layer (offline mocks)."""

from __future__ import annotations

from research_agent.llm.mock_llm import MockLLM
from research_agent.rag.embeddings import HashingEmbeddings
from research_agent.rag.vectorstore import KnowledgeBase
from research_agent.schemas import Claim, GroundednessReport, SynthesizedAnswer
from research_agent.tools.web_search import MockWebSearch

from ..conftest import FIXTURE_WEB, ScriptedLLM, make_settings


def test_internal_question_is_answered_with_citations(service):
    r = service.chat("What is our parental leave policy for primary caregivers?")
    assert r.status == "answered" and r.route == "internal"
    assert "search_web" not in r.trace
    assert r.citations and all(c.kind == "internal" for c in r.citations)
    assert any(c.doc_id == "hr-leave-policy" for c in r.citations)
    assert "[S" in r.answer and "20 weeks" in r.answer
    assert r.groundedness == 1.0


def test_web_question_uses_web_search_only(service):
    r = service.chat("What are the latest trends in real-time payments adoption?")
    assert r.route == "web"
    assert "search_web" in r.trace and "retrieve_internal" not in r.trace
    assert r.citations and all(c.kind == "web" and c.url for c in r.citations)


def test_hybrid_question_runs_both_retrievers_and_detects_conflict(make_service):
    # Lexical test embeddings can surface a confidential chunk; that gate is covered by other tests.
    service = make_service(hitl_confidential=False)
    r = service.chat("How do PayNova's current fees compare to our pricing?")
    assert r.route == "both"
    assert {"retrieve_internal", "search_web", "detect_conflicts"} <= set(r.trace)
    assert {c.kind for c in r.citations} == {"internal", "web"}
    assert r.conflicts, "expected PayNova 2.4% vs 1.9% conflict"
    c = r.conflicts[0]
    assert "2.4%" in c.internal_claim and "1.9%" in c.external_claim
    assert "Conflicting information" in r.answer


def test_clarification_interrupt_and_resume(service):
    r = service.chat("What's the limit?")
    assert r.status == "needs_clarification" and r.clarification_question
    r2 = service.resume(r.thread_id, answer="the API rate limit")
    assert r2.status == "answered"
    assert "clarify" in r2.trace
    assert "100 requests per second" in r2.answer


def test_multi_turn_follow_up_uses_context(service):
    r1 = service.chat("What uptime does our SLA guarantee?")
    r2 = service.chat("And what service credits do customers get if we miss it?", thread_id=r1.thread_id)
    assert r2.status == "answered"
    assert any(c.doc_id == "sla" for c in r2.citations)
    thread = service.get_thread(r1.thread_id)
    assert len([m for m in thread["messages"] if m["role"] == "user"]) == 2


def test_confidential_answer_requires_approval_then_released(service):
    r = service.chat("What is the status of Project Falcon?", user_id="analyst")
    assert r.status == "pending_approval" and r.approval.approval_type == "confidential_data"
    assert "project-falcon" in r.approval.details["confidential_documents"]
    assert service.db.list_approvals("pending")[0]["approval_id"] == r.approval.approval_id
    out = service.decide_approval(r.approval.approval_id, "approve", reviewer="cfo")
    assert out["chat"]["status"] == "answered"
    assert service.db.get_approval(r.approval.approval_id)["status"] == "approved"


def test_confidential_answer_rejected_is_withheld(service):
    r = service.chat("What is the status of Project Falcon?")
    r2 = service.resume(r.thread_id, decision="reject", reviewer="legal", comment="not on deal team")
    assert r2.status == "rejected"
    assert "withheld" in r2.answer and "Brightwire" not in r2.answer
    assert r2.citations == []


def test_privacy_guard_blocks_external_search_on_reject(service):
    r = service.chat("Search the web for the latest news about Project Falcon")
    assert r.status == "pending_approval" and r.approval.approval_type == "external_search_privacy"
    assert "Project Falcon" in r.approval.details["sensitive_terms"]
    r2 = service.resume(r.thread_id, decision="reject", reviewer="legal")
    assert "search_web" not in r2.trace
    assert any("skipped" in w for w in r2.warnings)


def test_chat_on_paused_thread_is_rejected(service):
    import pytest

    from research_agent.service import InvalidStateError

    r = service.chat("What's the limit?")
    with pytest.raises(InvalidStateError):
        service.chat("another question", thread_id=r.thread_id)


def test_unanswerable_question_triggers_reformulation_and_abstains(make_service):
    svc = make_service()
    r = svc.chat("What is our stance on quantum blockchain mining?")
    assert r.status == "answered"
    assert "reformulate" in r.trace
    assert not r.citations
    assert "couldn't find" in r.answer.lower()


def test_web_search_transient_failure_is_retried(make_service):
    svc = make_service(web=MockWebSearch(FIXTURE_WEB, fail_times=1))
    r = svc.chat("What are the latest trends in real-time payments adoption?")
    assert r.status == "answered" and r.citations
    tools = {(t["tool"], t["status"]): t["n"] for t in svc.db.metrics_summary()["tools"]}
    assert tools.get(("web_search", "retry")) == 1 and tools.get(("web_search", "ok")) == 1


def test_web_search_outage_degrades_gracefully(make_service):
    svc = make_service(web=MockWebSearch(FIXTURE_WEB, fail_times=100), hitl_confidential=False)
    r = svc.chat("How do PayNova's current fees compare to our pricing?")
    assert r.status == "answered"
    assert any("unavailable" in w for w in r.warnings)
    assert r.citations and all(c.kind == "internal" for c in r.citations)


def test_hallucinated_claims_trigger_regeneration(make_service):
    bad = SynthesizedAnswer(answer="PTO is 40 days [S1].", claims=[Claim(text="PTO is 40 days", source_ids=["S1"])], confidence=0.9)
    llm = ScriptedLLM({"synthesize": [lambda ctx: bad, lambda ctx: MockLLM()._synthesize(ctx)],
                       "groundedness": [lambda ctx: GroundednessReport(verdicts=[], score=0.0), lambda ctx: MockLLM()._groundedness(ctx)]})
    # first groundedness call: unsupported verdict
    from research_agent.schemas import ClaimVerdict

    llm.overrides["groundedness"][0] = lambda ctx: GroundednessReport(
        verdicts=[ClaimVerdict(claim="PTO is 40 days", supported=False, reason="not in source")], score=0.0)
    svc = make_service(llm=llm)
    r = svc.chat("How many PTO days do employees get?")
    assert r.trace.count("synthesize") == 2, r.trace
    assert "40 days" not in r.answer
    assert r.groundedness == 1.0


def test_llm_outage_in_synthesis_falls_back_to_extractive_answer(make_service):
    llm = ScriptedLLM({"synthesize": RuntimeError("openrouter down")})
    svc = make_service(llm=llm, hitl_low_confidence=False)
    r = svc.chat("How many PTO days do employees get?")
    assert r.status == "answered"
    assert "25 days" in r.answer
    assert any("extractive" in w for w in r.warnings)


def test_classifier_outage_fails_open_to_both_sources(make_service):
    svc = make_service(llm=ScriptedLLM({"classify": RuntimeError("timeout")}))
    r = svc.chat("How many PTO days do employees get?")
    assert r.route == "both" and r.status == "answered"
    assert any("classifier unavailable" in w.lower() for w in r.warnings)


def test_conversation_survives_service_restart(tmp_path, shared_kb):
    from research_agent.persistence.db import Database
    from research_agent.service import build_service

    s = make_settings(tmp_path)
    svc1 = build_service(s, llm=MockLLM(), web=MockWebSearch(FIXTURE_WEB), kb=shared_kb, db=Database(s.sqlite_path), auto_ingest=False)
    r = svc1.chat("What's the limit?")
    svc2 = build_service(s, llm=MockLLM(), web=MockWebSearch(FIXTURE_WEB), kb=shared_kb, db=Database(s.sqlite_path), auto_ingest=False)
    r2 = svc2.resume(r.thread_id, answer="API rate limit")
    assert r2.status == "answered" and "100 requests per second" in r2.answer


def test_long_history_is_summarised(make_service):
    svc = make_service(history_summarize_after=6, history_keep_last=2)
    r = svc.chat("What uptime does our SLA guarantee?")
    for q in ("What is the API rate limit?", "How long are webhooks retried?", "What is the chargeback fee?"):
        svc.chat(q, thread_id=r.thread_id)
    t = svc.get_thread(r.thread_id)
    assert t["summary"]
    assert len(t["messages"]) < 8


def test_kb_write_requires_approval_and_becomes_searchable(make_service, tmp_path):
    s = make_settings(tmp_path / "kbw")
    kb = KnowledgeBase(s, HashingEmbeddings())
    kb.ingest_directory(s.docs_dir)
    svc = make_service(kb=kb)
    from research_agent.schemas import DocumentIn

    out = svc.request_document_write(DocumentIn(doc_id="crypto-policy", title="Crypto Policy",
                                                content="LedgerLeaf does not offer cryptocurrency staking rewards to merchants."))
    assert out["status"] == "pending_approval"
    assert not any(d["doc_id"] == "crypto-policy" for d in kb.list_documents())
    svc.decide_approval(out["approval_id"], "approve", reviewer="admin")
    assert any(d["doc_id"] == "crypto-policy" for d in kb.list_documents())
    r = svc.chat("Does LedgerLeaf offer cryptocurrency staking rewards?")
    assert any(c.doc_id == "crypto-policy" for c in r.citations)
