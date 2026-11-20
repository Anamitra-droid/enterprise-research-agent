"""Application service: runs the graph per thread, maps interrupts to API states,
manages approvals (graph interrupts + knowledge-base writes) and audit logging."""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from research_agent.agents.deps import AgentDeps
from research_agent.config import Settings, get_settings
from research_agent.graph.builder import build_graph
from research_agent.graph.state import fresh_turn_input
from research_agent.llm import LLMClient, build_llm
from research_agent.observability.logging import bind_context, configure_logging, get_logger
from research_agent.observability.tracing import set_metrics_sink
from research_agent.persistence.db import Database
from research_agent.rag.embeddings import build_embeddings
from research_agent.rag.loader import SourceDoc
from research_agent.rag.vectorstore import KnowledgeBase
from research_agent.schemas import (
    ApprovalType,
    ChatResponse,
    Citation,
    DocumentIn,
    PendingApproval,
)
from research_agent.tools.web_search import WebSearch, build_web_search

log = get_logger("research_agent.service")

# Explicit allow-list of our Pydantic types that may be deserialized from checkpoints.
CHECKPOINT_TYPES = [
    ("research_agent.schemas", name)
    for name in ("Route", "QueryClassification", "Evidence", "Conflict", "SynthesizedAnswer", "Claim",
                 "GroundednessReport", "ClaimVerdict")
]


class NotFoundError(Exception): ...


class InvalidStateError(Exception): ...


@dataclass
class ResearchService:
    settings: Settings
    deps: AgentDeps
    db: Database
    graph: Any

    # ------------------------------------------------------------------ chat
    def chat(self, message: str, thread_id: str | None = None, user_id: str = "anonymous") -> ChatResponse:
        thread_id = thread_id or f"thr_{uuid.uuid4().hex[:12]}"
        config = self._config(thread_id)
        if self._pending_interrupt(config):
            raise InvalidStateError(
                f"Thread {thread_id} is waiting for a clarification or approval. Use POST /chat/{thread_id}/resume first."
            )
        turn_id = f"turn_{uuid.uuid4().hex[:10]}"
        bind_context(thread_id=thread_id, turn_id=turn_id, user_id=user_id)
        self.db.upsert_thread(thread_id, user_id, title=message)
        self.db.audit("query_received", thread_id, user_id, message=message, turn_id=turn_id)
        start = time.perf_counter()
        self.graph.invoke(fresh_turn_input(HumanMessage(content=message), message, user_id, turn_id), config)
        return self._build_response(thread_id, config, start)

    def resume(self, thread_id: str, *, answer: str | None = None, decision: str | None = None,
               reviewer: str = "anonymous", comment: str = "") -> ChatResponse:
        config = self._config(thread_id)
        intr = self._pending_interrupt(config)
        if not intr:
            raise InvalidStateError(f"Thread {thread_id} has nothing to resume.")
        bind_context(thread_id=thread_id)
        if intr["kind"] == "clarification":
            if not answer:
                raise InvalidStateError("This thread is waiting for a clarification: provide `answer`.")
            payload: dict[str, Any] = {"answer": answer}
            self.db.audit("clarification_answered", thread_id, None, answer=answer)
        else:
            if decision not in ("approve", "reject"):
                raise InvalidStateError("This thread is waiting for an approval: provide `decision` = approve|reject.")
            payload = {"decision": decision, "reviewer": reviewer, "comment": comment}
            apr = self.db.pending_approval_for_thread(thread_id)
            if apr:
                self.db.decide_approval(apr["approval_id"], decision == "approve", reviewer, comment)
            self.db.audit("approval_decided", thread_id, None, approval_type=intr.get("approval_type"),
                          decision=decision, reviewer=reviewer, comment=comment)
        start = time.perf_counter()
        self.graph.invoke(Command(resume=payload), config)
        return self._build_response(thread_id, config, start)

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        meta = self.db.get_thread(thread_id)
        if not meta:
            raise NotFoundError(thread_id)
        snap = self.graph.get_state(self._config(thread_id))
        values = snap.values or {}
        msgs = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in values.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))
        ]
        intr = self._pending_interrupt(self._config(thread_id))
        return {**meta, "summary": values.get("summary", ""), "messages": msgs, "pending": intr}

    # ------------------------------------------------------------------ approvals
    def decide_approval(self, approval_id: str, decision: str, reviewer: str, comment: str = "") -> dict[str, Any]:
        apr = self.db.get_approval(approval_id)
        if not apr:
            raise NotFoundError(approval_id)
        if apr["status"] != "pending":
            raise InvalidStateError(f"Approval {approval_id} already {apr['status']}.")
        if apr["thread_id"]:  # graph interrupt -> resume the conversation
            resp = self.resume(apr["thread_id"], decision=decision, reviewer=reviewer, comment=comment)
            return {"approval_id": approval_id, "status": "approved" if decision == "approve" else "rejected", "chat": resp.model_dump()}
        # knowledge-base write/delete
        approved = decision == "approve"
        result: dict[str, Any] = {}
        if approved:
            p = apr["payload"]
            if apr["approval_type"] == ApprovalType.KB_WRITE.value:
                chunks = self.deps.kb.add_document(SourceDoc(
                    doc_id=p["doc_id"], title=p["title"], content=p["content"], classification=p["classification"],
                    department=p.get("department", "general"), last_updated=p.get("last_updated")))
                result = {"doc_id": p["doc_id"], "chunks": chunks}
            elif apr["approval_type"] == ApprovalType.KB_DELETE.value:
                result = {"doc_id": p["doc_id"], "chunks_deleted": self.deps.kb.delete_document(p["doc_id"])}
        self.db.decide_approval(approval_id, approved, reviewer, comment)
        self.db.audit("approval_decided", None, reviewer, approval_id=approval_id, approval_type=apr["approval_type"],
                      decision=decision, result=result)
        return {"approval_id": approval_id, "status": "approved" if approved else "rejected", "result": result}

    # ------------------------------------------------------------------ documents
    def request_document_write(self, doc: DocumentIn) -> dict[str, Any]:
        if not self.settings.hitl_kb_writes:
            chunks = self.deps.kb.add_document(SourceDoc(**doc.model_dump(exclude={"requested_by"})))
            self.db.audit("document_ingested", None, doc.requested_by, doc_id=doc.doc_id, chunks=chunks)
            return {"status": "ingested", "doc_id": doc.doc_id, "chunks": chunks}
        apr = self.db.create_approval(ApprovalType.KB_WRITE.value, doc.model_dump(),
                                      reason=f"Add/replace document '{doc.doc_id}' ({doc.classification}) in the knowledge base",
                                      requested_by=doc.requested_by)
        self.db.audit("document_write_requested", None, doc.requested_by, doc_id=doc.doc_id, approval_id=apr)
        return {"status": "pending_approval", "approval_id": apr, "doc_id": doc.doc_id}

    def request_document_delete(self, doc_id: str, requested_by: str) -> dict[str, Any]:
        if not any(d["doc_id"] == doc_id for d in self.deps.kb.list_documents()):
            raise NotFoundError(doc_id)
        if not self.settings.hitl_kb_writes:
            n = self.deps.kb.delete_document(doc_id)
            return {"status": "deleted", "doc_id": doc_id, "chunks_deleted": n}
        apr = self.db.create_approval(ApprovalType.KB_DELETE.value, {"doc_id": doc_id},
                                      reason=f"Delete document '{doc_id}' from the knowledge base", requested_by=requested_by)
        self.db.audit("document_delete_requested", None, requested_by, doc_id=doc_id, approval_id=apr)
        return {"status": "pending_approval", "approval_id": apr, "doc_id": doc_id}

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    def _pending_interrupt(self, config: dict) -> dict[str, Any] | None:
        snap = self.graph.get_state(config)
        for task in snap.tasks or ():
            for intr in getattr(task, "interrupts", ()) or ():
                value = intr.value if isinstance(intr.value, dict) else {"kind": "clarification", "question": str(intr.value)}
                return {**value, "interrupt_id": getattr(intr, "id", None)}
        return None

    def _build_response(self, thread_id: str, config: dict, start: float) -> ChatResponse:
        snap = self.graph.get_state(config)
        v = snap.values or {}
        latency = int((time.perf_counter() - start) * 1000)
        evidence = v.get("evidence", []) or []
        draft = v.get("draft")
        g = v.get("groundedness")
        base = dict(
            thread_id=thread_id, route=v.get("route"), warnings=list(dict.fromkeys(v.get("warnings", []))),
            trace=v.get("trace", []), latency_ms=latency,
        )
        intr = self._pending_interrupt(config)
        if intr and intr.get("kind") == "clarification":
            self.db.audit("clarification_requested", thread_id, v.get("user_id"), question=intr["question"])
            return ChatResponse(status="needs_clarification", clarification_question=intr["question"], **base)
        if intr:
            apr_id = f"apr_{(intr.get('interrupt_id') or uuid.uuid4().hex)[:16]}"
            existing = self.db.get_approval(apr_id)
            if not existing:
                self.db.create_approval(intr["approval_type"], intr.get("details", {}), intr.get("reason", ""),
                                        thread_id=thread_id, requested_by=v.get("user_id", "anonymous"), approval_id=apr_id)
                self.db.audit("approval_requested", thread_id, v.get("user_id"), approval_id=apr_id,
                              approval_type=intr["approval_type"], reason=intr.get("reason"))
            return ChatResponse(
                status="pending_approval",
                approval=PendingApproval(approval_id=apr_id, approval_type=intr["approval_type"],
                                         reason=intr.get("reason", ""), details=intr.get("details", {})),
                **base,
            )
        status = v.get("status", "answered")
        cited = set()
        if draft is not None:
            from research_agent.agents.synthesizer import cited_ids

            cited = cited_ids(draft.answer) | {s for c in draft.claims for s in c.source_ids}
        citations = [
            Citation(id=e.source_id, kind=e.kind, title=e.title, url=e.url, doc_id=e.doc_id,
                     classification=e.classification, last_updated=e.last_updated, snippet=e.content[:280])
            for e in evidence if e.source_id in cited
        ] if status != "rejected" else []
        resp = ChatResponse(
            status="rejected" if status == "rejected" else "answered",
            answer=v.get("final_answer", ""),
            citations=citations,
            conflicts=v.get("conflicts", []) if status != "rejected" else [],
            confidence=draft.confidence if draft is not None else None,
            groundedness=g.score if g is not None else None,
            **base,
        )
        self.db.audit("answer_returned", thread_id, v.get("user_id"), status=resp.status, route=str(resp.route),
                      n_citations=len(resp.citations), n_conflicts=len(resp.conflicts), latency_ms=latency,
                      confidence=resp.confidence, groundedness=resp.groundedness)
        return resp


def build_service(settings: Settings | None = None, *, llm: LLMClient | None = None, web: WebSearch | None = None,
                  kb: KnowledgeBase | None = None, db: Database | None = None, auto_ingest: bool = True) -> ResearchService:
    """Composition root. Every dependency can be overridden (tests inject fakes)."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    configure_logging(settings.log_level, settings.log_json)
    db = db or Database(settings.sqlite_path)
    set_metrics_sink(db)
    kb = kb or KnowledgeBase(settings, build_embeddings(settings))
    if auto_ingest:
        if kb.count() == 0:
            log.info("kb_empty_ingesting", dir=str(settings.docs_dir))
            kb.ingest_directory(settings.docs_dir)
        else:
            kb.refresh_sensitive_terms(settings.docs_dir)
    deps = AgentDeps(settings=settings, llm=llm or build_llm(settings), kb=kb, web=web or build_web_search(settings))
    conn = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
    graph = build_graph(deps, checkpointer)
    log.info("service_ready", llm=deps.llm.name, web=deps.web.name, kb_chunks=kb.count(),
             embeddings=settings.embedding_provider)
    return ResearchService(settings=settings, deps=deps, db=db, graph=graph)
