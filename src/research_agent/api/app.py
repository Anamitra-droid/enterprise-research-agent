"""FastAPI application. Interactive docs at /docs (Swagger) and /redoc."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from research_agent import __version__
from research_agent.observability.logging import bind_context, clear_context, get_logger
from research_agent.schemas import (
    ApprovalDecision,
    ChatRequest,
    ChatResponse,
    DocumentIn,
    ResumeRequest,
)
from research_agent.service import InvalidStateError, NotFoundError, ResearchService, build_service

log = get_logger("research_agent.api")

TAGS = [
    {"name": "chat", "description": "Ask questions, answer clarifications, continue conversations."},
    {"name": "approvals", "description": "Human-in-the-loop review queue for sensitive operations."},
    {"name": "documents", "description": "Knowledge-base management (writes require approval)."},
    {"name": "ops", "description": "Health, metrics and audit log."},
]


def create_app(service: ResearchService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or build_service()
        yield

    app = FastAPI(
        title="Enterprise AI Research Agent",
        version=__version__,
        description=(
            "Multi-agent research assistant (LangGraph) that answers questions from internal LedgerLeaf documents (RAG) "
            "and the web, with citations, conflict detection, clarification and human-in-the-loop approvals.\n\n"
            "**Flow:** `POST /chat` returns `answered`, `needs_clarification` or `pending_approval`. For the last two, call "
            "`POST /chat/{thread_id}/resume` (or `POST /approvals/{id}`) to continue. Reuse `thread_id` for follow-up questions."
        ),
        openapi_tags=TAGS,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        clear_context()
        bind_context(request_id=rid)
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info("http_request", method=request.method, path=request.url.path, status=response.status_code,
                 ms=round((time.perf_counter() - start) * 1000, 1))
        return response

    @app.exception_handler(NotFoundError)
    async def _nf(_: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": f"Not found: {exc}"})

    @app.exception_handler(InvalidStateError)
    async def _conflict(_: Request, exc: InvalidStateError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    def svc(request: Request) -> ResearchService:
        return request.app.state.service

    # ------------------------------------------------------------------ chat
    @app.post("/chat", response_model=ChatResponse, tags=["chat"], summary="Ask a question (new or existing thread)")
    def chat(req: ChatRequest, s: ResearchService = Depends(svc)) -> ChatResponse:
        return s.chat(req.message, thread_id=req.thread_id, user_id=req.user_id)

    @app.post("/chat/{thread_id}/resume", response_model=ChatResponse, tags=["chat"],
              summary="Answer a clarification or approve/reject a pending operation")
    def resume(thread_id: str, req: ResumeRequest, s: ResearchService = Depends(svc)) -> ChatResponse:
        return s.resume(thread_id, answer=req.answer, decision=req.decision, reviewer=req.reviewer, comment=req.comment)

    @app.get("/threads/{thread_id}", tags=["chat"], summary="Conversation history and pending state")
    def get_thread(thread_id: str, s: ResearchService = Depends(svc)) -> dict:
        return s.get_thread(thread_id)

    @app.get("/threads", tags=["chat"], summary="List conversations")
    def list_threads(user_id: str | None = None, s: ResearchService = Depends(svc)) -> list[dict]:
        return s.db.list_threads(user_id)

    # ------------------------------------------------------------------ approvals
    @app.get("/approvals", tags=["approvals"], summary="List approval requests")
    def list_approvals(status: Literal["pending", "approved", "rejected"] | None = Query(default="pending"),
                       s: ResearchService = Depends(svc)) -> list[dict]:
        return s.db.list_approvals(status)

    @app.get("/approvals/{approval_id}", tags=["approvals"])
    def get_approval(approval_id: str, s: ResearchService = Depends(svc)) -> dict:
        apr = s.db.get_approval(approval_id)
        if not apr:
            raise HTTPException(404, "approval not found")
        return apr

    @app.post("/approvals/{approval_id}", tags=["approvals"], summary="Approve or reject a pending request")
    def decide(approval_id: str, body: ApprovalDecision, s: ResearchService = Depends(svc)) -> dict:
        return s.decide_approval(approval_id, body.decision, body.reviewer, body.comment)

    # ------------------------------------------------------------------ documents
    @app.get("/documents", tags=["documents"], summary="List indexed documents")
    def list_documents(s: ResearchService = Depends(svc)) -> list[dict]:
        return s.deps.kb.list_documents()

    @app.post("/documents", status_code=202, tags=["documents"], summary="Add or replace a document (requires approval)")
    def add_document(doc: DocumentIn, s: ResearchService = Depends(svc)) -> dict:
        return s.request_document_write(doc)

    @app.delete("/documents/{doc_id}", status_code=202, tags=["documents"], summary="Delete a document (requires approval)")
    def delete_document(doc_id: str, requested_by: str = "anonymous", s: ResearchService = Depends(svc)) -> dict:
        return s.request_document_delete(doc_id, requested_by)

    # ------------------------------------------------------------------ ops
    class Health(BaseModel):
        status: str
        version: str
        llm: str
        web_search: str
        embeddings: str
        kb_chunks: int

    @app.get("/health", response_model=Health, tags=["ops"])
    def health(s: ResearchService = Depends(svc)) -> Health:
        return Health(status="ok", version=__version__, llm=s.deps.llm.name, web_search=s.deps.web.name,
                      embeddings=s.settings.embedding_provider, kb_chunks=s.deps.kb.count())

    @app.get("/metrics", tags=["ops"], summary="Aggregated node latency, LLM token usage, tool retries and approvals")
    def metrics(s: ResearchService = Depends(svc)) -> dict:
        return s.db.metrics_summary()

    @app.get("/audit", tags=["ops"], summary="Audit log (most recent first)")
    def audit(thread_id: str | None = None, limit: int = 100, s: ResearchService = Depends(svc)) -> list[dict]:
        return s.db.audit_events(thread_id, limit)

    return app


app = create_app()
