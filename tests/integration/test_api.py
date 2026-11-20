"""API-level integration tests using FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from research_agent.api.app import create_app


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm"] == "mock" and h["kb_chunks"] > 0


def test_chat_roundtrip_and_request_id(client):
    r = client.post("/chat", json={"message": "What is the API rate limit for live keys?"}, headers={"x-request-id": "abc"})
    assert r.status_code == 200 and r.headers["x-request-id"] == "abc"
    body = r.json()
    assert body["status"] == "answered" and body["citations"] and body["thread_id"].startswith("thr_")


def test_validation_error(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_clarification_flow_and_409(client):
    r = client.post("/chat", json={"message": "What's the limit?"}).json()
    tid = r["thread_id"]
    assert r["status"] == "needs_clarification"
    assert client.post("/chat", json={"message": "hi", "thread_id": tid}).status_code == 409
    assert client.post(f"/chat/{tid}/resume", json={"decision": "approve"}).status_code == 409  # wrong kind
    r2 = client.post(f"/chat/{tid}/resume", json={"answer": "API rate limit"}).json()
    assert r2["status"] == "answered"
    thread = client.get(f"/threads/{tid}").json()
    assert thread["pending"] is None and len(thread["messages"]) >= 3


def test_approval_endpoints(client):
    r = client.post("/chat", json={"message": "What is the status of Project Falcon?"}).json()
    apr_id = r["approval"]["approval_id"]
    pending = client.get("/approvals").json()
    assert [a["approval_id"] for a in pending] == [apr_id]
    out = client.post(f"/approvals/{apr_id}", json={"decision": "reject", "reviewer": "legal"}).json()
    assert out["status"] == "rejected" and out["chat"]["status"] == "rejected"
    assert client.post(f"/approvals/{apr_id}", json={"decision": "approve"}).status_code == 409
    assert client.get("/approvals/does-not-exist").status_code == 404


def test_document_endpoints_require_approval(client):
    docs = client.get("/documents").json()
    assert any(d["doc_id"] == "sla" for d in docs)
    r = client.delete("/documents/sla", params={"requested_by": "intern"})
    assert r.status_code == 202 and r.json()["status"] == "pending_approval"
    assert client.delete("/documents/nope").status_code == 404
    # reject so the shared index is untouched
    client.post(f"/approvals/{r.json()['approval_id']}", json={"decision": "reject", "reviewer": "admin"})
    assert any(d["doc_id"] == "sla" for d in client.get("/documents").json())


def test_metrics_and_audit(client):
    client.post("/chat", json={"message": "How many PTO days do employees get?"})
    m = client.get("/metrics").json()
    assert m["turns"] >= 1 and any(n["node"] == "synthesize" for n in m["nodes"])
    events = [e["event"] for e in client.get("/audit").json()]
    assert "query_received" in events and "answer_returned" in events


def test_openapi_documents_all_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/chat", "/chat/{thread_id}/resume", "/approvals", "/approvals/{approval_id}", "/documents", "/metrics", "/health"):
        assert p in paths
