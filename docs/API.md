# API Reference

Interactive docs: **http://127.0.0.1:8000/docs** (Swagger) and **/redoc**. The OpenAPI schema is at `/openapi.json`.

All responses are JSON. Every response carries an `x-request-id` header (pass your own to correlate logs).

## Conversation lifecycle

```
POST /chat ──► status = answered            (done; reuse thread_id for follow-ups)
           ├─► status = needs_clarification ──► POST /chat/{thread_id}/resume {"answer": "..."}
           └─► status = pending_approval    ──► POST /chat/{thread_id}/resume {"decision": "approve"|"reject"}
                                                or POST /approvals/{approval_id}
```

A thread waiting for clarification or approval rejects new messages with **409** until resumed.

---

### `POST /chat`

Ask a question. Omit `thread_id` to start a new conversation.

```json
{ "message": "How do PayNova's current fees compare to our pricing?", "thread_id": null, "user_id": "analyst-42" }
```

**Response: `ChatResponse`**

| field | type | description |
|---|---|---|
| `thread_id` | string | conversation id (reuse for follow-ups) |
| `status` | `answered` \| `needs_clarification` \| `pending_approval` \| `rejected` \| `error` | |
| `answer` | markdown | answer with `[S1]` citation markers, conflicts section and source list |
| `citations` | `Citation[]` | `{id, kind: internal\|web, title, url, doc_id, classification, last_updated, snippet}` |
| `conflicts` | `Conflict[]` | `{topic, internal_claim, internal_source_ids, external_claim, external_source_ids, likely_resolution, severity}` |
| `route` | `internal` \| `web` \| `both` \| `clarify` \| `direct` | routing decision |
| `confidence` | 0–1 | synthesizer's self-assessed confidence |
| `groundedness` | 0–1 | fraction of claims verified against cited sources |
| `clarification_question` | string | set when `needs_clarification` |
| `approval` | `PendingApproval` | `{approval_id, approval_type, reason, details}` when `pending_approval` |
| `warnings` | string[] | degradations (e.g. web search unavailable, reformulated query) |
| `trace` | string[] | agent nodes executed this turn |
| `latency_ms` | int | |

Example (abridged):

```json
{
  "thread_id": "thr_3f2a9c1b7d4e",
  "status": "answered",
  "route": "both",
  "answer": "**From internal LedgerLeaf documents:**\n- PayNova charges 2.4% + $0.20 ... [S1]\n...\n### ⚠️ Conflicting information ...",
  "citations": [{"id": "S1", "kind": "internal", "title": "Competitor Analysis 2025 — PayNova", "doc_id": "competitor-analysis-2025", "classification": "internal", "last_updated": "2025-02-14", "snippet": "..."}],
  "conflicts": [{"topic": "PayNova", "internal_claim": "PayNova charges 2.4% + $0.20 ...", "internal_source_ids": ["S1"], "external_claim": "PayNova now charges 1.9% + $0.20 ...", "external_source_ids": ["S5"], "likely_resolution": "The external source is more recent (2026-09-02) ...", "severity": "medium"}],
  "confidence": 0.9, "groundedness": 1.0, "warnings": [],
  "trace": ["contextualize", "classify", "privacy_guard", "retrieve_internal", "search_web", "validate", "detect_conflicts", "synthesize", "check_groundedness", "approval_gate", "finalize"],
  "latency_ms": 5321
}
```

### `POST /chat/{thread_id}/resume`

Continue a paused thread.

```json
{ "answer": "The API rate limit" }
```
or
```json
{ "decision": "approve", "reviewer": "cfo@ledgerleaf", "comment": "ok" }
```

Returns a `ChatResponse`. **409** if the thread has nothing to resume or the wrong field is provided.

### `GET /threads/{thread_id}` and `GET /threads?user_id=`

Conversation history (`messages`), running `summary`, and `pending` (the open clarification/approval, if any).

---

### `GET /approvals?status=pending|approved|rejected`

Review queue. Each item: `approval_id, approval_type, status, thread_id, reason, payload, requested_by, reviewer, comment, created_at, decided_at`.

Approval types: `confidential_data`, `low_confidence`, `external_search_privacy`, `kb_write`, `kb_delete`.

### `GET /approvals/{approval_id}`

### `POST /approvals/{approval_id}`

```json
{ "decision": "approve", "reviewer": "kb-admin", "comment": "" }
```

For conversation approvals the paused thread resumes and the response contains `chat` (a `ChatResponse`). For knowledge-base approvals the write/delete is executed and `result` is returned. **409** if already decided, **404** if unknown.

---

### `GET /documents`

Indexed documents: `doc_id, title, classification, department, last_updated, chunks`.

### `POST /documents` (202, requires approval)

```json
{
  "doc_id": "crypto-policy",
  "title": "Cryptocurrency Policy",
  "content": "# Cryptocurrency Policy\n\nLedgerLeaf does not offer ...",
  "classification": "internal",
  "department": "compliance",
  "last_updated": "2026-11-01",
  "requested_by": "compliance-lead"
}
```
returns `{"status": "pending_approval", "approval_id": "...", "doc_id": "crypto-policy"}`. Same `doc_id` replaces the existing document.

### `DELETE /documents/{doc_id}?requested_by=` (202, requires approval)

---

### `GET /health`

`{status, version, llm: openrouter|mock, web_search: tavily|mock, embeddings, kb_chunks}`

### `GET /metrics`

```json
{
  "turns": 12,
  "nodes": [{"node": "synthesize", "runs": 11, "avg_ms": 2310.4, "max_ms": 5120.0, "errors": 0}],
  "llm": [{"purpose": "synthesize", "model": "anthropic/claude-sonnet-5.5", "calls": 11, "input_tokens": 42000, "output_tokens": 6100, "avg_ms": 2290.1, "failures": 0}],
  "tools": [{"tool": "web_search", "status": "retry", "n": 1}],
  "approvals": [{"approval_type": "confidential_data", "status": "approved", "n": 1}]
}
```

### `GET /audit?thread_id=&limit=100`

Audit events: `query_received`, `clarification_requested`, `clarification_answered`, `approval_requested`, `approval_decided`, `answer_returned`, `document_write_requested`, `document_delete_requested`, `document_ingested`.

## Errors

| code | when |
|---|---|
| 404 | unknown thread / approval / document |
| 409 | thread paused (or nothing to resume), approval already decided, wrong resume field |
| 422 | request validation failed (e.g. empty message, invalid `doc_id`) |

## curl quickstart

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"What is our parental leave policy?"}' | jq .answer

T=$(curl -s localhost:8000/chat -H 'content-type: application/json' -d '{"message":"What'"'"'s the limit?"}' | jq -r .thread_id)
curl -s localhost:8000/chat/$T/resume -H 'content-type: application/json' -d '{"answer":"API rate limit"}' | jq .answer

curl -s 'localhost:8000/approvals?status=pending' | jq
```
