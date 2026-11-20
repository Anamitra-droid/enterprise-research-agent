"""Run the demonstration scenarios through the real HTTP API (FastAPI TestClient, in-process)
and write a transcript to docs/DEMO.md.

    python scripts/demo.py            # uses whatever providers are configured (mock if no keys)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Fresh app/checkpoint databases for a reproducible transcript (the vector index is reused).
for name in ("demo_app.db", "demo_checkpoints.db"):
    for suffix in ("", "-wal", "-shm"):
        (ROOT / "var" / f"{name}{suffix}").unlink(missing_ok=True)
os.environ["SQLITE_PATH"] = str(ROOT / "var" / "demo_app.db")
os.environ["CHECKPOINT_PATH"] = str(ROOT / "var" / "demo_checkpoints.db")

from research_agent.api.app import create_app  # noqa: E402

OUT = ROOT / "docs" / "DEMO.md"
lines: list[str] = []


def md(text: str = "") -> None:
    lines.append(text)


def show_response(r: dict) -> None:
    md(f"**Status:** `{r['status']}` · **route:** `{r.get('route')}` · **confidence:** `{r.get('confidence')}` · "
       f"**groundedness:** `{r.get('groundedness')}` · **latency:** {r.get('latency_ms')} ms")
    md(f"\n**Agent trace:** `{' → '.join(r.get('trace', []))}`\n")
    if r["status"] == "needs_clarification":
        md(f"> 🤔 **Clarification requested:** {r['clarification_question']}\n")
    elif r["status"] == "pending_approval":
        a = r["approval"]
        md(f"> 🛑 **Approval required** (`{a['approval_type']}`, id `{a['approval_id']}`): {a['reason']}\n")
    else:
        md("**Answer:**\n")
        md("\n".join("> " + ln for ln in (r.get("answer") or "").splitlines()))
        md()
        if r.get("conflicts"):
            md("<details><summary>Structured conflicts (JSON)</summary>\n\n```json\n"
               + json.dumps(r["conflicts"], indent=2) + "\n```\n</details>\n")
    for w in r.get("warnings", []):
        md(f"- ⚠️ {w}")
    md()


def turn(client: TestClient, label: str, method: str, url: str, body: dict) -> dict:
    md(f"**{label}** `{method} {url}`\n```json\n{json.dumps(body, indent=2)}\n```")
    resp = client.request(method, url, json=body)
    resp.raise_for_status()
    data = resp.json()
    if "thread_id" in data and "status" in data:
        show_response(data)
    elif "chat" in data:  # approval decision that resumed a conversation
        md(f"Approval `{data['approval_id']}` → **{data['status']}**; the paused conversation resumes:\n")
        show_response(data["chat"])
    else:
        md(f"```json\n{json.dumps(data, indent=2)[:1500]}\n```\n")
    return data


def scenario(title: str, description: str) -> None:
    md(f"\n---\n\n## {title}\n\n_{description}_\n")
    print(f"• {title}")


def main() -> None:
    app = create_app()
    with TestClient(app) as client:
        health = client.get("/health").json()
        md("# Demonstration: Query Scenarios\n")
        md(f"Generated {time.strftime('%Y-%m-%d %H:%M')} · LLM: `{health['llm']}` · web search: `{health['web_search']}` · "
           f"embeddings: `{health['embeddings']}` · KB chunks: {health['kb_chunks']}\n")
        if health["llm"] == "mock":
            md("> ℹ️ Generated in **offline mock mode** (no API keys): routing/synthesis are keyword heuristics and the answers are "
               "extractive; web results come from a labelled fictional fixture corpus. With `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` "
               "set, the same script produces real LLM answers and live web results.\n")

        scenario("1. Internal knowledge-base question (RAG)", "Routed to internal retrieval only; every claim cites internal documents.")
        turn(client, "User", "POST", "/chat", {"message": "What is our parental leave policy and how many PTO days do employees get?"})

        scenario("2. External web question", "Needs current public information, so routed to web search only.")
        turn(client, "User", "POST", "/chat", {"message": "What are the latest trends in real-time payments adoption?"})

        scenario("3. Hybrid question with conflict detection",
                 "Needs internal competitor analysis AND current web data. Internal and web retrieval run in parallel; the conflict agent finds that the 2025 internal analysis is outdated.")
        turn(client, "User", "POST", "/chat", {"message": "How do PayNova's current fees compare to our pricing?"})

        scenario("4. Checking an internal assumption against current data",
                 "The treasury policy (Jan 2024) assumes a 5.25–5.50% Fed funds rate; the web says otherwise.")
        turn(client, "User", "POST", "/chat", {"message": "Is our treasury Fed funds rate assumption still valid given the current Fed rate?"})

        scenario("5. Ambiguous query → clarification (LangGraph interrupt)",
                 "The classifier asks a clarifying question; the graph pauses and resumes from its checkpoint on the user's reply.")
        r = turn(client, "User", "POST", "/chat", {"message": "What's the limit?"})
        if r["status"] == "needs_clarification":
            turn(client, "User clarifies", "POST", f"/chat/{r['thread_id']}/resume", {"answer": "The API rate limit"})

        scenario("6. Multi-turn conversation context",
                 "The follow-up uses a pronoun; the contextualizer rewrites it into a standalone question using the thread history.")
        r = turn(client, "User", "POST", "/chat", {"message": "What uptime does our SLA guarantee?"})
        turn(client, "User (follow-up, same thread)", "POST", "/chat",
             {"message": "And what service credits do customers get if we miss it?", "thread_id": r["thread_id"]})

        scenario("7. Human-in-the-loop: confidential documents",
                 "The answer relies on CONFIDENTIAL documents, so a reviewer must approve before it is released.")
        r = turn(client, "User", "POST", "/chat", {"message": "What is the status of Project Falcon?", "user_id": "analyst-42"})
        if r["status"] == "pending_approval":
            md("Reviewer checks the queue:\n")
            q = client.get("/approvals").json()
            md(f"```json\n{json.dumps([{k: a[k] for k in ('approval_id', 'approval_type', 'status', 'reason')} for a in q], indent=2)}\n```\n")
            turn(client, "Reviewer approves", "POST", f"/approvals/{r['approval']['approval_id']}",
                 {"decision": "approve", "reviewer": "cfo@ledgerleaf", "comment": "OK for deal-team analyst"})

        scenario("8. Human-in-the-loop: privacy guard for external search",
                 "The web query would leak a confidential codename to an external search engine. The reviewer rejects it, so the system answers from internal sources only, which then triggers the confidential-data gate.")
        r = turn(client, "User", "POST", "/chat", {"message": "Search the web for the latest news about Project Falcon"})
        if r["status"] == "pending_approval":
            r = turn(client, "Reviewer rejects external search", "POST", f"/chat/{r['thread_id']}/resume",
                     {"decision": "reject", "reviewer": "legal@ledgerleaf", "comment": "Codename must not leave the company"})
            if r["status"] == "pending_approval":
                turn(client, "Reviewer rejects confidential answer", "POST", f"/chat/{r['thread_id']}/resume",
                     {"decision": "reject", "reviewer": "legal@ledgerleaf", "comment": "requester not on deal team"})

        scenario("9. Unanswerable question (hallucination guard)",
                 "Nothing in the sources answers the question, so the system abstains instead of guessing (an empty retrieval would also trigger one reformulated retry first).")
        turn(client, "User", "POST", "/chat", {"message": "What is our policy on cryptocurrency staking rewards?"})

        scenario("10. Knowledge-base write with approval",
                 "Adding a document is a sensitive operation: it is queued until approved, then indexed and immediately searchable.")
        r = turn(client, "Request", "POST", "/documents", {
            "doc_id": "crypto-policy", "title": "Cryptocurrency Policy", "classification": "internal",
            "content": "# Cryptocurrency Policy\n\nLedgerLeaf does not offer cryptocurrency staking rewards. Merchants may not accept crypto payments through LedgerLeaf Pay. This policy is reviewed annually by the compliance team.",
            "requested_by": "compliance-lead"})
        turn(client, "Approve", "POST", f"/approvals/{r['approval_id']}", {"decision": "approve", "reviewer": "kb-admin"})
        turn(client, "User asks again", "POST", "/chat", {"message": "What is our policy on cryptocurrency staking rewards?"})
        client.delete("/documents/crypto-policy")  # queue clean-up request
        for a in client.get("/approvals").json():
            if a["approval_type"] == "kb_delete":
                client.post(f"/approvals/{a['approval_id']}", json={"decision": "approve", "reviewer": "kb-admin"})

        scenario("Observability", "Aggregated metrics from `GET /metrics` after the demo run.")
        m = client.get("/metrics").json()
        md("| node | runs | avg ms | max ms | errors |\n|---|---|---|---|---|")
        for n in m["nodes"]:
            md(f"| {n['node']} | {n['runs']} | {n['avg_ms']} | {n['max_ms']} | {n['errors']} |")
        md(f"\nLLM usage: `{json.dumps(m['llm'])}`\n\nApprovals: `{json.dumps(m['approvals'])}`\n")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDemo transcript written to {OUT}")


if __name__ == "__main__":
    main()
