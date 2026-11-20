"""Command-line interface.

  research-agent ingest [--reset]        # (re)build the vector index from data/docs
  research-agent serve [--port 8000]     # run the FastAPI server
  research-agent chat [--url URL]        # interactive chat (in-process, or against a running API)
  research-agent eval [--skip-answers]   # run the evaluation suite
"""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(add_completion=False, help="Enterprise AI Research Agent")
console = Console()


@app.command()
def ingest(reset: bool = typer.Option(False, help="Delete the existing index first")) -> None:
    """Index the markdown documents in data/docs into ChromaDB."""
    from research_agent.config import get_settings
    from research_agent.rag.embeddings import build_embeddings
    from research_agent.rag.vectorstore import KnowledgeBase

    s = get_settings()
    kb = KnowledgeBase(s, build_embeddings(s))
    stats = kb.ingest_directory(s.docs_dir, reset=reset)
    for doc_id, n in stats.items():
        console.print(f"  {doc_id:<28} {n} chunks")
    console.print(f"[green]Indexed {len(stats)} documents, {kb.count()} chunks total.[/green]")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the API server (Swagger UI at /docs)."""
    import uvicorn

    uvicorn.run("research_agent.api.app:app", host=host, port=port, reload=reload)


@app.command("eval")
def run_eval(skip_answers: bool = False, limit: int | None = None) -> None:
    """Run retrieval / routing / answer-quality evaluation."""
    from research_agent.evaluation import main

    args = ["--skip-answers"] if skip_answers else []
    if limit:
        args += ["--limit", str(limit)]
    main(args)


class _Client:
    """Same interface for in-process service and remote HTTP API."""

    def __init__(self, url: str | None):
        self.url = url
        if url:
            import httpx

            self.http = httpx.Client(base_url=url, timeout=180)
        else:
            from research_agent.service import build_service

            self.svc = build_service()

    def chat(self, message: str, thread_id: str | None, user_id: str) -> dict[str, Any]:
        if self.url:
            r = self.http.post("/chat", json={"message": message, "thread_id": thread_id, "user_id": user_id})
            r.raise_for_status()
            return r.json()
        return self.svc.chat(message, thread_id, user_id).model_dump(mode="json")

    def resume(self, thread_id: str, **kw) -> dict[str, Any]:
        if self.url:
            r = self.http.post(f"/chat/{thread_id}/resume", json=kw)
            r.raise_for_status()
            return r.json()
        return self.svc.resume(thread_id, **kw).model_dump(mode="json")


def _render(resp: dict[str, Any]) -> None:
    meta = f"route={resp.get('route')}  confidence={resp.get('confidence')}  groundedness={resp.get('groundedness')}  {resp.get('latency_ms')}ms"
    if resp["status"] in ("answered", "rejected"):
        console.print(Panel(Markdown(resp.get("answer") or ""), title=f"[bold]{resp['status']}[/bold]", subtitle=meta))
    for w in resp.get("warnings", []):
        console.print(f"[yellow]⚠ {w}[/yellow]")
    console.print(f"[dim]trace: {' → '.join(resp.get('trace', []))}[/dim]")


@app.command()
def chat(url: str | None = typer.Option(None, help="API base URL, e.g. http://127.0.0.1:8000 (default: in-process)"),
         user: str = "cli-user") -> None:
    """Interactive multi-turn chat. Type /new for a new thread, /quit to exit."""
    client = _Client(url)
    thread_id: str | None = None
    console.print("[bold green]LedgerLeaf Research Agent[/bold green]: ask a question (/new, /quit)")
    while True:
        msg = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/new":
            thread_id = None
            console.print("[dim]started a new conversation[/dim]")
            continue
        resp = client.chat(msg, thread_id, user)
        thread_id = resp["thread_id"]
        while resp["status"] in ("needs_clarification", "pending_approval"):
            if resp["status"] == "needs_clarification":
                ans = Prompt.ask(f"[magenta]clarify[/magenta] {resp['clarification_question']}")
                resp = client.resume(thread_id, answer=ans)
            else:
                apr = resp["approval"]
                console.print(Panel(f"{apr['reason']}\n\n{apr['details'].get('draft_answer', apr['details'].get('web_query', ''))}",
                                    title=f"[red]Approval required: {apr['approval_type']}[/red]"))
                d = Prompt.ask("approve or reject?", choices=["approve", "reject"], default="approve")
                resp = client.resume(thread_id, decision=d, reviewer=user)
        _render(resp)


if __name__ == "__main__":
    app()
