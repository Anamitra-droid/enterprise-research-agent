"""Evaluation harness: retrieval accuracy, routing accuracy and answer quality (LLM-as-judge).

Run: `python -m research_agent.evaluation [--skip-answers] [--limit N]`
Writes `reports/eval_report.json` and `reports/eval_report.md`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from research_agent.agents import prompts
from research_agent.config import PROJECT_ROOT
from research_agent.graph.state import ResearchState
from research_agent.schemas import Evidence, JudgeScores, QueryClassification
from research_agent.service import ResearchService, build_service

GOLDEN = PROJECT_ROOT / "eval" / "golden_set.json"
REPORTS = PROJECT_ROOT / "reports"


def load_golden(path: Path = GOLDEN) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- retrieval


def retrieval_metrics(svc: ResearchService, items: list[dict], k: int = 5) -> dict[str, Any]:
    """hit@k, recall@k and MRR at the document level, for items with labelled relevant docs."""
    rows = []
    for it in items:
        rel = set(it.get("relevant_docs") or [])
        if not rel:
            continue
        query = it["question"] + (f" {it['clarification_answer']}" if it.get("clarification_answer") else "")
        ranked: list[str] = []
        for ev in svc.deps.kb.search(query, k=k * 2):
            if ev.doc_id not in ranked:
                ranked.append(ev.doc_id)
        ranked = ranked[:k]
        hits = [d for d in ranked if d in rel]
        rr = next((1 / (i + 1) for i, d in enumerate(ranked) if d in rel), 0.0)
        rows.append({"id": it["id"], "hit": bool(hits), "recall": len(set(hits)) / len(rel), "rr": rr, "retrieved": ranked})
    n = len(rows) or 1
    return {
        "k": k,
        "n": len(rows),
        f"hit@{k}": round(sum(r["hit"] for r in rows) / n, 3),
        f"recall@{k}": round(sum(r["recall"] for r in rows) / n, 3),
        "mrr": round(sum(r["rr"] for r in rows) / n, 3),
        "rows": rows,
    }


# ----------------------------------------------------------------------------- routing


def routing_accuracy(svc: ResearchService, items: list[dict]) -> dict[str, Any]:
    from research_agent.agents.router import make_classify

    classify = make_classify(svc.deps)
    rows = []
    for it in items:
        state = ResearchState(query=it["question"], standalone_query=it["question"])
        out = classify(state)
        c: QueryClassification = out["classification"]
        rows.append({"id": it["id"], "expected": it["expected_route"], "predicted": c.route.value,
                     "correct": c.route.value == it["expected_route"]})
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        confusion.setdefault(r["expected"], {}).setdefault(r["predicted"], 0)
        confusion[r["expected"]][r["predicted"]] += 1
    return {"n": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / (len(rows) or 1), 3),
            "confusion": confusion, "rows": rows}


# ----------------------------------------------------------------------------- answers


def run_question(svc: ResearchService, it: dict) -> tuple[Any, list[str]]:
    """Run one question end-to-end, auto-answering clarifications and auto-approving HITL gates."""
    events: list[str] = []
    resp = svc.chat(it["question"], user_id="evaluator")
    for _ in range(4):
        if resp.status == "needs_clarification":
            events.append("clarification")
            resp = svc.resume(resp.thread_id, answer=it.get("clarification_answer") or "the most common interpretation")
        elif resp.status == "pending_approval":
            events.append(f"approval:{resp.approval.approval_type}")
            resp = svc.resume(resp.thread_id, decision="approve", reviewer="evaluator", comment="auto-approved by eval")
        else:
            break
    return resp, events


def answer_quality(svc: ResearchService, items: list[dict]) -> dict[str, Any]:
    rows = []
    for it in items:
        t0 = time.perf_counter()
        resp, events = run_question(svc, it)
        latency = round((time.perf_counter() - t0) * 1000)
        state = svc.graph.get_state(svc._config(resp.thread_id)).values
        evidence: list[Evidence] = state.get("evidence", []) or []
        answer = resp.answer or ""
        try:
            judge = svc.deps.llm.structured(
                JudgeScores, prompts.JUDGE,
                f"Question: {it['question']}\n\nReference answer: {it['reference']}\n\nEvidence:\n"
                f"{prompts.render_evidence(evidence, max_chars=1000)}\n\nAnswer to evaluate:\n{answer}",
                purpose="judge", tier="smart",
                context={"question": it["question"], "answer": answer, "reference": it["reference"], "evidence": evidence},
            )
        except Exception as exc:  # noqa: BLE001
            judge = JudgeScores(faithfulness=0, answer_relevance=0, citation_quality=0, completeness=0, rationale=f"judge failed: {exc}")
        abstained = not resp.citations
        rows.append({
            "id": it["id"], "status": resp.status, "route": str(resp.route), "events": events, "latency_ms": latency,
            "n_citations": len(resp.citations), "n_conflicts": len(resp.conflicts),
            "groundedness": resp.groundedness, "confidence": resp.confidence,
            "conflict_ok": (len(resp.conflicts) > 0) if it.get("expect_conflict") else None,
            "abstain_ok": abstained if it.get("expect_abstain") else None,
            **judge.model_dump(),
        })

    def avg(key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 3) if vals else None

    conflict_rows = [r for r in rows if r["conflict_ok"] is not None]
    abstain_rows = [r for r in rows if r["abstain_ok"] is not None]
    return {
        "n": len(rows),
        "faithfulness": avg("faithfulness"),
        "answer_relevance": avg("answer_relevance"),
        "citation_quality": avg("citation_quality"),
        "completeness": avg("completeness"),
        "groundedness": avg("groundedness"),
        "conflict_detection_rate": round(sum(r["conflict_ok"] for r in conflict_rows) / len(conflict_rows), 3) if conflict_rows else None,
        "abstention_rate_on_unanswerable": round(sum(r["abstain_ok"] for r in abstain_rows) / len(abstain_rows), 3) if abstain_rows else None,
        "avg_latency_ms": avg("latency_ms"),
        "rows": rows,
    }


# ----------------------------------------------------------------------------- report


def to_markdown(report: dict[str, Any]) -> str:
    r, ro, a = report["retrieval"], report["routing"], report.get("answers")
    k = r["k"]
    lines = [
        "# Evaluation Report", "",
        f"- LLM: `{report['llm']}` · Web search: `{report['web']}` · Embeddings: `{report['embeddings']}`",
        f"- Golden set: {report['n_items']} questions · generated {report['generated_at']}", "",
        "## Retrieval (document level)", "",
        f"| metric | value |\n|---|---|\n| hit@{k} | {r[f'hit@{k}']} |\n| recall@{k} | {r[f'recall@{k}']} |\n| MRR | {r['mrr']} |\n| n | {r['n']} |", "",
        "## Routing", "", f"Accuracy: **{ro['accuracy']}** (n={ro['n']})", "",
        "| expected \\ predicted | " + " | ".join(sorted({p for v in ro["confusion"].values() for p in v})) + " |",
    ]
    preds = sorted({p for v in ro["confusion"].values() for p in v})
    lines.append("|---" * (len(preds) + 1) + "|")
    for exp, row in sorted(ro["confusion"].items()):
        lines.append(f"| {exp} | " + " | ".join(str(row.get(p, 0)) for p in preds) + " |")
    if a:
        lines += [
            "", "## Answer quality (LLM-as-judge)", "",
            "| metric | value |", "|---|---|",
            *[f"| {m} | {a[m]} |" for m in ("faithfulness", "answer_relevance", "citation_quality", "completeness",
                                             "groundedness", "conflict_detection_rate", "abstention_rate_on_unanswerable", "avg_latency_ms")],
            "", "### Per question", "",
            "| id | status | route | citations | conflicts | faith | relev | cite | compl | events |",
            "|---|---|---|---|---|---|---|---|---|---|",
            *[f"| {x['id']} | {x['status']} | {x['route']} | {x['n_citations']} | {x['n_conflicts']} | {x['faithfulness']} | "
              f"{x['answer_relevance']} | {x['citation_quality']} | {x['completeness']} | {', '.join(x['events'])} |" for x in a["rows"]],
        ]
    if report["llm"] == "mock":
        lines += ["", "> Note: run in offline **mock** mode, so routing, synthesis and judging are keyword heuristics. "
                  "Set `OPENROUTER_API_KEY` (and `TAVILY_API_KEY`) for real scores."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Evaluate the research agent")
    ap.add_argument("--skip-answers", action="store_true", help="only retrieval + routing (fast, no full runs)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)

    svc = build_service()
    items = load_golden()[: args.limit]
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm": svc.deps.llm.name, "web": svc.deps.web.name, "embeddings": svc.settings.embedding_provider,
        "n_items": len(items),
        "retrieval": retrieval_metrics(svc, items, k=args.k),
        "routing": routing_accuracy(svc, items),
    }
    if not args.skip_answers:
        report["answers"] = answer_quality(svc, items)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "eval_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (REPORTS / "eval_report.md").write_text(to_markdown(report), encoding="utf-8")
    summary = {"retrieval": {k: v for k, v in report["retrieval"].items() if k != "rows"},
               "routing_accuracy": report["routing"]["accuracy"]}
    if "answers" in report:
        summary["answers"] = {k: v for k, v in report["answers"].items() if k != "rows"}
    print(json.dumps(summary, indent=2))
    print(f"\nReport written to {REPORTS / 'eval_report.md'}")
    return report


if __name__ == "__main__":
    main()
