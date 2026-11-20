"""Offline, deterministic heuristic "LLM".

Lets the whole system (API, graph, demo, tests) run with no API keys. Each `purpose`
maps to a small rule-based function that consumes the structured `context` the agent
passes alongside its prompt. It is intentionally simple and NOT a substitute for a real
model: answers are extractive and routing is keyword-based.
"""

from __future__ import annotations

import re
from typing import Any

from research_agent.llm.base import LLMClient, T, Tier
from research_agent.schemas import (
    Claim,
    ClaimVerdict,
    Conflict,
    ConflictReport,
    ConversationSummary,
    Evidence,
    GroundednessReport,
    JudgeScores,
    QueryClassification,
    ReformulatedQuery,
    RelevanceGrade,
    RelevanceGrades,
    Route,
    StandaloneQuery,
    SynthesizedAnswer,
)
from research_agent.text_utils import content_tokens, numbers_in, overlap, sentences, tokens

GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|who are you|what can you do)\b", re.I)
FOLLOW_UP = re.compile(r"^\s*(and|also|what about|how about|and what|what if|why|is that|does that|do they|are they)\b|\b(it|that|they|them|those|this|their|its)\b", re.I)

INTERNAL_HINTS = {
    "our", "we", "internal", "policy", "policies", "ledgerleaf", "employee", "employees", "company",
    "sla", "pto", "leave", "parental", "kyc", "aml", "pci", "project", "falcon", "salary", "board",
    "q1", "q2", "q3", "q4", "quarter", "quarterly", "revenue", "pricing", "fee", "fees", "api",
    "onboarding", "incident", "retention", "expense", "stipend", "treasury", "fraud", "chargeback",
    "handbook", "remote", "uptime", "credits", "merchant", "merchants", "tpv", "security", "gdpr",
    "rate-limit", "webhook", "webhooks", "assumption", "minutes", "bands", "brazil", "expansion",
}
WEB_HINTS = {
    "latest", "current", "currently", "today", "news", "recent", "recently", "now", "trend", "trends",
    "industry", "market", "regulation", "regulations", "regulator", "publicly", "worldwide", "global",
    "web", "internet", "online", "2026", "announced", "fed", "federal", "adoption", "outlook",
}
BOTH_HINTS = {"compare", "compared", "vs", "versus", "still", "valid", "match", "benchmark", "against", "competitor", "competitors", "paynova", "brightwire"}
AMBIGUOUS_TERMS = {"limit", "limits", "fee", "fees", "rate", "rates", "policy", "it", "that", "this", "number", "status", "deadline", "threshold", "cost"}

CLARIFY_OPTIONS = {
    "limit": "Which limit do you mean: (a) API rate limits, (b) card spending/velocity limits, or (c) KYC/AML transaction thresholds?",
    "limits": "Which limits do you mean: (a) API rate limits, (b) card spending/velocity limits, or (c) KYC/AML transaction thresholds?",
    "fee": "Which fee do you mean: (a) our card processing fees, (b) international/FX fees, or (c) a competitor's fees?",
    "fees": "Which fees do you mean: (a) our card processing fees, (b) international/FX fees, or (c) a competitor's fees?",
    "rate": "Which rate do you mean: (a) the Fed funds rate, (b) our processing rate, or (c) the API rate limit?",
}


class MockLLM(LLMClient):
    name = "mock"

    def structured(self, schema: type[T], system: str, prompt: str, *, purpose: str, tier: Tier = "fast", context: dict[str, Any] | None = None) -> T:
        ctx = context or {}
        handler = getattr(self, f"_{purpose}", None)
        if handler is None:
            raise NotImplementedError(f"MockLLM has no heuristic for purpose={purpose!r}")
        result = handler(ctx)
        assert isinstance(result, schema), f"mock {purpose} returned {type(result).__name__}, expected {schema.__name__}"
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _best_sentences(query: str, text: str, n: int = 2) -> list[str]:
        q = content_tokens(query)
        scored = sorted(((overlap(q, s), i, s) for i, s in enumerate(sentences(text))), key=lambda x: (-x[0], x[1]))
        return [s for score, _, s in scored[:n] if score > 0]

    # ------------------------------------------------------------------ purposes
    def _contextualize(self, ctx: dict) -> StandaloneQuery:
        query: str = ctx["query"]
        history: list[tuple[str, str]] = ctx.get("history", [])
        prev_user = next((t for r, t in reversed(history) if r == "user"), None)
        if prev_user and (FOLLOW_UP.search(query) or len(content_tokens(query)) <= 3):
            return StandaloneQuery(standalone_query=f"{query.rstrip('?')} (regarding: {prev_user.rstrip('?')})?", is_follow_up=True)
        return StandaloneQuery(standalone_query=query, is_follow_up=False)

    def _summarize(self, ctx: dict) -> ConversationSummary:
        parts = [f"{r}: {t[:120]}" for r, t in ctx.get("history", [])]
        prev = ctx.get("previous_summary", "")
        return ConversationSummary(summary=((prev + " | ") if prev else "") + " / ".join(parts)[:1500])

    def _classify(self, ctx: dict) -> QueryClassification:
        q: str = ctx["query"]
        toks = set(tokens(q))
        ctoks = content_tokens(q)
        can_clarify = ctx.get("clarification_count", 0) < ctx.get("max_clarifications", 2)

        if GREETING.search(q) and len(ctoks) <= 4:
            return QueryClassification(route=Route.DIRECT, reasoning="Greeting / small talk.", needs_current_info=False,
                                       direct_answer="Hello! I'm the LedgerLeaf research assistant. Ask me about internal policies, products and reports, or about current fintech and market information.")

        has_entity = bool(re.search(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-zA-Z]{2,}", q.strip()))
        if can_clarify and not has_entity and len(ctoks) <= 2 and (ctoks & AMBIGUOUS_TERMS or not ctoks):
            term = next(iter(sorted(ctoks & AMBIGUOUS_TERMS)), "that")
            question = CLARIFY_OPTIONS.get(term, f"Could you be more specific about '{term}'? For example, are you asking about an internal LedgerLeaf policy or about public market information?")
            return QueryClassification(route=Route.CLARIFY, reasoning="Query is too short/ambiguous to resolve a single intent.",
                                       needs_current_info=False, clarification_question=question)

        internal = bool(toks & INTERNAL_HINTS)
        web = bool(toks & WEB_HINTS)
        both = bool(toks & BOTH_HINTS)
        if (internal and web) or (both and (internal or web)) or both:
            route = Route.BOTH
        elif web:
            route = Route.WEB
        else:
            route = Route.INTERNAL
        q_clean = re.sub(r"^(search the web for|look up|find)\s+", "", q.strip(), flags=re.I)
        return QueryClassification(
            route=route,
            reasoning=f"keyword routing: internal={internal} web={web} comparative={both}",
            needs_current_info=web,
            internal_query=q_clean if route in (Route.INTERNAL, Route.BOTH) else "",
            web_query=q_clean if route in (Route.WEB, Route.BOTH) else "",
        )

    def _grade(self, ctx: dict) -> RelevanceGrades:
        q = content_tokens(ctx["query"])
        grades = []
        for ev in ctx["evidence"]:
            ov = overlap(q, ev.title + " " + ev.content)
            grades.append(RelevanceGrade(source_id=ev.source_id, relevant=ov >= 0.25 or (ov > 0 and ev.score >= 0.6), reason=f"lexical overlap {ov:.2f}"))
        return RelevanceGrades(grades=grades)

    def _reformulate(self, ctx: dict) -> ReformulatedQuery:
        q = " ".join(sorted(content_tokens(ctx["query"])))
        return ReformulatedQuery(internal_query=q if ctx.get("needs_internal") else "", web_query=q if ctx.get("needs_web") else "")

    def _conflicts(self, ctx: dict) -> ConflictReport:
        query: str = ctx["query"]
        internal: list[Evidence] = ctx["internal"]
        web: list[Evidence] = ctx["web"]
        topics = [w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", query) if w.lower() not in {"what", "how", "is", "does", "our", "the", "and", "compare", "who", "which"}]
        topics += sorted(content_tokens(query))
        conflicts: list[Conflict] = []
        seen: set[str] = set()
        pairs_seen: set[tuple[str, str]] = set()
        for topic in topics:
            t = topic.lower()
            if t in seen:
                continue
            for iev in internal:
                i_sents = [s for s in sentences(iev.content) if t in s.lower() and numbers_in(s)]
                for wev in web:
                    w_sents = [s for s in sentences(wev.content) if t in s.lower() and numbers_in(s)]
                    for is_ in i_sents:
                        for ws in w_sents:
                            shared = content_tokens(is_) & content_tokens(ws)
                            if len(shared) >= 3 and _numbers_disagree(is_, ws) and (is_, ws) not in pairs_seen:
                                pairs_seen.add((is_, ws))
                                conflicts.append(Conflict(
                                    topic=topic, internal_claim=is_, internal_source_ids=[iev.source_id],
                                    external_claim=ws, external_source_ids=[wev.source_id],
                                    likely_resolution=_resolution(iev, wev), severity="medium"))
                                seen.add(t)
                                break
                        if t in seen:
                            break
                    if t in seen:
                        break
                if t in seen:
                    break
            if len(conflicts) >= 2:
                break
        return ConflictReport(conflicts=conflicts)

    def _synthesize(self, ctx: dict) -> SynthesizedAnswer:
        query: str = ctx["query"]
        evidence: list[Evidence] = ctx["evidence"]
        claims: list[Claim] = []
        lines: list[str] = []
        for kind, heading in (("internal", "From internal LedgerLeaf documents"), ("web", "From external web sources")):
            group = [e for e in evidence if e.kind == kind]
            if not group:
                continue
            lines.append(f"**{heading}:**")
            for ev in sorted(group, key=lambda e: -e.score)[:4]:
                for s in self._best_sentences(query, ev.content, n=3):
                    s = s.rstrip(".")
                    lines.append(f"- {s}. [{ev.source_id}]")
                    claims.append(Claim(text=s, source_ids=[ev.source_id]))
        if not claims:
            return SynthesizedAnswer(answer="The retrieved sources do not contain information that answers this question.", claims=[], confidence=0.1, gaps=[query])
        conf = min(0.9, 0.45 + 0.08 * len(claims))
        return SynthesizedAnswer(answer="\n".join(lines), claims=claims, confidence=round(conf, 2), gaps=[])

    def _groundedness(self, ctx: dict) -> GroundednessReport:
        by_id = {e.source_id: e for e in ctx["evidence"]}
        verdicts = []
        for c in ctx["claims"]:
            text = " ".join(by_id[s].content for s in c.source_ids if s in by_id)
            ok = bool(text) and overlap(c.text, text) >= 0.6
            verdicts.append(ClaimVerdict(claim=c.text, supported=ok, reason="lexical support" if ok else "claim not found in cited sources"))
        score = sum(v.supported for v in verdicts) / len(verdicts) if verdicts else 1.0
        return GroundednessReport(verdicts=verdicts, score=round(score, 3))

    def _judge(self, ctx: dict) -> JudgeScores:
        answer: str = ctx["answer"]
        question: str = ctx["question"]
        reference: str = ctx.get("reference", "")
        ev_text = " ".join(e.content for e in ctx.get("evidence", []))
        body = answer.split("**Sources**")[0]
        # statement lines (citations trail the sentence, so score line-wise)
        sents = [ln.strip() for ln in body.splitlines()
                 if len(ln.strip()) > 25 and not ln.strip().endswith(":") and not ln.lstrip().startswith(("#", ">"))]
        cited = [s for s in sents if re.search(r"\[S\d+\]", s)]
        faith = overlap(re.sub(r"\[S\d+\]", "", answer), ev_text) if ev_text else 0.0
        return JudgeScores(
            faithfulness=round(min(1.0, faith), 3),
            answer_relevance=round(min(1.0, overlap(question, answer) * 1.2), 3),
            citation_quality=round(len(cited) / len(sents), 3) if sents else 0.0,
            completeness=round(overlap(reference, answer), 3) if reference else 0.5,
            rationale="heuristic mock judge (lexical overlap)",
        )


def _numbers_disagree(a: str, b: str) -> bool:
    """Same kind of figure, different values (percentages compared first, then all numbers)."""
    na, nb = numbers_in(a), numbers_in(b)
    pa, pb = {n for n in na if n.endswith("%")}, {n for n in nb if n.endswith("%")}
    if pa and pb:
        return pa != pb
    return bool(na and nb) and not (na & nb)


def _resolution(iev: Evidence, wev: Evidence) -> str:
    if iev.last_updated and wev.last_updated:
        newer = "external" if wev.last_updated > iev.last_updated else "internal"
        return f"The {newer} source is more recent ({wev.last_updated if newer == 'external' else iev.last_updated}); internal document dated {iev.last_updated} may be outdated."
    return "Internal document may be outdated; verify against the external source."
