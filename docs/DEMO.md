# Demonstration: Query Scenarios

Generated 2026-11-20 20:11 · LLM: `mock` · web search: `mock` · embeddings: `huggingface` · KB chunks: 72

> ℹ️ Generated in **offline mock mode** (no API keys): routing/synthesis are keyword heuristics and the answers are extractive; web results come from a labelled fictional fixture corpus. With `OPENROUTER_API_KEY` and `TAVILY_API_KEY` set, the same script produces real LLM answers and live web results.


---

## 1. Internal knowledge-base question (RAG)

_Routed to internal retrieval only; every claim cites internal documents._

**User** `POST /chat`
```json
{
  "message": "What is our parental leave policy and how many PTO days do employees get?"
}
```
**Status:** `answered` · **route:** `internal` · **confidence:** `0.9` · **groundedness:** `1.0` · **latency:** 201 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → validate → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> **From internal LedgerLeaf documents:**
> - Full-time employees receive 25 days of paid time off (PTO) per year plus 10 public holidays. [S1]
> - Up to 5 unused PTO days can be carried over into the next year. [S1]
> - Primary caregivers receive 20 weeks of fully paid parental leave. [S2]
> - Secondary caregivers receive 8 weeks of fully paid parental leave. [S2]
> - Parental leave can be taken within 12 months of birth or adoption. [S2]
> - Employees receive 10 paid sick days per year. [S3]
> - A doctor's note is required for absences longer than 3 consecutive days. [S3]
> - Employees near an office are expected in the office 2 days per week (Tuesday and Thursday). [S4]
> 
> **Sources**
> - [S1] Employee Handbook — Time Off and Leave — Paid time off (PTO) (internal:hr-leave-policy, 2026-01-05)
> - [S2] Employee Handbook — Time Off and Leave — Parental leave (internal:hr-leave-policy, 2026-01-05)
> - [S3] Employee Handbook — Time Off and Leave — Sick leave (internal:hr-leave-policy, 2026-01-05)
> - [S4] Hybrid Work and Expense Policy — Hybrid work (internal:remote-work-expenses, 2025-09-01)



---

## 2. External web question

_Needs current public information, so routed to web search only._

**User** `POST /chat`
```json
{
  "message": "What are the latest trends in real-time payments adoption?"
}
```
**Status:** `answered` · **route:** `web` · **confidence:** `0.77` · **groundedness:** `1.0` · **latency:** 102 ms

**Agent trace:** `contextualize → classify → privacy_guard → search_web → validate → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> **From external web sources:**
> - Real-time payments adoption accelerated in 2026. [S1]
> - More than 1,500 US financial institutions now participate in FedNow, and instant payment volumes in the EU grew sharply after the Instant Payments Regulation deadlines. [S1]
> - Global fintech funding rose 22% quarter over quarter in Q3 2026, led by payments infrastructure and B2B card issuing companies. [S2]
> - The B2B card issuing market is projected to grow about 20% annually through 2030, driven by virtual cards for vendor payments and expense management. [S3]
> 
> **Sources**
> - [S1] [mock] Real-time payments adoption trends 2026 (https://fintech-daily.example/2026/real-time-payments-trends, 2026-08-14)
> - [S2] [mock] Fintech funding rebounds in Q3 2026 (https://vc-pulse.example/fintech-q3-2026, 2026-10-10)
> - [S3] [mock] Embedded finance and card issuing market outlook (https://fintech-daily.example/2026/card-issuing-outlook, 2026-07-22)



---

## 3. Hybrid question with conflict detection

_Needs internal competitor analysis AND current web data. Internal and web retrieval run in parallel; the conflict agent finds that the 2025 internal analysis is outdated._

**User** `POST /chat`
```json
{
  "message": "How do PayNova's current fees compare to our pricing?"
}
```
**Status:** `answered` · **route:** `both` · **confidence:** `0.9` · **groundedness:** `1.0` · **latency:** 95 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → search_web → validate → detect_conflicts → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> **From internal LedgerLeaf documents:**
> - PayNova is our closest competitor in SMB payment processing. [S1]
> - PayNova charges 2.4% + $0.20 per domestic card transaction on its standard plan. [S1]
> - PayNova offers next-day payouts but has no card issuing product. [S1]
> - Enterprise plan: custom interchange-plus pricing, negotiated by the sales team for merchants above $5M monthly volume. [S3]
> - LedgerLeaf competes on combined processing plus card issuing and on transparent pricing. [S4]
> - Our Standard plan is more expensive than PayNova but cheaper than Brightwire. [S4]
> **From external web sources:**
> - PayNova announced a new standard plan for small businesses. [S5]
> - PayNova now charges 1.9% + $0.20 per domestic card transaction, down from its previous price. [S5]
> - PayNova also launched a card issuing beta for selected merchants. [S5]
> - Brightwire raised its published pricing in March 2026. [S6]
> - Acquirers face fees for merchants above the threshold. [S7]
> - Competition among issuing platforms is increasing pricing pressure on interchange revenue. [S8]
> 
> ### ⚠️ Conflicting information between internal and external sources
> - **PayNova** — Internal: “PayNova charges 2.4% + $0.20 per domestic card transaction on its standard plan.” [S1] vs. External: “PayNova now charges 1.9% + $0.20 per domestic card transaction, down from its previous price.” [S5]. _Assessment:_ The external source is more recent (2026-09-02); internal document dated 2025-02-14 may be outdated.
> 
> **Sources**
> - [S1] Competitor Analysis 2025 — PayNova (internal:competitor-analysis-2025, 2025-02-14)
> - [S3] Pricing and Fees Schedule — Card processing plans (internal:pricing-and-fees, 2026-07-01)
> - [S4] Competitor Analysis 2025 — Positioning (internal:competitor-analysis-2025, 2025-02-14)
> - [S5] [mock] PayNova cuts small-business card fees (https://fintech-daily.example/2026/09/paynova-cuts-fees, 2026-09-02)
> - [S6] [mock] Brightwire raises processing prices for mid-market merchants (https://payments-wire.example/brightwire-pricing-2026, 2026-03-10)
> - [S7] [mock] Visa replaces dispute monitoring programme with VAMP (https://card-network-news.example/visa-vamp, 2025-06-01)
> - [S8] [mock] Embedded finance and card issuing market outlook (https://fintech-daily.example/2026/card-issuing-outlook, 2026-07-22)

<details><summary>Structured conflicts (JSON)</summary>

```json
[
  {
    "topic": "PayNova",
    "internal_claim": "PayNova charges 2.4% + $0.20 per domestic card transaction on its standard plan.",
    "internal_source_ids": [
      "S1"
    ],
    "external_claim": "PayNova now charges 1.9% + $0.20 per domestic card transaction, down from its previous price.",
    "external_source_ids": [
      "S5"
    ],
    "likely_resolution": "The external source is more recent (2026-09-02); internal document dated 2025-02-14 may be outdated.",
    "severity": "medium"
  }
]
```
</details>



---

## 4. Checking an internal assumption against current data

_The treasury policy (Jan 2024) assumes a 5.25–5.50% Fed funds rate; the web says otherwise._

**User** `POST /chat`
```json
{
  "message": "Is our treasury Fed funds rate assumption still valid given the current Fed rate?"
}
```
**Status:** `answered` · **route:** `both` · **confidence:** `0.9` · **groundedness:** `1.0` · **latency:** 80 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → search_web → validate → detect_conflicts → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> **From internal LedgerLeaf documents:**
> - Interest income on customer balances is forecast using this Fed funds rate assumption. [S1]
> - For interest income forecasting, the treasury planning assumption is a federal funds target range of 5.25% to 5.50%. [S1]
> - Excess corporate cash may only be invested in US Treasury bills with a maturity of 6 months or less. [S3]
> **From external web sources:**
> - The Federal Reserve held the federal funds target range at 3.50% to 3.75% at its October 2026 meeting. [S4]
> - Officials signalled that further rate cuts depend on inflation data. [S4]
> - The company said the increase funds new reconciliation and treasury features. [S5]
> 
> ### ⚠️ Conflicting information between internal and external sources
> - **Fed** — Internal: “For interest income forecasting, the treasury planning assumption is a federal funds target range of 5.25% to 5.50%.” [S1] vs. External: “The Federal Reserve held the federal funds target range at 3.50% to 3.75% at its October 2026 meeting.” [S4]. _Assessment:_ The external source is more recent (2026-10-29); internal document dated 2024-01-15 may be outdated.
> 
> **Sources**
> - [S1] Treasury and Safeguarding Policy — Interest rate assumptions (internal:treasury-policy, 2024-01-15)
> - [S3] Treasury and Safeguarding Policy — Investments (internal:treasury-policy, 2024-01-15)
> - [S4] [mock] Federal Reserve holds federal funds target range (https://markets-monitor.example/fed-decision-october-2026, 2026-10-29)
> - [S5] [mock] Brightwire raises processing prices for mid-market merchants (https://payments-wire.example/brightwire-pricing-2026, 2026-03-10)

<details><summary>Structured conflicts (JSON)</summary>

```json
[
  {
    "topic": "Fed",
    "internal_claim": "For interest income forecasting, the treasury planning assumption is a federal funds target range of 5.25% to 5.50%.",
    "internal_source_ids": [
      "S1"
    ],
    "external_claim": "The Federal Reserve held the federal funds target range at 3.50% to 3.75% at its October 2026 meeting.",
    "external_source_ids": [
      "S4"
    ],
    "likely_resolution": "The external source is more recent (2026-10-29); internal document dated 2024-01-15 may be outdated.",
    "severity": "medium"
  }
]
```
</details>



---

## 5. Ambiguous query → clarification (LangGraph interrupt)

_The classifier asks a clarifying question; the graph pauses and resumes from its checkpoint on the user's reply._

**User** `POST /chat`
```json
{
  "message": "What's the limit?"
}
```
**Status:** `needs_clarification` · **route:** `clarify` · **confidence:** `None` · **groundedness:** `None` · **latency:** 17 ms

**Agent trace:** `contextualize → classify`

> 🤔 **Clarification requested:** Which limit do you mean: (a) API rate limits, (b) card spending/velocity limits, or (c) KYC/AML transaction thresholds?


**User clarifies** `POST /chat/thr_70b191c844f2/resume`
```json
{
  "answer": "The API rate limit"
}
```
**Status:** `answered` · **route:** `internal` · **confidence:** `0.9` · **groundedness:** `1.0` · **latency:** 72 ms

**Agent trace:** `contextualize → classify → clarify → classify → privacy_guard → retrieve_internal → validate → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> **From internal LedgerLeaf documents:**
> - The API rate limit is 100 requests per second per live API key, with bursts of up to 200 requests. [S1]
> - In test mode the rate limit is 25 requests per second. [S1]
> - Requests over the limit receive HTTP 429 with a Retry-After header. [S1]
> - The API is versioned by date. [S2]
> - The default card spending limit is $5,000 per day, which administrators can lower or raise up to $50,000 per day. [S3]
> - International cards: an additional 1.5% on top of the plan rate. [S4]
> 
> **Sources**
> - [S1] LedgerLeaf API Developer Guide — Rate limits (internal:api-guide, 2026-05-15)
> - [S2] LedgerLeaf API Developer Guide — Versioning (internal:api-guide, 2026-05-15)
> - [S3] LedgerLeaf Cards — Product Guide — Spend controls (internal:product-cards, 2026-05-01)
> - [S4] Pricing and Fees Schedule — Additional fees (internal:pricing-and-fees, 2026-07-01)



---

## 6. Multi-turn conversation context

_The follow-up uses a pronoun; the contextualizer rewrites it into a standalone question using the thread history._

**User** `POST /chat`
```json
{
  "message": "What uptime does our SLA guarantee?"
}
```
**Status:** `answered` · **route:** `internal` · **confidence:** `0.61` · **groundedness:** `1.0` · **latency:** 68 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → validate → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> **From internal LedgerLeaf documents:**
> - If monthly uptime falls below the SLA uptime commitment, customers receive service credits on their monthly fees:. [S1]
> - LedgerLeaf guarantees 99.95% monthly uptime for the Payments API and the merchant dashboard, excluding scheduled maintenance announced 48 hours in advance. [S2]
> 
> **Sources**
> - [S1] Service Level Agreement (SLA) — Service credits (internal:sla, 2026-01-10)
> - [S2] Service Level Agreement (SLA) — Uptime commitment (internal:sla, 2026-01-10)


**User (follow-up, same thread)** `POST /chat`
```json
{
  "message": "And what service credits do customers get if we miss it?",
  "thread_id": "thr_f4a8ec581447"
}
```
**Status:** `answered` · **route:** `internal` · **confidence:** `0.9` · **groundedness:** `1.0` · **latency:** 67 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → validate → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> **From internal LedgerLeaf documents:**
> - If monthly uptime falls below the SLA uptime commitment, customers receive service credits on their monthly fees:. [S1]
> - Customers must claim service credits within 30 days of the end of the affected month. [S1]
> - Below 99.95% but at least 99.0%: 10% service credit. [S1]
> - Priority 2 (degraded service): first response within 1 hour. [S2]
> - LedgerLeaf guarantees 99.95% monthly uptime for the Payments API and the merchant dashboard, excluding scheduled maintenance announced 48 hours in advance. [S3]
> - After 5 years of service employees are eligible for a 4-week paid sabbatical. [S4]
> 
> **Sources**
> - [S1] Service Level Agreement (SLA) — Service credits (internal:sla, 2026-01-10)
> - [S2] Service Level Agreement (SLA) — Support response times (internal:sla, 2026-01-10)
> - [S3] Service Level Agreement (SLA) — Uptime commitment (internal:sla, 2026-01-10)
> - [S4] Employee Handbook — Time Off and Leave — Sabbatical (internal:hr-leave-policy, 2026-01-05)



---

## 7. Human-in-the-loop: confidential documents

_The answer relies on CONFIDENTIAL documents, so a reviewer must approve before it is released._

**User** `POST /chat`
```json
{
  "message": "What is the status of Project Falcon?",
  "user_id": "analyst-42"
}
```
**Status:** `pending_approval` · **route:** `internal` · **confidence:** `None` · **groundedness:** `None` · **latency:** 55 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → validate → synthesize → check_groundedness`

> 🛑 **Approval required** (`confidential_data`, id `apr_b4708dac4de773c0`): Answer uses CONFIDENTIAL documents: board-minutes-q3-2026, project-falcon


Reviewer checks the queue:

```json
[
  {
    "approval_id": "apr_b4708dac4de773c0",
    "approval_type": "confidential_data",
    "status": "pending",
    "reason": "Answer uses CONFIDENTIAL documents: board-minutes-q3-2026, project-falcon"
  }
]
```

**Reviewer approves** `POST /approvals/apr_b4708dac4de773c0`
```json
{
  "decision": "approve",
  "reviewer": "cfo@ledgerleaf",
  "comment": "OK for deal-team analyst"
}
```
Approval `apr_b4708dac4de773c0` → **approved**; the paused conversation resumes:

**Status:** `answered` · **route:** `internal` · **confidence:** `0.85` · **groundedness:** `1.0` · **latency:** 15 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → validate → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> **From internal LedgerLeaf documents:**
> - Project Falcon due diligence is in progress. [S1]
> - Project Falcon is the planned acquisition of Brightwire. [S2]
> - The board reviewed Project Falcon and approved continued due diligence. [S3]
> - Only the deal team may discuss Project Falcon. [S4]
> - Do not mention Project Falcon outside the company. [S4]
> 
> **Sources**
> - [S1] Project Falcon — Acquisition Plan (CONFIDENTIAL) — Status (internal:project-falcon, 2026-09-30)
> - [S2] Project Falcon — Acquisition Plan (CONFIDENTIAL) — Summary (internal:project-falcon, 2026-09-30)
> - [S3] Board Meeting Minutes — Q3 2026 (CONFIDENTIAL) — Decisions (internal:board-minutes-q3-2026, 2026-10-05)
> - [S4] Project Falcon — Acquisition Plan (CONFIDENTIAL) — Confidentiality (internal:project-falcon, 2026-09-30)



---

## 8. Human-in-the-loop: privacy guard for external search

_The web query would leak a confidential codename to an external search engine. The reviewer rejects it, so the system answers from internal sources only, which then triggers the confidential-data gate._

**User** `POST /chat`
```json
{
  "message": "Search the web for the latest news about Project Falcon"
}
```
**Status:** `pending_approval` · **route:** `both` · **confidence:** `None` · **groundedness:** `None` · **latency:** 18 ms

**Agent trace:** `contextualize → classify`

> 🛑 **Approval required** (`external_search_privacy`, id `apr_b97cc4da9eec5562`): The external web search query contains confidential internal terms: Project Falcon.


**Reviewer rejects external search** `POST /chat/thr_73024e65bd98/resume`
```json
{
  "decision": "reject",
  "reviewer": "legal@ledgerleaf",
  "comment": "Codename must not leave the company"
}
```
**Status:** `pending_approval` · **route:** `both` · **confidence:** `None` · **groundedness:** `None` · **latency:** 65 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → validate → synthesize → check_groundedness`

> 🛑 **Approval required** (`confidential_data`, id `apr_d0a33f48e5cf4efe`): Answer uses CONFIDENTIAL documents: board-minutes-q3-2026, project-falcon

- ⚠️ External web search was skipped: reviewer did not approve sending confidential terms externally.

**Reviewer rejects confidential answer** `POST /chat/thr_73024e65bd98/resume`
```json
{
  "decision": "reject",
  "reviewer": "legal@ledgerleaf",
  "comment": "requester not on deal team"
}
```
**Status:** `rejected` · **route:** `both` · **confidence:** `0.85` · **groundedness:** `1.0` · **latency:** 11 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → validate → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> This answer was withheld after human review (requester not on deal team). Please contact the document owner or your manager if you need access to this information.

- ⚠️ External web search was skipped: reviewer did not approve sending confidential terms externally.


---

## 9. Unanswerable question (hallucination guard)

_Nothing in the sources answers the question, so the system abstains instead of guessing (an empty retrieval would also trigger one reformulated retry first)._

**User** `POST /chat`
```json
{
  "message": "What is our policy on cryptocurrency staking rewards?"
}
```
**Status:** `answered` · **route:** `internal` · **confidence:** `0.1` · **groundedness:** `None` · **latency:** 71 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → validate → synthesize → approval_gate → finalize`

**Answer:**

> The retrieved sources do not contain information that answers this question.



---

## 10. Knowledge-base write with approval

_Adding a document is a sensitive operation: it is queued until approved, then indexed and immediately searchable._

**Request** `POST /documents`
```json
{
  "doc_id": "crypto-policy",
  "title": "Cryptocurrency Policy",
  "classification": "internal",
  "content": "# Cryptocurrency Policy\n\nLedgerLeaf does not offer cryptocurrency staking rewards. Merchants may not accept crypto payments through LedgerLeaf Pay. This policy is reviewed annually by the compliance team.",
  "requested_by": "compliance-lead"
}
```
```json
{
  "status": "pending_approval",
  "approval_id": "apr_5a791b8fc64b",
  "doc_id": "crypto-policy"
}
```

**Approve** `POST /approvals/apr_5a791b8fc64b`
```json
{
  "decision": "approve",
  "reviewer": "kb-admin"
}
```
```json
{
  "approval_id": "apr_5a791b8fc64b",
  "status": "approved",
  "result": {
    "doc_id": "crypto-policy",
    "chunks": 1
  }
}
```

**User asks again** `POST /chat`
```json
{
  "message": "What is our policy on cryptocurrency staking rewards?"
}
```
**Status:** `answered` · **route:** `internal` · **confidence:** `0.61` · **groundedness:** `1.0` · **latency:** 69 ms

**Agent trace:** `contextualize → classify → privacy_guard → retrieve_internal → validate → synthesize → check_groundedness → approval_gate → finalize`

**Answer:**

> **From internal LedgerLeaf documents:**
> - LedgerLeaf does not offer cryptocurrency staking rewards. [S1]
> - This policy is reviewed annually by the compliance team. [S1]
> 
> **Sources**
> - [S1] Cryptocurrency Policy — Cryptocurrency Policy (internal:crypto-policy)



---

## Observability

_Aggregated metrics from `GET /metrics` after the demo run._

| node | runs | avg ms | max ms | errors |
|---|---|---|---|---|
| approval_gate | 13 | 0.0 | 0.1 | 0 |
| classify | 12 | 0.1 | 0.4 | 0 |
| privacy_guard | 12 | 0.1 | 0.7 | 0 |
| contextualize | 11 | 0.0 | 0.0 | 0 |
| finalize | 11 | 0.2 | 1.4 | 0 |
| synthesize | 11 | 0.4 | 0.9 | 0 |
| validate | 11 | 0.5 | 0.8 | 0 |
| check_groundedness | 10 | 0.3 | 0.6 | 0 |
| retrieve_internal | 10 | 44.7 | 136.0 | 0 |
| search_web | 3 | 4.3 | 8.5 | 0 |
| clarify | 2 | 0.1 | 0.1 | 0 |
| detect_conflicts | 2 | 2.1 | 2.6 | 0 |

LLM usage: `[]`

Approvals: `[{"approval_type": "confidential_data", "status": "approved", "n": 1}, {"approval_type": "confidential_data", "status": "rejected", "n": 1}, {"approval_type": "external_search_privacy", "status": "rejected", "n": 1}, {"approval_type": "kb_delete", "status": "approved", "n": 1}, {"approval_type": "kb_write", "status": "approved", "n": 1}]`

