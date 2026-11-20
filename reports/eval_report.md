# Evaluation Report

- LLM: `mock` · Web search: `mock` · Embeddings: `huggingface`
- Golden set: 30 questions · generated 2026-11-20 18:44:14

## Retrieval (document level)

| metric | value |
|---|---|
| hit@5 | 1.0 |
| recall@5 | 1.0 |
| MRR | 0.981 |
| n | 26 |

## Routing

Accuracy: **0.9** (n=30)

| expected \ predicted | both | clarify | internal | web |
|---|---|---|---|---|
| both | 4 | 0 | 0 | 0 |
| clarify | 0 | 2 | 0 | 0 |
| internal | 2 | 0 | 19 | 0 |
| web | 1 | 0 | 0 | 2 |

## Answer quality (LLM-as-judge)

| metric | value |
|---|---|
| faithfulness | 0.718 |
| answer_relevance | 0.892 |
| citation_quality | 0.762 |
| completeness | 0.883 |
| groundedness | 1.0 |
| conflict_detection_rate | 1.0 |
| abstention_rate_on_unanswerable | 1.0 |
| avg_latency_ms | 49.8 |

### Per question

| id | status | route | citations | conflicts | faith | relev | cite | compl | events |
|---|---|---|---|---|---|---|---|---|---|
| q01 | answered | internal | 4 | 0 | 0.843 | 1.0 | 0.857 | 0.923 |  |
| q02 | answered | internal | 1 | 0 | 0.615 | 0.96 | 0.75 | 1.0 |  |
| q03 | answered | internal | 2 | 0 | 0.871 | 0.8 | 0.667 | 1.0 |  |
| q04 | answered | internal | 3 | 0 | 0.911 | 1.0 | 0.833 | 1.0 |  |
| q05 | answered | internal | 4 | 0 | 0.884 | 1.0 | 0.875 | 1.0 |  |
| q06 | answered | internal | 1 | 0 | 0.625 | 0.9 | 0.667 | 0.875 |  |
| q07 | answered | internal | 3 | 0 | 0.831 | 1.0 | 0.875 | 0.889 |  |
| q08 | answered | internal | 4 | 0 | 0.775 | 0.9 | 0.857 | 1.0 |  |
| q09 | answered | internal | 4 | 0 | 0.769 | 0.857 | 0.833 | 1.0 |  |
| q10 | answered | internal | 3 | 0 | 0.712 | 1.0 | 0.833 | 0.909 |  |
| q11 | answered | internal | 3 | 0 | 0.692 | 0.9 | 0.8 | 1.0 |  |
| q12 | answered | internal | 1 | 0 | 0.688 | 0.9 | 0.5 | 1.0 |  |
| q13 | answered | internal | 3 | 0 | 0.737 | 0.6 | 0.8 | 1.0 |  |
| q14 | answered | internal | 2 | 0 | 0.838 | 0.857 | 0.8 | 0.875 |  |
| q15 | answered | internal | 4 | 0 | 0.84 | 1.0 | 0.8 | 1.0 |  |
| q16 | answered | both | 6 | 1 | 0.705 | 1.0 | 0.8 | 1.0 |  |
| q17 | answered | both | 7 | 0 | 0.653 | 1.0 | 0.818 | 0.75 | approval:confidential_data |
| q18 | answered | internal | 2 | 0 | 0.694 | 0.48 | 0.667 | 0.667 |  |
| q19 | answered | internal | 4 | 0 | 0.803 | 1.0 | 0.833 | 0.875 |  |
| q20 | answered | internal | 4 | 0 | 0.702 | 1.0 | 0.833 | 1.0 |  |
| q21 | answered | web | 3 | 0 | 0.742 | 1.0 | 0.8 | 1.0 |  |
| q22 | answered | web | 2 | 0 | 0.733 | 0.9 | 0.75 | 0.9 |  |
| q23 | answered | both | 5 | 1 | 0.598 | 0.857 | 0.75 | 1.0 | approval:confidential_data |
| q24 | answered | both | 7 | 1 | 0.67 | 1.0 | 0.867 | 0.611 |  |
| q25 | answered | both | 4 | 1 | 0.63 | 0.857 | 0.778 | 0.667 |  |
| q26 | answered | both | 6 | 2 | 0.615 | 1.0 | 0.846 | 1.0 |  |
| q27 | answered | both | 2 | 0 | 0.729 | 1.0 | 0.6 | 0.421 |  |
| q28 | answered | internal | 4 | 0 | 0.774 | 1.0 | 0.875 | 1.0 | clarification |
| q29 | answered | internal | 4 | 0 | 0.85 | 1.0 | 0.889 | 0.875 | clarification |
| q30 | answered | internal | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.25 |  |

> Note: run in offline **mock** mode, so routing, synthesis and judging are keyword heuristics. Set `ANTHROPIC_API_KEY` (and `TAVILY_API_KEY`) for real scores.
