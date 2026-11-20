.PHONY: install ingest serve chat test test-live lint eval eval-fast demo docker clean

install:        ## create venv and install all dependencies
	uv sync --extra local-embeddings

ingest:         ## (re)build the ChromaDB index from data/docs
	uv run research-agent ingest --reset

serve:          ## run the API at http://127.0.0.1:8000 (Swagger at /docs)
	uv run research-agent serve --reload

chat:           ## interactive CLI chat (in-process)
	uv run research-agent chat

test:           ## unit + integration tests (offline, no keys)
	uv run pytest

test-live:      ## end-to-end tests against real OpenRouter + Tavily
	uv run pytest -m live

lint:
	uv run ruff check src tests scripts

eval:           ## full evaluation -> reports/eval_report.md
	uv run python -m research_agent.evaluation

eval-fast:      ## retrieval + routing only
	uv run python -m research_agent.evaluation --skip-answers

demo:           ## run the demo scenarios -> docs/DEMO.md
	uv run python scripts/demo.py

docker:
	docker compose up --build

clean:
	rm -rf var .pytest_cache .ruff_cache
