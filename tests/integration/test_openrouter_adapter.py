"""Contract tests for the OpenRouter adapter against a local fake OpenAI-compatible server.

Verifies the exact wire format we send (endpoint, auth, attribution headers, JSON-schema
structured output, provider routing, no sampling params, fallback models) and our handling
of usage, refusals, retries and non-retryable errors, without needing a real API key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from research_agent.config import Settings
from research_agent.llm.openrouter_llm import OpenRouterLLM
from research_agent.resilience import ToolError
from research_agent.schemas import QueryClassification

GOOD = {"route": "internal", "reasoning": "company policy", "needs_current_info": False,
        "internal_query": "pto days", "web_query": "", "clarification_question": "", "direct_answer": ""}


def completion(model: str, content: str | None, finish_reason: str = "stop", refusal: str | None = None) -> dict:
    return {
        "id": "gen-1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content, "refusal": refusal}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class FakeOpenRouter(BaseHTTPRequestHandler):
    requests: list[dict] = []
    script: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        FakeOpenRouter.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        step = FakeOpenRouter.script.pop(0) if len(FakeOpenRouter.script) > 1 else FakeOpenRouter.script[0]
        status, payload = step(body) if callable(step) else step
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def ok(body):
    return 200, completion(body["model"], json.dumps(GOOD))


@pytest.fixture
def fake_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeOpenRouter.requests = []
    FakeOpenRouter.base_url = f"http://127.0.0.1:{server.server_port}/api/v1"
    yield FakeOpenRouter
    server.shutdown()


def make_llm(fake, **kw) -> OpenRouterLLM:
    return OpenRouterLLM(Settings(_env_file=None, openrouter_api_key="sk-or-test", openrouter_base_url=fake.base_url,
                                  tool_retry_base_delay_s=0.01, **kw))


def test_structured_output_wire_format(fake_api):
    fake_api.script = [ok]
    out = make_llm(fake_api, openrouter_app_url="https://example.org").structured(
        QueryClassification, "system prompt", "Query: pto", purpose="classify", tier="smart")
    assert isinstance(out, QueryClassification) and out.route == "internal"
    req = fake_api.requests[-1]
    assert req["path"] == "/api/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer sk-or-test"
    assert req["headers"]["x-title"] == "Enterprise Research Agent"
    assert req["headers"]["http-referer"] == "https://example.org"
    body = req["body"]
    assert body["model"] == "anthropic/claude-sonnet-5.5"
    assert "temperature" not in body and "tool_choice" not in body
    assert body["response_format"]["type"] == "json_schema"
    assert "route" in body["response_format"]["json_schema"]["schema"]["properties"]
    assert body["provider"] == {"require_parameters": True}
    assert body["messages"][0] == {"role": "system", "content": "system prompt"}
    assert "models" not in body


def test_fast_tier_and_fallback_models(fake_api):
    fake_api.script = [ok]
    make_llm(fake_api, openrouter_fallback_models=["openai/gpt-5.5"]).structured(
        QueryClassification, "s", "p", purpose="classify", tier="fast")
    body = fake_api.requests[-1]["body"]
    assert body["model"] == "anthropic/claude-haiku-5.5"
    assert body["models"] == ["anthropic/claude-haiku-5.5", "openai/gpt-5.5"]


def test_usage_is_recorded(fake_api, tmp_path):
    from research_agent.observability.tracing import get_metrics_sink, set_metrics_sink
    from research_agent.persistence.db import Database

    prev = get_metrics_sink()
    db = Database(tmp_path / "m.db")
    set_metrics_sink(db)
    try:
        fake_api.script = [ok]
        make_llm(fake_api).structured(QueryClassification, "s", "p", purpose="classify")
        row = db.metrics_summary()["llm"][0]
        assert row["input_tokens"] == 11 and row["output_tokens"] == 7 and row["failures"] == 0
    finally:
        set_metrics_sink(prev)


def test_retries_on_rate_limit_then_succeeds(fake_api):
    fake_api.script = [(429, {"error": {"code": 429, "message": "rate limited"}}), ok]
    out = make_llm(fake_api).structured(QueryClassification, "s", "p", purpose="classify")
    assert out.route == "internal" and len(fake_api.requests) == 2


def test_insufficient_credits_is_not_retried(fake_api):
    fake_api.script = [(402, {"error": {"code": 402, "message": "Insufficient credits"}})]
    with pytest.raises(ToolError):
        make_llm(fake_api).structured(QueryClassification, "s", "p", purpose="classify")
    assert len(fake_api.requests) == 1


def test_refusal_is_not_retried(fake_api):
    fake_api.script = [lambda body: (200, completion(body["model"], None, refusal="I can't help with that."))]
    with pytest.raises(ToolError, match="refused"):
        make_llm(fake_api).structured(QueryClassification, "s", "p", purpose="classify")
    assert len(fake_api.requests) == 1


def test_invalid_json_is_retried(fake_api):
    fake_api.script = [lambda body: (200, completion(body["model"], '{"route": "nonsense"}')), ok]
    out = make_llm(fake_api).structured(QueryClassification, "s", "p", purpose="classify")
    assert out.route == "internal" and len(fake_api.requests) == 2
