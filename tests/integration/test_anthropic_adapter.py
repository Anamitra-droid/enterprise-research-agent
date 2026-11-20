"""Contract tests for the Anthropic adapter against a local fake Messages API server.

Verifies the exact wire format we send (native structured outputs, no sampling params,
refusal-fallback beta) and our handling of usage, refusals, retries and bad output,
without needing a real API key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from research_agent.config import Settings
from research_agent.llm.anthropic_llm import AnthropicLLM
from research_agent.resilience import ToolError
from research_agent.schemas import QueryClassification

GOOD = {"route": "internal", "reasoning": "company policy", "needs_current_info": False,
        "internal_query": "pto days", "web_query": "", "clarification_question": "", "direct_answer": ""}


class FakeAnthropic(BaseHTTPRequestHandler):
    requests: list[dict] = []
    script: list[tuple[int, dict]] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        FakeAnthropic.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        status, payload = FakeAnthropic.script.pop(0) if len(FakeAnthropic.script) > 1 else FakeAnthropic.script[0]
        if status == 200 and "content" not in payload:
            payload = {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                       "content": [{"type": "text", "text": json.dumps(payload["json"])}],
                       "stop_reason": payload.get("stop_reason", "end_turn"), "stop_sequence": None,
                       "usage": {"input_tokens": 11, "output_tokens": 7}}
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_api(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeAnthropic)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    monkeypatch.setenv("ANTHROPIC_API_URL", url)
    FakeAnthropic.requests = []
    yield FakeAnthropic
    server.shutdown()


def make_llm(**kw) -> AnthropicLLM:
    return AnthropicLLM(Settings(_env_file=None, anthropic_api_key="sk-test", tool_retry_base_delay_s=0.01, **kw))


def test_structured_output_wire_format(fake_api):
    fake_api.script = [(200, {"json": GOOD})]
    out = make_llm().structured(QueryClassification, "system prompt", "Query: pto", purpose="classify", tier="smart")
    assert isinstance(out, QueryClassification) and out.route == "internal"
    req = fake_api.requests[-1]
    body = req["body"]
    assert body["model"] == "claude-sonnet-5-5"
    assert "temperature" not in body  # Sonnet 5.5 rejects non-default sampling params
    assert "tool_choice" not in body  # forced tool use is rejected; we use native structured outputs
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema" and "route" in fmt["schema"]["properties"]
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")


def test_fast_tier_uses_haiku_without_fallback(fake_api):
    fake_api.script = [(200, {"json": GOOD})]
    make_llm().structured(QueryClassification, "s", "p", purpose="classify", tier="fast")
    body = fake_api.requests[-1]["body"]
    assert body["model"] == "claude-haiku-5-5" and "fallbacks" not in body


def test_retries_on_overload_then_succeeds(fake_api):
    fake_api.script = [(529, {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
                       (200, {"json": GOOD})]
    out = make_llm().structured(QueryClassification, "s", "p", purpose="classify")
    assert out.route == "internal" and len(fake_api.requests) == 2


def test_bad_request_is_not_retried(fake_api):
    fake_api.script = [(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})]
    with pytest.raises(ToolError):
        make_llm().structured(QueryClassification, "s", "p", purpose="classify")
    assert len(fake_api.requests) == 1


def test_refusal_is_not_retried(fake_api):
    fake_api.script = [(200, {"json": GOOD, "stop_reason": "refusal"})]
    with pytest.raises(ToolError, match="refused"):
        make_llm().structured(QueryClassification, "s", "p", purpose="classify")
    assert len(fake_api.requests) == 1


def test_fallback_can_be_disabled(fake_api):
    fake_api.script = [(200, {"json": GOOD})]
    make_llm(anthropic_server_fallback=False).structured(QueryClassification, "s", "p", purpose="classify", tier="smart")
    req = fake_api.requests[-1]
    assert "fallbacks" not in req["body"] and "server-side-fallback" not in req["headers"].get("anthropic-beta", "")
