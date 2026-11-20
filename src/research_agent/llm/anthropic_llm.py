"""Anthropic Claude implementation via langchain-anthropic.

Notes on current Claude models (Sonnet 5.5 / Haiku 5.5):
- Forced `tool_choice` is rejected by Sonnet 5.5, so LangChain's default
  `function_calling` structured-output method is NOT used. We use
  `method="json_schema"` (native structured outputs, `output_config.format`).
- Non-default sampling params are rejected, so `temperature` is never set.
- Refusals (`stop_reason == "refusal"`) are surfaced as non-retryable errors.
"""

from __future__ import annotations

import os
import time
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.config import Settings
from research_agent.llm.base import LLMClient, T, Tier
from research_agent.observability.logging import get_logger
from research_agent.observability.tracing import get_metrics_sink
from research_agent.resilience import NonRetryableError, call_with_retry

log = get_logger("research_agent.llm")

# Models that support the server-side refusal fallback beta.
_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5-2"}


class StructuredOutputError(Exception):
    """Model returned output that failed schema validation (retryable)."""


class AnthropicLLM(LLMClient):
    name = "anthropic"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: dict[str, ChatAnthropic] = {}

    def _chat(self, tier: Tier) -> ChatAnthropic:
        model = self.settings.smart_model if tier == "smart" else self.settings.fast_model
        if model not in self._models:
            kwargs: dict[str, Any] = {
                "model": model,
                "api_key": self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
                "max_tokens": self.settings.llm_max_tokens,
                "timeout": self.settings.llm_timeout_s,
                "max_retries": 0,  # retries are handled (and logged) by call_with_retry
            }
            if self.settings.anthropic_server_fallback and model in _FALLBACK_MODELS:
                kwargs["betas"] = ["server-side-fallback-2026-07-01"]
                kwargs["model_kwargs"] = {"extra_body": {"fallbacks": "default"}}
            self._models[model] = ChatAnthropic(**kwargs)
        return self._models[model]

    def structured(self, schema: type[T], system: str, prompt: str, *, purpose: str, tier: Tier = "fast", context: dict[str, Any] | None = None) -> T:
        chat = self._chat(tier)
        runnable = chat.with_structured_output(schema, method="json_schema", include_raw=True)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

        def _invoke() -> T:
            start = time.perf_counter()
            status = "ok"
            usage: dict[str, Any] = {}
            try:
                out = runnable.invoke(messages)
                raw = out.get("raw")
                usage = getattr(raw, "usage_metadata", None) or {}
                stop = (getattr(raw, "response_metadata", None) or {}).get("stop_reason")
                if stop == "refusal":
                    status = "refusal"
                    raise NonRetryableError(f"model refused the request ({purpose})")
                if out.get("parsing_error") or out.get("parsed") is None:
                    status = "parse_error"
                    raise StructuredOutputError(str(out.get("parsing_error")))
                parsed = out["parsed"]
                return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
            except (NonRetryableError, StructuredOutputError):
                raise
            except Exception:
                status = "error"
                raise
            finally:
                ms = (time.perf_counter() - start) * 1000
                sink = get_metrics_sink()
                if sink:
                    sink.record_llm_call(purpose, chat.model, usage.get("input_tokens"), usage.get("output_tokens"), ms, status)
                log.info("llm_call", purpose=purpose, model=chat.model, ms=round(ms), status=status,
                         in_tok=usage.get("input_tokens"), out_tok=usage.get("output_tokens"))

        return call_with_retry(
            f"llm:{purpose}",
            _invoke,
            max_attempts=self.settings.tool_max_retries,
            base_delay=self.settings.tool_retry_base_delay_s,
        )
