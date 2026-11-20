"""OpenRouter implementation via langchain-openai (OpenRouter exposes an OpenAI-compatible API).

Notes:
- Structured outputs use `method="json_schema"` (OpenAI `response_format`), and
  `provider.require_parameters=true` makes OpenRouter route only to providers that
  support it. The default Claude 5.5 models support structured outputs on OpenRouter.
- `temperature` is never sent: current Claude models reject non-default sampling params.
- Refusals / content-filter stops are surfaced as non-retryable errors.
- Optional `OPENROUTER_FALLBACK_MODELS` uses OpenRouter's server-side model fallback.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from research_agent.config import Settings
from research_agent.llm.base import LLMClient, T, Tier
from research_agent.observability.logging import get_logger
from research_agent.observability.tracing import get_metrics_sink
from research_agent.resilience import NonRetryableError, call_with_retry

log = get_logger("research_agent.llm")


class StructuredOutputError(Exception):
    """Model returned output that failed schema validation (retryable)."""


class OpenRouterLLM(LLMClient):
    name = "openrouter"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: dict[str, ChatOpenAI] = {}

    def _chat(self, tier: Tier) -> ChatOpenAI:
        s = self.settings
        model = s.smart_model if tier == "smart" else s.fast_model
        if model not in self._models:
            headers = {"X-Title": s.openrouter_app_name}
            if s.openrouter_app_url:
                headers["HTTP-Referer"] = s.openrouter_app_url
            extra_body: dict[str, Any] = {"provider": {"require_parameters": True}}
            if s.openrouter_fallback_models:
                extra_body["models"] = [model, *[m for m in s.openrouter_fallback_models if m != model]]
            self._models[model] = ChatOpenAI(
                model=model,
                api_key=s.openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url=s.openrouter_base_url,
                max_tokens=s.llm_max_tokens,
                timeout=s.llm_timeout_s,
                max_retries=0,  # retries are handled (and logged) by call_with_retry
                default_headers=headers,
                extra_body=extra_body,
            )
        return self._models[model]

    def structured(self, schema: type[T], system: str, prompt: str, *, purpose: str, tier: Tier = "fast", context: dict[str, Any] | None = None) -> T:
        chat = self._chat(tier)
        runnable = chat.with_structured_output(schema, method="json_schema", include_raw=True)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

        def _invoke() -> T:
            start = time.perf_counter()
            status = "ok"
            usage: dict[str, Any] = {}
            served_by = chat.model_name
            try:
                try:
                    out = runnable.invoke(messages)
                except (ValidationError, json.JSONDecodeError) as exc:
                    status = "parse_error"  # model emitted schema-invalid JSON: worth a retry
                    raise StructuredOutputError(str(exc)[:500]) from exc
                raw = out.get("raw")
                usage = getattr(raw, "usage_metadata", None) or {}
                meta = getattr(raw, "response_metadata", None) or {}
                served_by = meta.get("model_name") or served_by
                refusal = (getattr(raw, "additional_kwargs", None) or {}).get("refusal")
                if refusal or meta.get("finish_reason") in ("content_filter", "refusal"):
                    status = "refusal"
                    raise NonRetryableError(f"model refused the request ({purpose}): {refusal or meta.get('finish_reason')}")
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
                    sink.record_llm_call(purpose, served_by, usage.get("input_tokens"), usage.get("output_tokens"), ms, status)
                log.info("llm_call", purpose=purpose, model=served_by, ms=round(ms), status=status,
                         in_tok=usage.get("input_tokens"), out_tok=usage.get("output_tokens"))

        return call_with_retry(
            f"llm:{purpose}",
            _invoke,
            max_attempts=self.settings.tool_max_retries,
            base_delay=self.settings.tool_retry_base_delay_s,
        )
