"""Retry helpers for flaky external calls (LLM, web search, vector store)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from research_agent.observability.logging import get_logger
from research_agent.observability.tracing import get_metrics_sink

log = get_logger("research_agent.retry")
T = TypeVar("T")


class ToolError(Exception):
    """Raised when an external tool fails after all retries."""

    def __init__(self, tool: str, cause: BaseException):
        super().__init__(f"{tool} failed after retries: {type(cause).__name__}: {cause}")
        self.tool = tool
        self.cause = cause


class NonRetryableError(Exception):
    """Wrap errors that must not be retried (e.g. bad request / auth)."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, NonRetryableError):
        return False
    # OpenAI-compatible SDK (OpenRouter): retry rate limits, timeouts, connection and 5xx errors; not 4xx (e.g. 402 no credits).
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429):
        return False
    return not isinstance(exc, (ValueError, TypeError, KeyError))


def call_with_retry(
    tool: str,
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
) -> T:
    """Run `fn` with exponential backoff + jitter. Raises ToolError when exhausted."""

    def _before_sleep(rs: RetryCallState) -> None:
        exc = rs.outcome.exception() if rs.outcome else None
        log.warning("tool_retry", tool=tool, attempt=rs.attempt_number, error=str(exc)[:200])
        sink = get_metrics_sink()
        if sink:
            sink.record_tool_call(tool, rs.attempt_number, "retry", str(exc)[:300])

    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=base_delay, max=8, jitter=base_delay),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_before_sleep,
        reraise=True,
    )
    sink = get_metrics_sink()
    try:
        for attempt in retrying:
            with attempt:
                result = fn()
        if sink:
            sink.record_tool_call(tool, retrying.statistics.get("attempt_number", 1), "ok")
        return result  # type: ignore[possibly-undefined]
    except Exception as exc:
        if sink:
            sink.record_tool_call(tool, retrying.statistics.get("attempt_number", 1), "failed", str(exc)[:300])
        raise ToolError(tool, exc) from exc
