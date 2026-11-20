"""Node-level tracing: timing, structured logs and persisted metrics for every graph node."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from langgraph.config import get_config
from langgraph.errors import GraphInterrupt

from research_agent.observability.logging import get_logger

log = get_logger("research_agent.trace")

# Set by the container at startup so nodes can persist metrics without import cycles.
_metrics_sink: Any = None


def set_metrics_sink(db: Any) -> None:
    global _metrics_sink
    _metrics_sink = db


def get_metrics_sink() -> Any:
    return _metrics_sink


def traced_node(name: str) -> Callable:
    """Decorator for LangGraph node functions.

    Logs start/end, measures latency, persists a `node_runs` row and appends the node
    name to `state.trace` so API responses show the path taken through the graph.
    LangGraph interrupts (clarification / HITL) are re-raised untouched.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            thread_id = None
            try:
                thread_id = get_config().get("configurable", {}).get("thread_id")
            except Exception:  # noqa: BLE001 - called outside a graph run (unit tests)
                pass
            turn_id = getattr(state, "turn_id", None)
            start = time.perf_counter()
            status, error = "ok", None
            try:
                result = fn(state, *args, **kwargs) or {}
                if isinstance(result, dict):
                    result = {**result, "trace": [name]}
                return result
            except GraphInterrupt:
                status = "interrupted"
                raise
            except Exception as exc:  # noqa: BLE001 - recorded then re-raised
                status, error = "error", f"{type(exc).__name__}: {exc}"
                log.exception("node_failed", node=name)
                raise
            finally:
                ms = (time.perf_counter() - start) * 1000
                log.info("node_done", node=name, ms=round(ms, 1), status=status)
                if _metrics_sink is not None:
                    try:
                        _metrics_sink.record_node(thread_id, turn_id, name, ms, status, error)
                    except Exception:  # noqa: BLE001 - metrics must never break the graph
                        log.warning("metrics_write_failed", node=name)

        return wrapper

    return decorator
