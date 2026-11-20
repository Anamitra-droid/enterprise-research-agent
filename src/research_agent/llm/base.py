"""Provider-agnostic LLM interface used by every agent.

Agents call `llm.structured(Schema, system, prompt, purpose=..., context=...)`.
- Real providers use `system` + `prompt`.
- The offline `MockLLM` ignores the prose and uses the structured `context` dict
  (raw query, evidence objects...) to compute deterministic heuristic outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
Tier = Literal["fast", "smart"]


class LLMClient(ABC):
    name: str = "base"

    @abstractmethod
    def structured(
        self,
        schema: type[T],
        system: str,
        prompt: str,
        *,
        purpose: str,
        tier: Tier = "fast",
        context: dict[str, Any] | None = None,
    ) -> T:
        """Return an instance of `schema` produced by the model."""
