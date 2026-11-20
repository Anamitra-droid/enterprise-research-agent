from research_agent.config import Settings
from research_agent.llm.base import LLMClient


def build_llm(settings: Settings) -> LLMClient:
    if settings.resolved_llm_provider == "anthropic":
        from research_agent.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(settings)
    from research_agent.llm.mock_llm import MockLLM

    return MockLLM()


__all__ = ["LLMClient", "build_llm"]
