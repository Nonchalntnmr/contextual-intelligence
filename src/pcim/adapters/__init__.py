"""Layer F — the foundation-model interface.

The one fact that shapes this layer: closed models accept tokens, not embeddings or
adapters. Whatever CI learns about a person reaches GPT, Claude or Gemini only as text it
chose to send. So the portable artifact is the `ContextPacket` (what to say), and each
adapter only translates it into one provider's request shape (how to say it).

    base.py           ContextPacket, RenderedPrompt, ModelResponse, the Adapter protocol
    render.py         packet → prompt: trusted system section, fenced untrusted data block
    pricing.py        per-model prices with an as-of date; unknown models cost None, not 0
    mock.py           deterministic, offline
    anthropic.py      official `anthropic` SDK (1.x), prompt caching, refusal fallbacks
    openai_compat.py  /v1/chat/completions over httpx: OpenAI, vLLM, Ollama, OpenRouter, ...
    gemini.py         Gemini generateContent REST over httpx
"""

from .base import ContextPacket, ModelResponse, PacketItem, RenderedPrompt, Usage
from .render import render

__all__ = ["ContextPacket", "PacketItem", "RenderedPrompt", "ModelResponse", "Usage", "render", "get_adapter"]


def get_adapter(provider: str, model: str | None = None, **kw):
    if provider == "mock":
        from .mock import MockAdapter
        return MockAdapter(model or "mock-1", **kw)
    if provider == "anthropic":
        from .anthropic import AnthropicAdapter
        return AnthropicAdapter(model or "claude-opus-5-5", **kw)
    if provider in ("openai", "local"):
        from .openai_compat import OpenAICompatAdapter
        if not model:
            raise ValueError(f"{provider} adapter needs an explicit model name")
        return OpenAICompatAdapter(model, provider=provider, **kw)
    if provider == "google":
        from .gemini import GeminiAdapter
        if not model:
            raise ValueError("google adapter needs an explicit model name")
        return GeminiAdapter(model, **kw)
    raise ValueError(f"unknown provider {provider!r}")
