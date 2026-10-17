"""OpenAI-compatible Chat Completions adapter over httpx.

One request shape covers OpenAI itself and the many servers that speak it (vLLM, Ollama,
LM Studio, llama.cpp server, OpenRouter, Together, ...). `provider="local"` points it at
a server on this machine, which is the one configuration where SENSITIVE items may be
granted egress without leaving the device.

Written against the documented /v1/chat/completions shape and tested against a mock
transport. **Not yet run against a live endpoint** in this repository's test suite.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import ModelResponse, RenderedPrompt, Usage
from .pricing import cost_usd

DEFAULT_BASE = {"openai": "https://api.openai.com/v1", "local": "http://127.0.0.1:11434/v1"}


class OpenAICompatAdapter:
    def __init__(
        self,
        model: str,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        system_role: str = "system",
        token_param: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
    ):
        self.model, self.provider = model, provider
        self.base_url = (base_url or os.environ.get("PCIM_OPENAI_BASE_URL") or DEFAULT_BASE[provider]).rstrip("/")
        key = api_key or os.environ.get("OPENAI_API_KEY" if provider == "openai" else "PCIM_LOCAL_API_KEY", "")
        if provider == "openai" and not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.max_tokens = max_tokens
        self.system_role = system_role
        # api.openai.com uses max_completion_tokens; most compatible servers use max_tokens.
        self.token_param = token_param or ("max_completion_tokens" if provider == "openai" else "max_tokens")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport)

    def generate(self, prompt: RenderedPrompt, max_tokens: int | None = None) -> ModelResponse:
        body = {
            "model": self.model,
            "messages": [
                {"role": self.system_role, "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            self.token_param: max_tokens or self.max_tokens,
        }
        t0 = time.perf_counter()
        r = self.http.post("/chat/completions", json=body)
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        d = r.json()
        choice = d["choices"][0]
        msg = choice.get("message") or {}
        refusal = bool(msg.get("refusal"))
        text = msg.get("refusal") if refusal else (msg.get("content") or "")
        u = d.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        usage = Usage(
            input_tokens=max(0, (u.get("prompt_tokens") or 0) - cached),
            output_tokens=u.get("completion_tokens") or 0,
            cache_read_tokens=cached,
        )
        served = d.get("model", self.model)
        return ModelResponse(
            text=text, usage=usage, cost_usd=cost_usd(served, usage), latency_ms=latency,
            stop_reason=choice.get("finish_reason") or "", model=served, provider=self.provider,
            request_id=r.headers.get("x-request-id") or d.get("id"), refusal=refusal,
        )
