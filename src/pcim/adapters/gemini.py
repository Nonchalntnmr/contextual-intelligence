"""Gemini adapter over the generateContent REST endpoint (httpx).

Written against the documented v1beta request/response shape and tested against a mock
transport. **Not yet run against the live API** in this repository's test suite.
Thinking tokens (`thoughtsTokenCount`) are billed as output and counted as output here.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import ModelResponse, RenderedPrompt, Usage
from .pricing import cost_usd

BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiAdapter:
    provider = "google"

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 8192,
                 transport: httpx.BaseTransport | None = None, timeout: float = 120.0):
        self.model = model
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.max_tokens = max_tokens
        self.http = httpx.Client(base_url=BASE, headers={"x-goog-api-key": key}, timeout=timeout, transport=transport)

    def generate(self, prompt: RenderedPrompt, max_tokens: int | None = None) -> ModelResponse:
        body = {
            "systemInstruction": {"parts": [{"text": prompt.system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt.user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens or self.max_tokens},
        }
        t0 = time.perf_counter()
        r = self.http.post(f"/models/{self.model}:generateContent", json=body)
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        d = r.json()
        cands = d.get("candidates") or []
        finish = cands[0].get("finishReason", "") if cands else (d.get("promptFeedback") or {}).get("blockReason", "")
        parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        refusal = finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII") or not cands
        um = d.get("usageMetadata") or {}
        cached = um.get("cachedContentTokenCount", 0) or 0
        usage = Usage(
            input_tokens=max(0, (um.get("promptTokenCount") or 0) - cached),
            output_tokens=(um.get("candidatesTokenCount") or 0) + (um.get("thoughtsTokenCount") or 0),
            cache_read_tokens=cached,
        )
        return ModelResponse(
            text=text if not refusal else f"[declined by the model ({finish or 'blocked'})]",
            usage=usage, cost_usd=cost_usd(self.model, usage), latency_ms=latency, stop_reason=finish,
            model=d.get("modelVersion", self.model), provider=self.provider, request_id=d.get("responseId"),
            refusal=refusal,
        )
