"""Deterministic offline model. It does not reason; it reports what it was given.

Useful for exercising the whole loop (consent → selection → egress → render → response →
feedback → update) without a key or a network, and for asserting in tests exactly which
notes reached the model.
"""

from __future__ import annotations

import re
import time

from ..types import approx_tokens
from .base import ModelResponse, RenderedPrompt, Usage

_NOTE = re.compile(r"^\[(\d+)\] \(([^)]*)\) (.*)$", re.MULTILINE)


class MockAdapter:
    provider = "mock"

    def __init__(self, model: str = "mock-1"):
        self.model = model
        self.calls: list[RenderedPrompt] = []

    def generate(self, prompt: RenderedPrompt, max_tokens: int | None = None) -> ModelResponse:
        t0 = time.perf_counter()
        self.calls.append(prompt)
        notes = _NOTE.findall(prompt.user)
        task = prompt.user.rsplit("Request:\n", 1)[-1].strip()
        lines = [f"[mock model] Request: {task[:200]}"]
        if notes:
            lines.append(f"I was given {len(notes)} note(s) about you:")
            lines += [f"  [{n}] {text[:120]}" for n, _, text in notes]
        else:
            lines.append("I was given no notes about you.")
        text = "\n".join(lines)
        usage = Usage(input_tokens=approx_tokens(prompt.system + prompt.user), output_tokens=approx_tokens(text))
        return ModelResponse(
            text=text, usage=usage, cost_usd=0.0, latency_ms=(time.perf_counter() - t0) * 1000,
            stop_reason="end_turn", model=self.model, provider=self.provider,
        )
