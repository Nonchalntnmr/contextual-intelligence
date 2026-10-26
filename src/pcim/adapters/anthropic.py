"""Anthropic adapter, on the official `anthropic` SDK (1.x).

- The stable system section (preamble + directives + confirmed profile) carries a
  `cache_control` breakpoint, so repeat requests from the same person read it from the
  prompt cache. It only caches once it is ≥512 tokens on current models; shorter prefixes
  silently do not, and `usage.cache_read_input_tokens` stays 0.
- Retrieved notes and the task go in the user turn, after the breakpoint: they change
  every request and must not invalidate the cached prefix.
- `stop_reason == "refusal"` is checked before content is read.
- Server-side refusal fallbacks (`fallbacks: "default"`, beta
  `server-side-fallback-2026-07-01`) are on by default for the models that support them;
  pass `fallbacks=False` to opt out. Claude Haiku 5.5 has no server-side fallback.
- No temperature/top_p: current models reject sampling parameters.
"""

from __future__ import annotations

import os
import time

from .base import AgentStep, ModelResponse, RenderedPrompt, Usage
from .pricing import cost_usd

FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        fallbacks: bool = True,
        client=None,
        timeout: float = 120.0,
    ):
        import anthropic  # optional dependency: pip install "contextual-intelligence[anthropic]"

        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks and model in FALLBACK_MODELS
        # The endpoint is pinned: ANTHROPIC_BASE_URL in the environment would otherwise send
        # content a project granted to Anthropic to a proxy. PCIM_ANTHROPIC_BASE_URL is the
        # explicit opt-in for an endpoint the person counts as Anthropic.
        self.client = client or anthropic.Anthropic(
            timeout=timeout, base_url=os.environ.get("PCIM_ANTHROPIC_BASE_URL") or "https://api.anthropic.com")

    def _request(self, prompt: RenderedPrompt, max_tokens: int | None) -> dict:
        return dict(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=[{"type": "text", "text": prompt.system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt.user}],
            output_config={"effort": self.effort},
        )

    def generate(self, prompt: RenderedPrompt, max_tokens: int | None = None) -> ModelResponse:
        kwargs = self._request(prompt, max_tokens)
        t0 = time.perf_counter()
        if self.fallbacks:
            resp = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        refusal = resp.stop_reason == "refusal"
        text = "" if refusal else "".join(b.text for b in resp.content if b.type == "text")
        if refusal:
            details = getattr(resp, "stop_details", None)
            cat = getattr(details, "category", None) if details else None
            text = f"[declined by the model{f' ({cat})' if cat else ''}]"
        served_by = resp.model or self.model
        return ModelResponse(
            text=text, usage=usage, cost_usd=cost_usd(served_by, usage), latency_ms=latency,
            stop_reason=resp.stop_reason or "", model=served_by, provider=self.provider,
            request_id=getattr(resp, "_request_id", None) or getattr(resp, "id", None), refusal=refusal,
        )

    def agent_step(self, system: str, messages: list[dict], tools: list[dict],
                   max_tokens: int | None = None) -> AgentStep:
        """One turn of a tool-using session (manual loop, so every call passes the governor).

        Tools are sent with `strict: true` and `tool_choice` left at auto, as current models
        require. The assistant content is returned as plain dicts to be appended back
        unchanged.
        """
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=[{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"],
                    "strict": True} for t in tools],
            output_config={"effort": self.effort},
        )
        if self.fallbacks:
            resp = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        u = resp.usage
        usage = Usage(input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
                      cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                      cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)
        content = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
        refusal = resp.stop_reason == "refusal"
        text = "" if refusal else "".join(b.text for b in resp.content if b.type == "text")
        tool_uses = [] if refusal else [{"id": b.id, "name": b.name, "input": b.input}
                                        for b in resp.content if b.type == "tool_use"]
        served = resp.model or self.model
        return AgentStep(content=content, text=text or ("[declined by the model]" if refusal else ""),
                         tool_uses=tool_uses, stop_reason=resp.stop_reason or "", usage=usage,
                         cost_usd=cost_usd(served, usage), model=served, refusal=refusal)

    def count_tokens(self, prompt: RenderedPrompt) -> int:
        """Exact pre-flight input count from the API (budgeting uses an estimate otherwise)."""
        r = self.client.messages.count_tokens(
            model=self.model, system=prompt.system, messages=[{"role": "user", "content": prompt.user}]
        )
        return r.input_tokens
