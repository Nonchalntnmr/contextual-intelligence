"""Provider-neutral types for Layer F."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class PacketItem:
    id: str
    kind: str
    content: str
    source: str
    trust: str
    valid_from: str  # ISO date
    confidence: float
    conflicted: bool = False
    confirmed_instructions: bool = False


@dataclasses.dataclass
class ContextPacket:
    """Everything CI decided to tell the model, before any provider formatting.

    `profile` holds user-confirmed pinned items: trusted, stable across requests, rendered
    in the system section (and therefore cacheable). `memories` holds what the policy
    selected for this task: rendered as fenced *data* in the user turn, never as
    instructions, whatever their trust.
    """

    packet_id: str
    task: str
    directives: list[str]
    profile: list[PacketItem]
    memories: list[PacketItem]
    nonce: str
    meta: dict = dataclasses.field(default_factory=dict)

    def item_ids(self) -> list[str]:
        return [i.id for i in self.profile] + [i.id for i in self.memories]


@dataclasses.dataclass(frozen=True)
class RenderedPrompt:
    system: str  # stable: preamble + directives + profile
    user: str  # volatile: fenced data block + the task

    def digest(self) -> str:
        """Hash of exactly what was sent. Stored instead of the text itself."""
        return hashlib.sha256(json.dumps([self.system, self.user]).encode()).hexdigest()


@dataclasses.dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclasses.dataclass
class ModelResponse:
    text: str
    usage: Usage
    cost_usd: float | None  # None = price unknown for this model; never silently 0
    latency_ms: float
    stop_reason: str
    model: str
    provider: str
    request_id: str | None = None
    refusal: bool = False


@dataclasses.dataclass
class AgentStep:
    """One model turn in a tool-using session.

    `content` is the provider-native assistant content, appended back unchanged on the
    next request. (Current Claude models bind thinking blocks to the conversation, so
    history must be append-only.)
    """

    content: list
    text: str
    tool_uses: list[dict]  # [{"id", "name", "input"}]
    stop_reason: str
    usage: Usage
    cost_usd: float | None
    model: str
    refusal: bool = False


class Adapter(Protocol):
    provider: str
    model: str

    def generate(self, prompt: RenderedPrompt, max_tokens: int | None = None) -> ModelResponse: ...
