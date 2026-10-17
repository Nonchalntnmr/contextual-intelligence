"""Types shared by every layer. Kept dependency-free so the simulator can use them too."""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import uuid
from typing import Any

# The memory kinds and task types are a closed vocabulary on purpose: the context policy
# learns one value per (task type, kind) cell, so every new label is a new parameter that
# has to be learned from feedback.
KINDS: tuple[str, ...] = ("fact", "preference", "goal", "constraint", "episode", "skill")
TASK_TYPES: tuple[str, ...] = ("code", "writing", "research", "planning", "learning")
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}
TASK_INDEX = {t: i for i, t in enumerate(TASK_TYPES)}

# Kinds that shape future behaviour rather than describe the world. An untrusted source
# may never create one of these directly (see sovereignty/injection.py).
STANDING_KINDS = frozenset({"preference", "goal", "constraint"})


class Trust(str, enum.Enum):
    """Where an assertion came from. Higher rank = more authority over CI's behaviour."""

    UNTRUSTED = "untrusted"  # text from a source that can carry adversarial instructions
    INFERRED = "inferred"  # CI derived it; always carries a confidence below 1
    IMPORTED = "imported"  # from a connector the user granted
    USER_STATED = "user_stated"  # the user said it
    USER_CONFIRMED = "user_confirmed"  # the user reviewed it in the dashboard and confirmed

    @property
    def rank(self) -> int:
        return _TRUST_RANK[self]


_TRUST_RANK = {
    Trust.UNTRUSTED: 0,
    Trust.INFERRED: 1,
    Trust.IMPORTED: 2,
    Trust.USER_STATED: 3,
    Trust.USER_CONFIRMED: 4,
}


class Sensitivity(str, enum.Enum):
    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"  # health, finances, relationships, location history
    SECRET = "secret"  # credentials, government IDs: never leaves the device

    @property
    def rank(self) -> int:
        return _SENS_RANK[self]


_SENS_RANK = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.PERSONAL: 1,
    Sensitivity.SENSITIVE: 2,
    Sensitivity.SECRET: 3,
}


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def approx_tokens(text: str) -> int:
    """Provider-neutral token estimate (≈4 chars/token for English).

    Used for budgeting only. Billing uses the provider's own usage numbers, and the
    Anthropic adapter can call count_tokens when an exact pre-flight count matters.
    """
    return max(1, (len(text) + 3) // 4)


@dataclasses.dataclass
class MemoryItem:
    """One assertion in the bitemporal store.

    Valid time  [valid_from, valid_to)      — when the assertion is true in the world.
    System time [recorded_at, superseded_at) — when CI believed it.
    A correction never overwrites: it closes the old row's system time and links forward.
    Deletion is different: it destroys the row and its key (crypto-shredding).
    """

    id: str
    user_id: str
    kind: str
    content: str
    source: str
    trust: Trust
    sensitivity: Sensitivity
    confidence: float
    valid_from: dt.datetime
    recorded_at: dt.datetime
    key: str | None = None
    value: Any = None
    valid_to: dt.datetime | None = None
    superseded_at: dt.datetime | None = None
    superseded_by: str | None = None
    status: str = "active"  # active | retracted
    pinned: bool = False
    parents: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    provenance: dict = dataclasses.field(default_factory=dict)
    tokens: int = 0

    def is_current(self, now: dt.datetime | None = None) -> bool:
        now = now or utcnow()
        return (
            self.status == "active"
            and self.superseded_at is None
            and self.valid_from <= now
            and (self.valid_to is None or self.valid_to > now)
        )

    def age_days(self, now: dt.datetime | None = None) -> float:
        now = now or utcnow()
        return max(0.0, (now - self.valid_from).total_seconds() / 86400.0)


@dataclasses.dataclass
class Task:
    text: str
    task_type_probs: dict[str, float]
    embedding: Any = None  # np.ndarray, kept untyped to keep this module numpy-free

    @property
    def task_type(self) -> str:
        return max(self.task_type_probs, key=self.task_type_probs.get)
