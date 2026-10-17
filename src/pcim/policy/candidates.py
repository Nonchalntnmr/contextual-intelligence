"""Candidate sets: the struct-of-arrays view every policy scores."""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np

from ..types import KIND_INDEX, MemoryItem, utcnow


@dataclasses.dataclass
class CandidateSet:
    ids: list[str]
    sim: np.ndarray  # cosine similarity to the task
    kind: np.ndarray  # int index into KINDS
    age_days: np.ndarray
    confidence: np.ndarray  # effective (decayed) confidence
    tokens: np.ndarray  # int
    superseded: np.ndarray  # bool — only ever True in experiments (production filters them)
    pinned: np.ndarray  # bool
    sensitivity: np.ndarray  # int rank
    emb: np.ndarray  # (n, d) unit rows, for near-duplicate suppression

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, mask: np.ndarray) -> "CandidateSet":
        idx = np.flatnonzero(mask)
        return CandidateSet(
            ids=[self.ids[i] for i in idx], sim=self.sim[idx], kind=self.kind[idx],
            age_days=self.age_days[idx], confidence=self.confidence[idx], tokens=self.tokens[idx],
            superseded=self.superseded[idx], pinned=self.pinned[idx], sensitivity=self.sensitivity[idx],
            emb=self.emb[idx],
        )


def build_candidates(
    items: list[MemoryItem],
    item_emb: dict[str, np.ndarray],
    task_emb: np.ndarray,
    confidence_fn,
    now: dt.datetime | None = None,
    top_sim: int = 30,
    standing_max: int = 30,
    recent_episodes: int = 10,
) -> CandidateSet:
    """Multi-channel candidate generation, as in G0: similarity, standing items, recency.

    Similarity alone misses items whose relevance is not lexical/semantic — a dietary
    constraint is relevant to "plan dinner on Friday" whether or not the words overlap —
    which is exactly the structure a learned policy is supposed to find. So candidates
    come from three channels and the policy decides.
    """
    now = now or utcnow()
    items = [i for i in items if i.id in item_emb]
    if not items:
        d = task_emb.shape[0]
        e = np.zeros(0)
        return CandidateSet([], e, e.astype(int), e, e, e.astype(int), e.astype(bool), e.astype(bool), e.astype(int), np.zeros((0, d)))
    emb = np.vstack([item_emb[i.id] for i in items])
    sims = emb @ task_emb
    chosen: dict[int, None] = {}
    for j in np.argsort(-sims)[:top_sim]:
        chosen[int(j)] = None
    standing = [j for j, it in enumerate(items) if it.kind in ("preference", "constraint", "goal")]
    standing.sort(key=lambda j: items[j].valid_from, reverse=True)
    for j in standing[:standing_max]:
        chosen[j] = None
    episodes = [j for j, it in enumerate(items) if it.kind == "episode"]
    episodes.sort(key=lambda j: items[j].valid_from, reverse=True)
    for j in episodes[:recent_episodes]:
        chosen[j] = None
    for j, it in enumerate(items):  # pinned items are always candidates
        if it.pinned:
            chosen[j] = None
    idx = np.array(sorted(chosen), dtype=int)
    sel = [items[j] for j in idx]
    return CandidateSet(
        ids=[i.id for i in sel],
        sim=sims[idx].astype(float),
        kind=np.array([KIND_INDEX[i.kind] for i in sel], dtype=int),
        age_days=np.array([i.age_days(now) for i in sel], dtype=float),
        confidence=np.array([confidence_fn(i, now) for i in sel], dtype=float),
        tokens=np.array([i.tokens for i in sel], dtype=int),
        superseded=np.array([not i.is_current(now) for i in sel], dtype=bool),
        pinned=np.array([i.pinned for i in sel], dtype=bool),
        sensitivity=np.array([i.sensitivity.rank for i in sel], dtype=int),
        emb=emb[idx],
    )
