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
    band: tuple[float, float] = (0.2, 0.8)  # the embedder's cosine at relevance 0 and 1 (embed.py)

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, mask: np.ndarray) -> "CandidateSet":
        idx = np.flatnonzero(mask)
        return CandidateSet(
            ids=[self.ids[i] for i in idx], sim=self.sim[idx], kind=self.kind[idx],
            age_days=self.age_days[idx], confidence=self.confidence[idx], tokens=self.tokens[idx],
            superseded=self.superseded[idx], pinned=self.pinned[idx], sensitivity=self.sensitivity[idx],
            emb=self.emb[idx], band=self.band,
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
    band: tuple[float, float] = (0.2, 0.8),
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
        return CandidateSet([], e, e.astype(int), e, e, e.astype(int), e.astype(bool), e.astype(bool), e.astype(int),
                            np.zeros((0, d)), band)
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
    return _assemble(items, np.array(sorted(chosen), dtype=int), sims, emb, confidence_fn, now, band)


# The planned channel's rule, calibrated on the E2 development set (experiments/e2/CALIBRATION.md).
# Floors are in relevance units (relevance_ramp over the embedder's band), not raw cosine.
PER_QUERY, QUERY_FLOOR = 2, 0.25
TASK_TOP, TASK_FLOOR = 2, 0.5


def build_planned_candidates(
    items: list[MemoryItem],
    item_emb: dict[str, np.ndarray],
    task_emb: np.ndarray,
    query_embs: np.ndarray,
    confidence_fn,
    now: dt.datetime | None = None,
    band: tuple[float, float] = (0.2, 0.8),
) -> CandidateSet:
    """Candidates for a request whose retrieval was planned (planner.py).

    Each planned search contributes its best PER_QUERY matches above QUERY_FLOOR, the
    request itself its best TASK_TOP above TASK_FLOOR, and pinned items always come. An
    empty plan means the planner judged that no personal fact matters: only pinned items.
    An item's similarity is its best match over the request and the searches.
    """
    now = now or utcnow()
    items = [i for i in items if i.id in item_emb]
    if not items:
        return build_candidates([], item_emb, task_emb, confidence_fn, now, band=band)
    emb = np.vstack([item_emb[i.id] for i in items])
    task_sims = emb @ task_emb
    chosen: dict[int, None] = {}
    lo, hi = band

    def floor(rel: float) -> float:  # relevance units -> cosine for this embedder
        return lo + rel * (hi - lo)

    sims = task_sims
    if len(query_embs):
        q = emb @ np.asarray(query_embs).T  # (items, queries)
        for c in range(q.shape[1]):
            for j in np.argsort(-q[:, c], kind="stable")[:PER_QUERY]:
                if q[j, c] >= floor(QUERY_FLOOR):
                    chosen[int(j)] = None
        sims = np.maximum(task_sims, q.max(axis=1))
        for j in np.argsort(-task_sims, kind="stable")[:TASK_TOP]:
            if task_sims[j] >= floor(TASK_FLOOR):
                chosen[int(j)] = None
    for j, it in enumerate(items):
        if it.pinned:
            chosen[j] = None
    return _assemble(items, np.array(sorted(chosen), dtype=int), sims, emb, confidence_fn, now, band)


def _assemble(items, idx, sims, emb, confidence_fn, now, band) -> CandidateSet:
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
        band=band,
    )
