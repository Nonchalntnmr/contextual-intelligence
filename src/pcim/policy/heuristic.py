"""Non-learning baselines. These are the bar the learned policy has to clear.

B2 is meant to be the strongest memory system a careful engineer would build without
outcome learning — the Generative-Agents recipe (relevance + recency + importance) plus
temporal filtering, near-duplicate suppression, kind priors and an inclusion threshold.
In G0 its ten parameters are tuned against *noiseless true quality* on the training
users, which is more information than any learned policy is given. If PCIM cannot beat
B2 tuned that way, it has not earned the word "learned".
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..types import KIND_INDEX, KINDS
from .candidates import CandidateSet
from .select import Selection, greedy_select


class SimilarityPolicy:
    """B1 — naive vector memory: top similarity until the budget is full.

    Ignores temporal status on purpose (it is the "vector DB of everything ever said"
    baseline), so it will happily send a superseded preference.
    """

    name = "B1-similarity"

    def select(self, cs: CandidateSet, task_probs: np.ndarray, budget: int, rng=None) -> Selection:
        idx = greedy_select(cs.sim, cs.tokens, budget, threshold=-np.inf, forced=cs.pinned)
        return Selection(idx=idx, scores=cs.sim, policy=self.name)


@dataclasses.dataclass
class HeuristicParams:
    w_recency: float = 0.2
    w_confidence: float = 0.1
    w_tokens: float = 0.0  # per 100 tokens; negative penalises long items
    kind_bonus: tuple[float, ...] = (0.0,) * len(KINDS)
    threshold: float = 0.35
    recency_half_life_days: float = 90.0
    dedup_cos: float = 0.92
    rank_density: bool = False  # order by (score − threshold) per token instead of score

    @classmethod
    def random(cls, rng: np.random.Generator, allow_density: bool = False) -> "HeuristicParams":
        p = cls(
            w_recency=float(rng.uniform(0, 1)),
            w_confidence=float(rng.uniform(0, 1)),
            w_tokens=float(rng.uniform(-0.3, 0.0)),
            kind_bonus=tuple(float(x) for x in rng.uniform(-0.5, 0.5, len(KINDS))),
            threshold=float(rng.uniform(0.0, 0.8)),
        )
        if allow_density:  # drawn only when enabled, so G0's random stream is unchanged
            p.rank_density = bool(rng.random() < 0.5)
        return p


class HeuristicPolicy:
    """B2 — strong temporal heuristic memory."""

    name = "B2-heuristic"

    def __init__(self, params: HeuristicParams | None = None):
        self.p = params or HeuristicParams()

    def scores(self, cs: CandidateSet) -> np.ndarray:
        p = self.p
        rec = 0.5 ** (cs.age_days / p.recency_half_life_days)
        return (
            cs.sim
            + p.w_recency * rec
            + p.w_confidence * cs.confidence
            + p.w_tokens * cs.tokens / 100.0
            + np.asarray(p.kind_bonus)[cs.kind]
        )

    def select(self, cs: CandidateSet, task_probs: np.ndarray, budget: int, rng=None) -> Selection:
        s = self.scores(cs)
        if self.p.rank_density:
            key, thr = (s - self.p.threshold) / np.maximum(cs.tokens, 1), 0.0
        else:
            key, thr = s, self.p.threshold
        idx = greedy_select(
            key, cs.tokens, budget, emb=cs.emb, threshold=thr, dedup_cos=self.p.dedup_cos,
            forced=cs.pinned & ~cs.superseded, eligible=~cs.superseded,
        )
        return Selection(idx=idx, scores=s, policy=self.name)


class ProfilePlusSimilarityPolicy:
    """C — proxy for native assistant memory: a few saved preferences/constraints are
    always injected, the rest of the budget goes to similarity retrieval.

    This is a *proxy*. Real native memory (ChatGPT, Claude, Gemini) is a product
    feature that cannot be run at a controlled token budget through an API; docs/04
    explains how G1 handles that comparison.
    """

    name = "C-profile+similarity"

    def __init__(self, n_profile: int = 5):
        self.n_profile = n_profile

    def select(self, cs: CandidateSet, task_probs: np.ndarray, budget: int, rng=None) -> Selection:
        standing = np.isin(cs.kind, [KIND_INDEX["preference"], KIND_INDEX["constraint"]]) & ~cs.superseded
        rank = cs.confidence * 0.5 ** (cs.age_days / 180.0)
        prof = np.zeros(len(cs), dtype=bool)
        if standing.any():
            order = np.flatnonzero(standing)[np.argsort(-rank[standing])][: self.n_profile]
            prof[order] = True
        idx = greedy_select(
            cs.sim, cs.tokens, budget, threshold=-np.inf, forced=prof | (cs.pinned & ~cs.superseded),
            eligible=~cs.superseded,
        )
        return Selection(idx=idx, scores=cs.sim, policy=self.name)
