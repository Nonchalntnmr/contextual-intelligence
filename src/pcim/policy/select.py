"""Budgeted greedy selection, shared by every policy so that only the scores differ."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Selection:
    idx: np.ndarray  # selected candidate indices, in inclusion order
    scores: np.ndarray  # per-candidate scores the decision used
    propensities: np.ndarray | None = None  # per-candidate P(included) under the policy
    x: np.ndarray | None = None  # regression vector, for learned policies
    policy: str = ""
    version: int = 0

    @property
    def n(self) -> int:
        return len(self.idx)


def greedy_select(
    scores: np.ndarray,
    tokens: np.ndarray,
    budget: int,
    emb: np.ndarray | None = None,
    threshold: float = 0.0,
    dedup_cos: float | None = None,
    forced: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Take items in descending score order while they clear `threshold` and fit `budget`.

    `forced` items (user-pinned) go first regardless of score. `eligible=False` items are
    never taken. With `dedup_cos`, an item within that cosine of an already-selected item
    is skipped: a near-duplicate adds tokens and no information.
    """
    n = len(scores)
    if n == 0 or budget <= 0:
        return np.zeros(0, dtype=int)
    ok = np.ones(n, dtype=bool) if eligible is None else eligible.copy()
    picked: list[int] = []
    used = 0

    def try_add(j: int) -> None:
        nonlocal used
        if used + tokens[j] > budget:
            return
        if dedup_cos is not None and emb is not None and picked:
            if float(np.max(emb[picked] @ emb[j])) > dedup_cos:
                return
        picked.append(j)
        used += int(tokens[j])

    if forced is not None:
        for j in np.flatnonzero(forced & ok):
            try_add(int(j))
    for j in np.argsort(-scores, kind="stable"):
        j = int(j)
        if not ok[j] or j in picked or scores[j] <= threshold:
            continue
        try_add(j)
    return np.array(picked, dtype=int)
