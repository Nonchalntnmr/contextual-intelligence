"""The item feature map φ(m, task) and the regression vector of a selected set.

Value is modelled as type × kind × relevance: "how useful is a relevant *constraint* for
a *planning* task, for this user". Those 30 cells are where both population structure
and personal structure live, so they are the coordinates personalization moves. The rest
are generic: two features let standing kinds stay useful on dissimilar tasks, age for
preferences/goals (old ones may be silently outdated), confidence, length, raw
similarity, and an inclusion bias.

These features were written before G0's simulator, from the design argument above.
They are not the simulator's functional form: relevance there is a sharp sigmoid and
staleness a step at one year; here they are a linear ramp and a linear age trend.
"""

from __future__ import annotations

import numpy as np

from ..types import KIND_INDEX, KINDS, TASK_TYPES
from .candidates import CandidateSet

N_KIND, N_TASK = len(KINDS), len(TASK_TYPES)
_K_CON, _K_PREF, _K_GOAL = KIND_INDEX["constraint"], KIND_INDEX["preference"], KIND_INDEX["goal"]

FEATURE_NAMES: list[str] = (
    [f"value[{t}|{k}]" for k in KINDS for t in TASK_TYPES]
    + ["constraint_offtopic", "preference_offtopic", "recency", "standing_age_years",
       "confidence", "tokens_per_100", "similarity", "similarity_above_0.5", "bias"]
)
N_FEATURES = len(FEATURE_NAMES)  # 39
N_REGRESSION = N_FEATURES + N_TASK  # + per-task-type intercepts


def relevance_ramp(sim: np.ndarray) -> np.ndarray:
    return np.clip((sim - 0.2) / 0.6, 0.0, 1.0)


def featurize(cs: CandidateSet, task_probs: np.ndarray) -> np.ndarray:
    n = len(cs)
    F = np.zeros((n, N_FEATURES))
    if n == 0:
        return F
    s = relevance_ramp(cs.sim)
    cell = cs.kind * N_TASK  # row block for the item's kind
    F[:, :N_KIND * N_TASK][np.arange(n)[:, None], cell[:, None] + np.arange(N_TASK)] = (
        task_probs[None, :] * s[:, None]
    )
    j = N_KIND * N_TASK
    F[:, j] = (cs.kind == _K_CON) * (1.0 - s)
    F[:, j + 1] = (cs.kind == _K_PREF) * (1.0 - s)
    F[:, j + 2] = np.exp(-cs.age_days / 180.0)
    F[:, j + 3] = ((cs.kind == _K_PREF) | (cs.kind == _K_GOAL)) * (cs.age_days / 365.0)
    F[:, j + 4] = cs.confidence
    F[:, j + 5] = cs.tokens / 100.0
    F[:, j + 6] = cs.sim
    F[:, j + 7] = np.maximum(0.0, cs.sim - 0.5)
    F[:, j + 8] = 1.0
    return F


def regression_vector(F: np.ndarray, selected: np.ndarray, task_probs: np.ndarray) -> np.ndarray:
    """x = [Σ_{m in S} φ_m, p(task type)]: reward ≈ x·β under the additive model."""
    summed = F[selected].sum(axis=0) if len(selected) else np.zeros(F.shape[1])
    return np.concatenate([summed, task_probs])
