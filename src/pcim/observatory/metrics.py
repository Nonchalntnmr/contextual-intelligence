"""Statistics for paired, user-clustered comparisons.

Interactions within a user are not independent — a policy that learned a user's
preferences keeps winning on that user — so every interval here resamples *users*, and
every comparison is paired within user (same tasks, same candidates, common random
numbers for feedback).
"""

from __future__ import annotations

import numpy as np


def window_mean(Q: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Per-user mean over interactions lo..hi (1-indexed, inclusive). Q: (users, T)."""
    return Q[:, lo - 1:hi].mean(axis=1)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff(Qa: np.ndarray, Qb: np.ndarray, lo: int, hi: int, n_boot: int,
                rng: np.random.Generator) -> dict:
    d = window_mean(Qa, lo, hi) - window_mean(Qb, lo, hi)
    m, l, h = bootstrap_mean_ci(d, n_boot, rng)
    return {"mean": m, "lo": l, "hi": h, "window": [lo, hi]}


def capture_fraction(Qe: np.ndarray, Qb: np.ndarray, Qo: np.ndarray, lo: int, hi: int, n_boot: int,
                     rng: np.random.Generator) -> dict:
    """(E − B) / (O − B), as a ratio of user-averaged means, with a user bootstrap."""
    e, b, o = (window_mean(Q, lo, hi) for Q in (Qe, Qb, Qo))
    n = len(e)
    point = float((e - b).mean() / max((o - b).mean(), 1e-12))
    idx = rng.integers(0, n, size=(n_boot, n))
    num = (e - b)[idx].mean(axis=1)
    den = np.maximum((o - b)[idx].mean(axis=1), 1e-12)
    r = num / den
    return {"mean": point, "lo": float(np.quantile(r, 0.025)), "hi": float(np.quantile(r, 0.975)),
            "headroom": float((o - b).mean()), "window": [lo, hi]}


def onset(Qe: np.ndarray, Qd: np.ndarray, width: int, n_boot: int, rng: np.random.Generator) -> int | None:
    """End of the first `width`-interaction window whose E − D lower 95% bound is > 0."""
    T = Qe.shape[1]
    for start in range(1, T + 1, width):
        end = min(start + width - 1, T)
        if paired_diff(Qe, Qd, start, end, n_boot, rng)["lo"] > 0:
            return end
    return None


def identifiability(est: np.ndarray, true: np.ndarray) -> dict:
    """Per-user Pearson correlation between estimated and true deviation vectors."""
    cors = []
    for a, b in zip(est, true):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(a, b)[0, 1]))
    cors = np.array(cors)
    return {"median": float(np.median(cors)), "q25": float(np.quantile(cors, 0.25)),
            "q75": float(np.quantile(cors, 0.75)), "n": int(len(cors))}


def learning_curve(Q: np.ndarray, width: int = 25) -> list[float]:
    T = Q.shape[1]
    return [float(Q[:, s:min(s + width, T)].mean()) for s in range(0, T, width)]
