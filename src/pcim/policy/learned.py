"""Learned context selection: a Bayesian linear value model with a population prior.

Model. For a selected set S on a task of type distribution p:

    reward ≈ x(S)·β_i + ε,   x(S) = [Σ_{m∈S} φ(m), p],   ε ~ N(0, σ²)
    β_i = β_0 + δ_i,          δ_i ~ N(0, Σ_δ)

β_0 is what the population has taught CI about which context helps which task. δ_i is
what is specific to user i. Σ_δ is learned from the population by EM and projected onto
a low-rank-plus-diagonal form, Σ_δ = U Λ Uᵀ + d·I — so a user's deviation is mostly a
point z_i in a small learned space (the latent part of "Intelligence DNA"), with a little
freedom outside it. In the linear-Gaussian case this *is* meta-learning a user-embedding
space: the prior N(0, UΛUᵀ) over δ is the prior N(0, I) over z pushed through UΛ^½.

Selection. Thompson sampling: draw β̃ from the user's posterior, score items by φ·β̃,
take them greedily under the budget while the score is positive. Items are included
only when the sampled model says they help, so the policy can send *less* context than
the budget allows. Exploration comes from posterior width alone; it shrinks as evidence
accumulates.

Credit assignment. Feedback scores a whole response, not an item. The additive model
assigns credit through regression on x(S), which is identifiable only because Thompson
sampling varies S. That is a modelling assumption with a known failure mode (two items
that are useful only together), discussed in docs/03 §4.

Propensities. `select(..., propensity_samples=K)` estimates each candidate's inclusion
probability by Monte Carlo over K posterior draws, and logs them for off-policy
evaluation of candidate policies before they are promoted (learning.py).
"""

from __future__ import annotations

import dataclasses
import io

import numpy as np

from .candidates import CandidateSet
from .features import N_FEATURES, N_REGRESSION, featurize, regression_vector
from .select import Selection, greedy_select


@dataclasses.dataclass
class PopulationPrior:
    mean: np.ndarray  # β_0, (d,)
    cov: np.ndarray  # Σ_δ, (d, d)
    noise_var: float  # σ²
    basis: np.ndarray | None = None  # U, (d, r) — the personalization subspace
    basis_var: np.ndarray | None = None  # Λ, (r,)
    floor: float = 0.0  # d
    n_users: int = 0

    @classmethod
    def weak(cls, d: int = N_REGRESSION, sd: float = 1.0, noise_var: float = 0.25) -> "PopulationPrior":
        """No population knowledge: what a per-user-only learner starts from (E1)."""
        return cls(mean=np.zeros(d), cov=np.eye(d) * sd**2, noise_var=noise_var)

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(
            buf, mean=self.mean, cov=self.cov, noise_var=self.noise_var,
            basis=self.basis if self.basis is not None else np.zeros((0, 0)),
            basis_var=self.basis_var if self.basis_var is not None else np.zeros(0),
            floor=self.floor, n_users=self.n_users,
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, b: bytes) -> "PopulationPrior":
        z = np.load(io.BytesIO(b))
        basis = z["basis"] if z["basis"].size else None
        return cls(
            mean=z["mean"], cov=z["cov"], noise_var=float(z["noise_var"]), basis=basis,
            basis_var=z["basis_var"] if basis is not None else None, floor=float(z["floor"]),
            n_users=int(z["n_users"]),
        )


def _low_rank_plus_diag(S: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Probabilistic-PCA projection of a covariance: top-`rank` directions + isotropic floor."""
    S = (S + S.T) / 2
    vals, vecs = np.linalg.eigh(S)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    rest = vals[rank:]
    floor = float(max(np.mean(rest) if rest.size else 0.0, 1e-6))
    lam = np.maximum(vals[:rank] - floor, 0.0)
    U = vecs[:, :rank]
    return U @ np.diag(lam) @ U.T + floor * np.eye(S.shape[0]), U, lam, floor


def fit_population_prior(
    users: list[tuple[np.ndarray, np.ndarray]],
    rank: int = 4,
    iters: int = 25,
    ridge: float = 1e-2,
    max_weight_per_user: int | None = None,
) -> PopulationPrior:
    """EM for the random-effects linear model above, from per-user (X_i, r_i) logs.

    `max_weight_per_user` caps how many of a user's interactions count toward the shared
    parameters, so one user (or one adversary) flooding feedback cannot move β_0 alone.
    """
    users = [(X, r) for X, r in users if len(r) > 0]
    if not users:
        raise ValueError("no training data")
    d = users[0][0].shape[1]
    stats = []
    for X, r in users:
        w = 1.0
        if max_weight_per_user and len(r) > max_weight_per_user:
            w = max_weight_per_user / len(r)
        stats.append((w * X.T @ X, w * X.T @ r, w * float(r @ r), w * len(r)))
    N = sum(s[3] for s in stats)
    A = sum(s[0] for s in stats) + ridge * np.eye(d)
    beta = np.linalg.solve(A, sum(s[1] for s in stats))
    noise = max(sum(rr - 2 * beta @ Xr + beta @ XX @ beta for XX, Xr, rr, _ in stats) / N, 1e-4)
    Sigma = 0.05 * np.eye(d)
    U = lam = None
    floor = 0.05
    for _ in range(iters):
        Sinv = np.linalg.inv(Sigma)
        ms, Vs = [], []
        for XX, Xr, _, _ in stats:
            V = np.linalg.inv(Sinv + XX / noise)
            ms.append(V @ (Xr - XX @ beta) / noise)
            Vs.append(V)
        beta = np.linalg.solve(A, sum(Xr - XX @ m for (XX, Xr, _, _), m in zip(stats, ms)))
        S = sum(np.outer(m, m) + V for m, V in zip(ms, Vs)) / len(stats)
        Sigma, U, lam, floor = _low_rank_plus_diag(S, rank)
        resid = 0.0
        for (XX, Xr, rr, _), m, V in zip(stats, ms, Vs):
            b = beta + m
            resid += rr - 2 * b @ Xr + b @ XX @ b + np.trace(XX @ V)
        noise = max(resid / N, 1e-4)
    return PopulationPrior(mean=beta, cov=Sigma, noise_var=float(noise), basis=U, basis_var=lam,
                           floor=floor, n_users=len(users))


class BayesLinearPolicy:
    """D (frozen=True) or E (learning) depending on how it is used.

    `forgetting` γ < 1 is *stabilised* forgetting: the posterior decays toward the
    population prior rather than toward nothing, so the model drifts with the user without
    its uncertainty blowing up during a quiet spell.
    """

    def __init__(
        self,
        prior: PopulationPrior,
        name: str = "E2-pcim",
        forgetting: float = 1.0,
        ts_scale: float = 1.0,
        frozen: bool = False,
        greedy: bool = False,
        rank: str = "density",
        dedup_cos: float = 0.92,
        cost_per_token: float = 0.0,
        risk_per_token: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    ):
        self.prior = prior
        self.name = name
        self.forgetting = forgetting
        self.ts_scale = ts_scale
        self.frozen = frozen
        # greedy=True: learn from feedback but act on the posterior mean (no Thompson
        # draw). Relies on natural variation in tasks and candidates for exploration.
        self.greedy = greedy
        # "density" ranks by expected value per token — the greedy-knapsack order. "value"
        # ranks by expected value and ignores what a long item crowds out; it is what G0
        # registered, and G0's negative control is what it costs (experiments/g0b).
        if rank not in ("value", "density"):
            raise ValueError("rank must be 'value' or 'density'")
        self.rank = rank
        self.dedup_cos = dedup_cos
        # λ and ρ from the objective (docs/03 §1): an item is included only if its
        # expected quality gain exceeds its token cost plus its privacy exposure.
        self.cost_per_token = cost_per_token
        self.risk_per_token = np.asarray(risk_per_token, dtype=float)
        self.P0 = np.linalg.inv(prior.cov)
        self.b0 = self.P0 @ prior.mean
        self.P = self.P0.copy()
        self.b = self.b0.copy()
        self.n_updates = 0
        self.version = 0

    # -------------------------------------------------------------- posterior
    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.P, self.b)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        mu = self.mean()
        L = np.linalg.cholesky(self.P)
        z = rng.standard_normal(len(mu))
        return mu + self.ts_scale * np.linalg.solve(L.T, z)

    def update(self, x: np.ndarray, r: float) -> None:
        if self.frozen:
            return
        g = self.forgetting
        if g < 1.0:
            self.P = g * self.P + (1 - g) * self.P0
            self.b = g * self.b + (1 - g) * self.b0
        s2 = self.prior.noise_var
        self.P += np.outer(x, x) / s2
        self.b += x * r / s2
        self.n_updates += 1

    def deviation(self) -> np.ndarray:
        """Posterior-mean deviation from the population, δ̂_i."""
        return self.mean() - self.prior.mean

    def latent(self) -> np.ndarray | None:
        """z_i: coordinates of δ̂_i in the learned personalization subspace (standardised)."""
        if self.prior.basis is None:
            return None
        lam = np.maximum(self.prior.basis_var, 1e-9)
        return (self.prior.basis.T @ self.deviation()) / np.sqrt(lam)

    # -------------------------------------------------------------- selection
    def _item_scores(self, beta: np.ndarray, F: np.ndarray, cs: CandidateSet) -> np.ndarray:
        s = F @ beta[:N_FEATURES]
        if self.cost_per_token or self.risk_per_token.any():
            s = s - cs.tokens * (self.cost_per_token + self.risk_per_token[cs.sensitivity])
        return s

    def _choose(self, beta: np.ndarray, F: np.ndarray, cs: CandidateSet, budget: int) -> np.ndarray:
        s = self._item_scores(beta, F, cs)
        order_key = s / np.maximum(cs.tokens, 1) if self.rank == "density" else s
        # threshold 0 on either key means "include only if the expected gain is positive".
        return greedy_select(
            order_key, cs.tokens, budget, emb=cs.emb, threshold=0.0, dedup_cos=self.dedup_cos,
            forced=cs.pinned & ~cs.superseded, eligible=~cs.superseded,
        ), s

    def select(
        self,
        cs: CandidateSet,
        task_probs: np.ndarray,
        budget: int,
        rng: np.random.Generator | None = None,
        propensity_samples: int = 0,
        explore: bool | None = None,
        F: np.ndarray | None = None,
    ) -> Selection:
        rng = rng or np.random.default_rng()
        if F is None:
            F = featurize(cs, task_probs)
        explore = (not self.frozen and not self.greedy) if explore is None else explore
        beta = self.sample(rng) if explore else self.mean()
        idx, s = self._choose(beta, F, cs, budget)
        props = None
        if propensity_samples and len(cs):
            counts = np.zeros(len(cs))
            for _ in range(propensity_samples):
                b = self.sample(rng) if explore else beta
                counts[self._choose(b, F, cs, budget)[0]] += 1
            props = counts / propensity_samples
        x = regression_vector(F, idx, task_probs)
        return Selection(idx=idx, scores=s, propensities=props, x=x, policy=self.name, version=self.version)

    # -------------------------------------------------------------- persistence
    def state_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, P=self.P, b=self.b, n=self.n_updates, version=self.version)
        return buf.getvalue()

    def load_state(self, blob: bytes) -> None:
        z = np.load(io.BytesIO(blob))
        self.P, self.b = z["P"], z["b"]
        self.n_updates, self.version = int(z["n"]), int(z["version"])
