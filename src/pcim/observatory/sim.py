"""G0 synthetic world. Every semantic choice the pre-registration left open is fixed here
and listed in PREREGISTRATION.md → Deviations (D1), decided before any system was run.

Ground truth (per condition, shared across users):
    A[type, kind]   ~ N(0, affinity_sd²) · λ_shared        population affinity
    P_k[type, kind] ~ N(0, user_basis_sd²),  k = 1..r     personal directions
Per user i:
    z_i ~ N(0, I_r) · λ_user,   ε_i[type, kind] ~ N(0, user_idio_sd²) · λ_user
    value_i = value_base + A + Σ_k z_ik P_k + ε_i
Per item m and task t:
    s     = cos(e_m, e_t)
    g     = sigmoid((s − center) / width);   g' = floor_kind + (1 − floor_kind)·g
    u     = g' · value_i[type_t, kind_m]            if the item is current and true
    u     = −stale_harm_M · g'                      if superseded, or silently outdated
    Q(S)  = sigmoid(η_t + Σ_{dup groups in S} max u + Σ_{stale in S} u − γ_M · tokens(S)/1000)
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..policy.candidates import CandidateSet
from ..policy.select import greedy_select
from ..types import KIND_INDEX, KINDS, TASK_TYPES

N_KIND, N_TASK = len(KINDS), len(TASK_TYPES)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@dataclasses.dataclass
class World:
    cfg: dict  # config["world"]
    A: np.ndarray  # (N_TASK, N_KIND)
    P: np.ndarray  # (r, N_TASK, N_KIND)
    lambda_user: float
    similarity_world: bool  # negative control: relevance is similarity alone

    @classmethod
    def create(cls, cfg: dict, lambda_shared: float, lambda_user: float, rng: np.random.Generator,
               similarity_world: bool = False) -> "World":
        A = rng.normal(0, cfg["affinity_sd"], (N_TASK, N_KIND)) * lambda_shared
        P = rng.normal(0, cfg["user_basis_sd"], (cfg["user_rank"], N_TASK, N_KIND))
        return cls(cfg, A, P, lambda_user, similarity_world)

    def _noisy(self, center: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
        # D1: topic_noise is the RMS norm of the isotropic Gaussian noise added to a unit
        # topic centre (per-coordinate SD = topic_noise / sqrt(dim)).
        d = self.cfg["dim"]
        return _unit(center[None, :] + rng.normal(0, self.cfg["topic_noise"] / np.sqrt(d), (n, d)))

    def new_user(self, rng: np.random.Generator) -> "SimUser":
        c = self.cfg
        d, n = c["dim"], c["n_items"]
        topics = _unit(rng.normal(size=(c["n_topics"], d)))
        kinds_p = np.array([c["kind_probs"][k] for k in KINDS])
        kind = rng.choice(N_KIND, size=n, p=kinds_p / kinds_p.sum())
        topic = rng.integers(0, c["n_topics"], size=n)
        emb = np.vstack([self._noisy(topics[t], rng, 1)[0] for t in topic])
        age = rng.exponential(c["age_mean_days"], size=n)
        conf = rng.beta(*c["confidence_beta"], size=n)
        tokens = np.clip(np.round(rng.lognormal(c["tokens_lognorm_mu"], c["tokens_lognorm_sigma"], n)),
                         c["tokens_min"], c["tokens_max"]).astype(int)
        group = np.arange(n)
        superseded = np.zeros(n, dtype=bool)
        # Near-duplicates and stale versions are copies of *original* items. (D2: copies
        # were first drawn from any item, so a source could later be overwritten by its
        # own copy step and a "duplicate" group would hold unrelated items.)
        roles = rng.random(n)
        is_dup = roles < c["dup_prob"]
        is_stale = ~is_dup & (roles < c["dup_prob"] + c["stale_prob"])
        originals = np.flatnonzero(~(is_dup | is_stale))
        copy_of = np.arange(n)
        for j in np.flatnonzero(is_dup | is_stale):
            src = int(rng.choice(originals))
            copy_of[j] = src
            kind[j] = kind[src]
            if is_dup[j]:
                emb[j], group[j] = _unit(emb[src] + rng.normal(0, 0.01, d)), src
            else:
                emb[j] = _unit(emb[src] + rng.normal(0, 0.02, d))
                age[j] = age[src] + rng.exponential(c["age_mean_days"])
                superseded[j] = True
        standing_pg = np.isin(kind, [KIND_INDEX["preference"], KIND_INDEX["goal"]])
        silent = (standing_pg & (age > c["silent_stale_age_days"]) & ~superseded
                  & (rng.random(n) < c["silent_stale_prob"]))
        z = rng.normal(size=self.cfg["user_rank"]) * self.lambda_user
        eps = rng.normal(0, c["user_idio_sd"], (N_TASK, N_KIND)) * self.lambda_user
        mix = rng.dirichlet(np.full(N_TASK, c["task_type_dirichlet"]))
        if self.similarity_world:
            # No staleness in the negative control: what would have been a stale copy is
            # just another copy of the same content, i.e. a duplicate (D2.6).
            group[is_stale] = copy_of[is_stale]
            superseded[:] = False
            silent[:] = False
        return SimUser(self, topics, emb, kind, age, conf, tokens, group, superseded, silent, z, eps, mix)


@dataclasses.dataclass
class SimTask:
    emb: np.ndarray
    type_idx: int
    difficulty: float
    novel: bool

    @property
    def probs(self) -> np.ndarray:
        p = np.zeros(N_TASK)
        p[self.type_idx] = 1.0
        return p


@dataclasses.dataclass
class SimUser:
    world: World
    topics: np.ndarray
    emb: np.ndarray
    kind: np.ndarray
    age: np.ndarray
    conf: np.ndarray
    tokens: np.ndarray
    group: np.ndarray
    superseded: np.ndarray
    silent: np.ndarray
    z: np.ndarray
    eps: np.ndarray
    mix: np.ndarray

    @property
    def deviation(self) -> np.ndarray:
        """True personal deviation Δ_i[type, kind], (N_TASK, N_KIND)."""
        return np.tensordot(self.z, self.world.P, axes=1) + self.eps

    @property
    def value(self) -> np.ndarray:
        w = self.world
        if w.similarity_world:
            return np.full((N_TASK, N_KIND), w.cfg["value_base"])
        return w.cfg["value_base"] + w.A + self.deviation

    def redraw_z(self, rng: np.random.Generator) -> None:
        self.z = rng.normal(size=self.z.shape) * self.world.lambda_user

    def new_task(self, rng: np.random.Generator) -> SimTask:
        c = self.world.cfg
        novel = bool(rng.random() < c["novel_topic_prob"])
        center = _unit(rng.normal(size=c["dim"])) if novel else self.topics[int(rng.integers(0, len(self.topics)))]
        emb = self.world._noisy(center, rng, 1)[0]
        t = int(rng.choice(N_TASK, p=self.mix))
        eta = float(rng.normal(c["task_difficulty_mean"], c["task_difficulty_sd"]))
        return SimTask(emb, t, eta, novel)

    # ------------------------------------------------------------------ truth
    def relevance(self, task: SimTask) -> np.ndarray:
        c = self.world.cfg
        s = self.emb @ task.emb
        g = _sigmoid((s - c["relevance_center"]) / c["relevance_width"])
        if not self.world.similarity_world:
            for k, floor in c["global_relevance_floor"].items():
                m = self.kind == KIND_INDEX[k]
                g[m] = floor + (1 - floor) * g[m]
        return g

    def utilities(self, task: SimTask, model: dict) -> np.ndarray:
        g = self.relevance(task)
        u = g * self.value[task.type_idx, self.kind]
        bad = self.superseded | self.silent
        u[bad] = -model["stale_harm"] * g[bad]
        return u

    def quality(self, task: SimTask, idx: np.ndarray, util: np.ndarray, model: dict) -> float:
        logit = task.difficulty - model["length_penalty_per_1k"] * self.tokens[idx].sum() / 1000.0
        if len(idx):
            stale = self.superseded[idx] | self.silent[idx]
            logit += util[idx][stale].sum()
            good = idx[~stale]
            if len(good):
                best: dict[int, float] = {}
                for j in good:
                    g = int(self.group[j])
                    best[g] = max(best.get(g, -np.inf), float(util[j]))
                logit += sum(best.values())
        return float(_sigmoid(logit))

    # ------------------------------------------------------------------ views
    def candidates(self, task: SimTask, cand_cfg: dict) -> tuple[np.ndarray, CandidateSet]:
        """Same three channels as production build_candidates; superseded items can enter
        through similarity (so B1 can be fooled) and are flagged for everyone else."""
        s = self.emb @ task.emb
        chosen = set(np.argsort(-s)[: cand_cfg["top_sim"]].tolist())
        standing = np.flatnonzero(np.isin(self.kind, [KIND_INDEX[k] for k in ("preference", "constraint", "goal")])
                                  & ~self.superseded)
        chosen.update(standing[np.argsort(self.age[standing])][: cand_cfg["standing_max"]].tolist())
        eps_ = np.flatnonzero((self.kind == KIND_INDEX["episode"]) & ~self.superseded)
        chosen.update(eps_[np.argsort(self.age[eps_])][: cand_cfg["recent_episodes"]].tolist())
        idx = np.array(sorted(chosen), dtype=int)
        n = len(idx)
        cs = CandidateSet(
            ids=[str(j) for j in idx], sim=s[idx], kind=self.kind[idx], age_days=self.age[idx],
            confidence=self.conf[idx], tokens=self.tokens[idx], superseded=self.superseded[idx],
            pinned=np.zeros(n, dtype=bool), sensitivity=np.zeros(n, dtype=int), emb=self.emb[idx],
        )
        return idx, cs


def oracle_select(user: SimUser, task: SimTask, idx: np.ndarray, util: np.ndarray, model: dict,
                  budget: int) -> np.ndarray:
    """Exact optimum over current candidates.

    Q is a sigmoid of a logit that is additive across duplicate groups, so maximising Q is
    a *multiple-choice* knapsack on gains u_j − γ·tokens_j/1000: at most one member per
    duplicate group (a second member adds tokens, not value). Stale and silently-outdated
    items have negative gain and are never worth taking. Solved exactly by dynamic
    programming over the token budget. Keeping only each group's best-gain member is NOT
    exact — a shorter duplicate with nearly the same gain can free budget — which is how
    the brute-force test caught the first version.
    """
    gamma = model["length_penalty_per_1k"] / 1000.0
    gains = util[idx] - gamma * user.tokens[idx]
    groups: dict[int, list[int]] = {}
    for j in range(len(idx)):
        m = idx[j]
        if user.superseded[m] or user.silent[m] or gains[j] <= 0 or user.tokens[m] > budget:
            continue
        groups.setdefault(int(user.group[m]), []).append(j)
    if not groups:
        return np.zeros(0, dtype=int)
    dp = np.zeros(budget + 1)
    choice = np.full((len(groups), budget + 1), -1, dtype=int)
    for gi, members in enumerate(groups.values()):
        new, ch = dp.copy(), np.full(budget + 1, -1, dtype=int)
        for j in members:
            w = int(user.tokens[idx[j]])
            cand = np.full(budget + 1, -np.inf)
            cand[w:] = dp[: budget + 1 - w] + gains[j]
            better = cand > new
            new = np.where(better, cand, new)
            ch = np.where(better, j, ch)
        choice[gi], dp = ch, new
    chosen, c = [], budget
    for gi in range(len(groups) - 1, -1, -1):
        j = choice[gi, c]
        if j >= 0:
            chosen.append(int(j))
            c -= int(user.tokens[idx[j]])
    return np.array(chosen[::-1], dtype=int)


def random_select(cs: CandidateSet, budget: int, p_include: float, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(len(cs))
    mask = (rng.random(len(cs)) < p_include) & ~cs.superseded
    picked, used = [], 0
    for j in order:
        if mask[j] and used + cs.tokens[j] <= budget:
            picked.append(int(j))
            used += int(cs.tokens[j])
    return np.array(picked, dtype=int)


def feedback(q: float, kind: str, cfg: dict, u: float, u2: float, noise: float) -> float | None:
    """Observed reward. `u`, `u2`, `noise` are shared across systems (common random numbers)."""
    if kind == "dense":
        return float(q + cfg["dense_sd"] * noise)
    if kind == "binary":
        return float(u < q)
    if kind == "sparse":
        return float(u < q) if u2 < cfg["sparse_rate"] else None
    raise ValueError(kind)


__all__ = ["World", "SimUser", "SimTask", "oracle_select", "random_select", "feedback", "greedy_select"]
