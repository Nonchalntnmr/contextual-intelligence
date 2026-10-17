"""Correctness of the pieces G0's verdict rests on. Each test plants a known answer."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from pcim.observatory.sim import World, oracle_select
from pcim.policy.candidates import CandidateSet
from pcim.policy.features import FEATURE_NAMES, N_FEATURES, N_REGRESSION, featurize
from pcim.policy.heuristic import HeuristicPolicy, SimilarityPolicy
from pcim.policy.learned import BayesLinearPolicy, PopulationPrior, fit_population_prior
from pcim.policy.select import greedy_select
from pcim.types import KIND_INDEX

CFG = json.loads((Path(__file__).parents[1] / "experiments/g0/config.json").read_text())


def _cs(n, rng, kinds=None):
    emb = rng.normal(size=(n, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return CandidateSet(
        ids=[str(i) for i in range(n)], sim=rng.uniform(0, 1, n),
        kind=np.array(kinds if kinds is not None else rng.integers(0, 6, n)),
        age_days=rng.uniform(0, 400, n), confidence=rng.uniform(0.5, 1, n),
        tokens=rng.integers(20, 80, n), superseded=np.zeros(n, bool), pinned=np.zeros(n, bool),
        sensitivity=np.zeros(n, int), emb=emb,
    )


# ---------------------------------------------------------------- selection
def test_greedy_select_respects_budget_threshold_and_order():
    scores = np.array([0.9, 0.8, 0.1, -0.2, 0.5])
    tokens = np.array([50, 60, 10, 10, 200])
    idx = greedy_select(scores, tokens, budget=120, threshold=0.0)
    assert list(idx) == [0, 1, 2]  # 0.5-scored item does not fit; negative one never taken
    assert tokens[idx].sum() <= 120


def test_greedy_select_dedup_forced_and_eligible():
    emb = np.eye(4)
    emb[1] = emb[0]  # item 1 duplicates item 0
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    tokens = np.full(4, 10)
    forced = np.array([False, False, False, True])
    eligible = np.array([True, True, False, True])
    idx = greedy_select(scores, tokens, 100, emb=emb, dedup_cos=0.99, forced=forced, eligible=eligible)
    assert list(idx) == [3, 0]  # forced first; duplicate skipped; ineligible never taken


# ---------------------------------------------------------------- features
def test_feature_layout_and_soft_task_types():
    rng = np.random.default_rng(0)
    cs = _cs(3, rng, kinds=[KIND_INDEX["constraint"], KIND_INDEX["episode"], KIND_INDEX["fact"]])
    cs.sim[:] = [0.8, 0.8, 0.1]
    p = np.array([0.0, 0.0, 0.0, 0.75, 0.25])  # planning 0.75, learning 0.25
    F = featurize(cs, p)
    assert F.shape == (3, N_FEATURES) and N_FEATURES == len(FEATURE_NAMES) == 39
    k = KIND_INDEX["constraint"]
    np.testing.assert_allclose(F[0, k * 5:(k + 1) * 5], p * 1.0)  # ramp(0.8) = 1
    assert F[2, :30].sum() == 0  # similarity 0.1 is below the ramp: no value cell
    assert F[0, FEATURE_NAMES.index("constraint_offtopic")] == 0.0
    assert F[:, FEATURE_NAMES.index("bias")].tolist() == [1, 1, 1]


# ---------------------------------------------------------------- learning
def test_bayes_linear_recovers_planted_coefficients():
    rng = np.random.default_rng(1)
    d = N_REGRESSION
    beta = rng.normal(0, 0.3, d)
    pol = BayesLinearPolicy(PopulationPrior.weak(d=d, sd=1.0, noise_var=0.05))
    for _ in range(3000):
        x = rng.normal(0, 1, d)
        pol.update(x, float(x @ beta + rng.normal(0, np.sqrt(0.05))))
    err = np.abs(pol.mean() - beta).max()
    assert err < 0.03, err


def test_forgetting_tracks_a_change_and_frozen_never_moves():
    rng = np.random.default_rng(2)
    d = N_REGRESSION
    b1, b2 = rng.normal(0, 0.5, d), rng.normal(0, 0.5, d)
    prior = PopulationPrior.weak(d=d, sd=1.0, noise_var=0.05)
    slow, fast = BayesLinearPolicy(prior), BayesLinearPolicy(prior, forgetting=0.97)
    frozen = BayesLinearPolicy(prior, frozen=True)
    for t in range(2000):
        x = rng.normal(0, 1, d)
        b = b1 if t < 1000 else b2
        r = float(x @ b + rng.normal(0, 0.2))
        for p in (slow, fast, frozen):
            p.update(x, r)
    assert np.abs(fast.mean() - b2).mean() < np.abs(slow.mean() - b2).mean()
    np.testing.assert_allclose(frozen.mean(), prior.mean)


def test_em_recovers_the_personalization_subspace():
    rng = np.random.default_rng(3)
    d, r = 12, 2
    U = np.linalg.qr(rng.normal(size=(d, r)))[0]
    beta0 = rng.normal(0, 0.3, d)
    users = []
    for _ in range(300):
        delta = U @ rng.normal(0, 0.6, r)
        X = rng.normal(0, 1, (60, d))
        users.append((X, X @ (beta0 + delta) + rng.normal(0, 0.3, 60)))
    prior = fit_population_prior(users, rank=r, iters=30)
    # principal angles between the true and learned subspaces
    s = np.linalg.svd(U.T @ prior.basis, compute_uv=False)
    assert s.min() > 0.95, s
    assert np.abs(prior.mean - beta0).max() < 0.1
    assert abs(prior.noise_var - 0.09) < 0.02


def test_population_weight_cap_limits_a_flooding_user():
    rng = np.random.default_rng(4)
    d = 6
    honest = [(X := rng.normal(size=(40, d)), X @ np.ones(d)) for _ in range(30)]
    Xa = rng.normal(size=(20000, d))
    attacker = (Xa, Xa @ (-5 * np.ones(d)))
    capped = fit_population_prior(honest + [attacker], rank=1, iters=5, max_weight_per_user=40)
    uncapped = fit_population_prior(honest + [attacker], rank=1, iters=5)
    assert np.abs(capped.mean - 1).max() < np.abs(uncapped.mean - 1).max()
    assert capped.mean.mean() > 0.5  # still pointing the honest way


def test_thompson_propensities_are_probabilities():
    rng = np.random.default_rng(5)
    cs = _cs(20, rng)
    pol = BayesLinearPolicy(PopulationPrior.weak(sd=0.3))
    sel = pol.select(cs, np.full(5, 0.2), budget=200, rng=rng, propensity_samples=64)
    assert sel.propensities.shape == (20,)
    assert ((sel.propensities >= 0) & (sel.propensities <= 1)).all()
    assert cs.tokens[sel.idx].sum() <= 200
    assert sel.x.shape == (N_REGRESSION,)


# ---------------------------------------------------------------- simulator
@pytest.fixture(scope="module")
def world():
    return World.create(CFG["world"], 1.0, 1.0, np.random.default_rng(7))


def test_sim_duplicates_are_duplicates_and_stale_is_flagged(world):
    rng = np.random.default_rng(8)
    for _ in range(10):
        u = world.new_user(rng)
        for g in np.unique(u.group):
            members = np.flatnonzero(u.group == g)
            if len(members) > 1:
                C = u.emb[members] @ u.emb[members].T
                assert C.min() > 0.99
        assert not (u.superseded & u.silent).any()


def test_oracle_is_exact_against_brute_force(world):
    rng = np.random.default_rng(9)
    model = CFG["models"]["M1"]
    checked = 0
    while checked < 25:
        u = world.new_user(rng)
        task = u.new_task(rng)
        idx, cs = u.candidates(task, {"top_sim": 7, "standing_max": 3, "recent_episodes": 2})
        if len(idx) > 13:
            continue
        util = u.utilities(task, model)
        budget = 180
        best = max(
            (u.quality(task, idx[list(S)], util, model)
             for k in range(len(idx) + 1) for S in itertools.combinations(range(len(idx)), k)
             if u.tokens[idx[list(S)]].sum() <= budget and not u.superseded[idx[list(S)]].any()),
        )
        sel = oracle_select(u, task, idx, util, model, budget)
        assert u.tokens[idx[sel]].sum() <= budget
        assert u.quality(task, idx[sel], util, model) == pytest.approx(best, abs=1e-9)
        checked += 1


def test_superseded_items_reach_only_the_naive_baseline(world):
    rng = np.random.default_rng(10)
    hits_b1 = hits_b2 = 0
    for _ in range(30):
        u = world.new_user(rng)
        task = u.new_task(rng)
        idx, cs = u.candidates(task, CFG["candidates"])
        hits_b1 += cs.superseded[SimilarityPolicy().select(cs, task.probs, 600).idx].sum()
        hits_b2 += cs.superseded[HeuristicPolicy().select(cs, task.probs, 600).idx].sum()
        pol = BayesLinearPolicy(PopulationPrior.weak(sd=0.5))
        assert not cs.superseded[pol.select(cs, task.probs, 600, rng).idx].any()
    assert hits_b2 == 0 and hits_b1 > 0
