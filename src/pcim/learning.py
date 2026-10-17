"""Layer H — controlled continual learning.

Per-user learning is a closed-form posterior update (policy/learned.py), so it cannot
"forget" catastrophically: the posterior after n updates is exactly prior + data. What
can go wrong is (a) the data — drifting preferences, manipulated feedback — and (b) the
shared prior, which every user inherits. The controls here:

- **Versioned state, rollback.** Snapshots are immutable versions, written at the first
  update and every `snapshot_every` updates after. Between snapshots the evolving posterior
  lives in a separate working slot, so a snapshot always means exactly what it says.
  `rollback(v)` re-activates snapshot v and discards the working slot. Snapshots and
  rollbacks are audited. (The first version of this module overwrote the active snapshot
  in place; the rollback test caught it restoring update 9 when asked for update 1.)
- **Stabilised forgetting** is available per user (posterior decays toward the prior, not
  toward nothing). G0's drift condition measured it: it helped after a change point
  (+1.2pp) and cost slightly before one, so it is off by default.
- **Influence caps.** `fit_population_prior(max_weight_per_user=…)` bounds how much any one
  user's logs can move the shared prior.
- **Promotion gate for shared updates.** A new population prior is promoted only if a
  paired, user-clustered non-inferiority test against the incumbent passes
  (`noninferiority_gate`), run on interleaved traffic or a frozen regression suite.

Not built: off-policy evaluation from logged propensities (they are logged, see
interactions.propensities). Slate OPE under a budget coupling is its own research item
(docs/03 §5); until it exists, promotion uses interleaved online comparison only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .policy.learned import BayesLinearPolicy, PopulationPrior
from .sovereignty.audit import AuditLog
from .store import Store


@dataclasses.dataclass
class GateResult:
    promote: bool
    mean_diff: float
    lo: float
    hi: float
    margin: float
    n_users: int


def noninferiority_gate(candidate: np.ndarray, incumbent: np.ndarray, margin: float = 0.01,
                        n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> GateResult:
    """Promote iff the lower (1−α) bound of mean(candidate − incumbent) is above −margin.

    Inputs are per-user mean rewards under each policy (paired by user).
    """
    d = np.asarray(candidate, float) - np.asarray(incumbent, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return GateResult(bool(lo > -margin), float(d.mean()), float(lo), float(hi), margin, len(d))


class ContinualLearner:
    def __init__(self, store: Store, audit: AuditLog, prior: PopulationPrior, name: str = "pcim-context",
                 snapshot_every: int = 10, forgetting: float = 1.0, greedy: bool = True):
        self.store, self.audit, self.prior, self.name = store, audit, prior, name
        self.snapshot_every, self.forgetting, self.greedy = snapshot_every, forgetting, greedy
        self._cache: dict[str, BayesLinearPolicy] = {}

    def policy_for(self, user_id: str) -> BayesLinearPolicy:
        pol = self._cache.get(user_id)
        if pol is not None:
            return pol
        # Posterior-mean action and density ranking: the configuration G0b registered after
        # G0 showed Thompson sampling's exploration cost and value ranking's budget blindness.
        pol = BayesLinearPolicy(self.prior, name=self.name, forgetting=self.forgetting, greedy=self.greedy,
                                rank="density")
        snap = self.store.load_policy_state(user_id, self.name)
        if snap:
            pol.load_state(snap[1])
            pol.version = snap[0]
            working = self.store.load_policy_state(user_id, self._working)
            if working:
                pol.load_state(working[1])
                pol.version = snap[0]  # the working state descends from the active snapshot
        self._cache[user_id] = pol
        return pol

    @property
    def _working(self) -> str:
        return self.name + "#working"

    def update(self, user_id: str, x: np.ndarray, reward: float) -> int:
        pol = self.policy_for(user_id)
        pol.update(np.asarray(x, float), float(reward))
        if pol.version == 0 or pol.n_updates % self.snapshot_every == 0:
            pol.version = self.store.save_policy_state(
                user_id, self.name, pol.state_bytes(), note=f"after {pol.n_updates} updates",
                parent=pol.version or None,
            )
            self.store.delete_policy_states(user_id, self._working)
            self.audit.append(user_id, "policy.snapshot", policy=self.name, version=pol.version, n=pol.n_updates)
        elif self.store.load_policy_state(user_id, self._working) is None:
            self.store.save_policy_state(user_id, self._working, pol.state_bytes(), note="working state")
        else:
            self.store.overwrite_policy_state(user_id, self._working, 1, pol.state_bytes())
        return pol.version

    def rollback(self, user_id: str, version: int) -> None:
        self.store.activate_policy_version(user_id, self.name, version)
        self.store.delete_policy_states(user_id, self._working)
        self._cache.pop(user_id, None)
        self.audit.append(user_id, "policy.rollback", policy=self.name, version=version)

    def reset(self, user_id: str) -> None:
        """Forget everything learned about this user's context preferences (keep memories)."""
        self._cache.pop(user_id, None)
        pol = BayesLinearPolicy(self.prior, name=self.name, forgetting=self.forgetting, greedy=self.greedy,
                                rank="density")
        pol.version = self.store.save_policy_state(user_id, self.name, pol.state_bytes(), note="reset to prior")
        self.store.delete_policy_states(user_id, self._working)
        self._cache[user_id] = pol
        self.audit.append(user_id, "policy.reset", policy=self.name, version=pol.version)

    def versions(self, user_id: str) -> list[dict]:
        return self.store.list_policy_versions(user_id, self.name)
