"""The designed cold-start prior: what Personal CI believes before any population exists.

G0 measured what is at stake here: in simulation, a population-fitted prior was worth
about +15pp of response quality over a weak prior in the first 50 interactions (E2 − E1).
A single-user deployment has no population, so it starts from this hand-set prior, which
encodes the B2 heuristic's beliefs — relevant items help a little, standing preferences
and constraints help even off-topic, old preferences may be stale, long items cost —
with enough variance for the person's own feedback to move it.

This is an engineering default, not a learned quantity, and it is replaced by
`fit_population_prior` as soon as there are enough consenting users to fit one (with
`max_weight_per_user` set, and the promotion gate in learning.py).
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES, N_REGRESSION
from .learned import PopulationPrior

_MEAN = {
    "constraint_offtopic": 0.03,
    "preference_offtopic": 0.02,
    "recency": 0.02,
    "standing_age_years": -0.02,
    "confidence": 0.02,
    "tokens_per_100": -0.01,
    "similarity": 0.05,
    "similarity_above_0.5": 0.05,
    "bias": -0.02,
}


def designed_prior(value_cell: float = 0.05, sd: float = 0.05, intercept: float = 0.6,
                   intercept_sd: float = 0.25, noise_var: float = 0.1) -> PopulationPrior:
    mean = np.zeros(N_REGRESSION)
    for i, name in enumerate(FEATURE_NAMES):
        mean[i] = _MEAN.get(name, value_cell if name.startswith("value[") else 0.0)
    mean[N_FEATURES:] = intercept
    var = np.full(N_REGRESSION, sd**2)
    var[N_FEATURES:] = intercept_sd**2
    return PopulationPrior(mean=mean, cov=np.diag(var), noise_var=noise_var)
