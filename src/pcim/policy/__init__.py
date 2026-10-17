"""Layer D — context selection.

Every policy sees the same thing: a `CandidateSet` (struct of arrays over memory items
already filtered by consent) and a task-type distribution, and returns a `Selection`
under a token budget. The production engine and the G0 simulator build `CandidateSet`s
from different sources and then run *this* code, so G0 measures what ships.

    heuristic.py  B1 similarity, B2 strong temporal heuristic, C profile + similarity
    learned.py    D/E: Bayesian linear value model, Thompson sampling, population prior
    neural.py     experimental, torch: shared nonlinear scorer (not used by any gate)
"""

from .candidates import CandidateSet, build_candidates
from .features import N_FEATURES, featurize, regression_vector
from .select import Selection, greedy_select

__all__ = [
    "CandidateSet", "build_candidates", "N_FEATURES", "featurize", "regression_vector",
    "Selection", "greedy_select",
]
