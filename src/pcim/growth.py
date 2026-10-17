"""Cognitive growth engine — measuring skill without measuring dependence.

Bayesian Knowledge Tracing (Corbett & Anderson, 1994) per skill, with one rule that
matters more than the model: **only unassisted probes update mastery.** An assisted
success is evidence the assistance worked, not that the person learned; counting it would
make the H term in the objective reward exactly the dependency it is meant to prevent.

`next_challenge` picks the skill whose predicted success is closest to 0.85 — the
"eighty-five percent rule" (Wilson et al., Nat. Commun. 2019) used as a default for
desirable difficulty, not as a finding about this person.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class BKT:
    p_mastery: float = 0.2
    p_learn: float = 0.15
    p_slip: float = 0.10
    p_guess: float = 0.20
    n_probes: int = 0

    def p_correct(self) -> float:
        return self.p_mastery * (1 - self.p_slip) + (1 - self.p_mastery) * self.p_guess

    def observe(self, correct: bool) -> float:
        m, s, g = self.p_mastery, self.p_slip, self.p_guess
        if correct:
            post = m * (1 - s) / (m * (1 - s) + (1 - m) * g)
        else:
            post = m * s / (m * s + (1 - m) * (1 - g))
        self.p_mastery = post + (1 - post) * self.p_learn
        self.n_probes += 1
        return self.p_mastery

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def next_challenge(skills: dict[str, BKT], target: float = 0.85) -> str | None:
    if not skills:
        return None
    return min(skills, key=lambda k: abs(skills[k].p_correct() - target))
