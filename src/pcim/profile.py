"""Layer A — the personal representation, as a document the person can read ("Intelligence DNA").

Three parts, deliberately kept apart because they deserve different amounts of trust:

1. **Stated** — preferences, constraints, goals the person said or confirmed. Symbolic,
   editable, portable as-is to any model. This is most of what makes CI useful.
2. **Inferred** — anything CI derived, always shown with its confidence and decay.
   CI never infers sensitive categories (health, religion, sexuality, politics, finances);
   it only records them if the person states them.
3. **Learned** — what outcome feedback has taught the context policy: which kinds of
   context have helped this person on which kinds of task, *only where the posterior is
   confident* (|mean| > 2 posterior SD). G0's identifiability gate (median correlation
   0.85 with the planted truth in simulation) is the evidence this is not noise there; on
   real data the same check is a G2 deliverable, and until it passes this section is
   labelled experimental.

Plus skill estimates from unassisted probes only (growth.py).
"""

from __future__ import annotations

import numpy as np

from .growth import BKT
from .memory import TemporalMemory, effective_confidence
from .policy.features import FEATURE_NAMES, N_FEATURES
from .policy.learned import BayesLinearPolicy
from .types import KINDS, TASK_TYPES, Trust, utcnow

_STANDING = ("preference", "constraint", "goal")


def _item(i) -> dict:
    return {"id": i.id, "kind": i.kind, "content": i.content, "since": i.valid_from.date().isoformat(),
            "until": i.valid_to.date().isoformat() if i.valid_to else None, "source": i.source,
            "pinned": i.pinned}


def learned_section(pol: BayesLinearPolicy, min_updates: int = 20) -> dict:
    if pol.n_updates < min_updates:
        return {"status": f"not enough feedback yet ({pol.n_updates}/{min_updates})", "findings": []}
    dev = pol.deviation()
    sd = np.sqrt(np.diag(np.linalg.inv(pol.P)))
    findings = []
    for j in range(len(KINDS) * len(TASK_TYPES)):
        if abs(dev[j]) > 2 * sd[j]:
            k, t = KINDS[j // len(TASK_TYPES)], TASK_TYPES[j % len(TASK_TYPES)]
            findings.append({
                "statement": f"On {t} tasks, your {k} notes have been {'more' if dev[j] > 0 else 'less'} "
                             f"useful than average",
                "effect": round(float(dev[j]), 4), "posterior_sd": round(float(sd[j]), 4),
            })
    for j in range(len(KINDS) * len(TASK_TYPES), N_FEATURES):
        if abs(dev[j]) > 2 * sd[j]:
            findings.append({"statement": f"{FEATURE_NAMES[j]} matters {'more' if dev[j] > 0 else 'less'} for you "
                                          f"than average", "effect": round(float(dev[j]), 4),
                             "posterior_sd": round(float(sd[j]), 4)})
    z = pol.latent()
    return {"status": "experimental — identifiability on real data not yet established (docs/04, G2)",
            "n_feedback": pol.n_updates, "findings": findings,
            "latent": [round(float(v), 3) for v in z] if z is not None else None}


def intelligence_dna(memory: TemporalMemory, user_id: str, pol: BayesLinearPolicy,
                     skills: dict[str, dict]) -> dict:
    now = utcnow()
    current = memory.current(user_id, now)
    stated = [i for i in current if i.kind in _STANDING and i.trust.rank >= Trust.USER_STATED.rank]
    inferred = [i for i in current if i.trust is Trust.INFERRED]
    return {
        "format": "pcim-dna/1",
        "generated_at": now.isoformat(),
        "stated": {k: [_item(i) for i in stated if i.kind == k] for k in _STANDING},
        "inferred": [dict(_item(i), confidence=round(effective_confidence(i, now), 3)) for i in inferred],
        "learned": learned_section(pol),
        "skills": {name: {"p_mastery": round(s["p_mastery"], 3), "p_correct": round(BKT(**s).p_correct(), 3),
                          "unassisted_probes": s["n_probes"]} for name, s in skills.items()},
        "counts": {"current_items": len(current), "all_items_including_history": len(memory.all(user_id))},
    }
