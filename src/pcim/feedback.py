"""Layer G (signals) — turning what a person did into a reward the policy may learn from.

The reward is where personalization goes wrong. Three findings shape this function:

- RL on user feedback reliably learns to manipulate, and targets the vulnerable minority
  even when it is 2% of users (Williams et al., ICLR 2025).
- Memory profiles are the largest single driver of agreement sycophancy (Jain et al.,
  CHI 2026), and sycophancy is preferred by users while it harms their judgment (Cheng et
  al., Science 2026).
- So "the person liked it" is the weakest signal and the most dangerous one to optimise.

Hence the ordering: a *verifiable* outcome (tests passed, the answer checked out, the
plan worked) dominates whenever it exists; corrections and re-asks — evidence that the
response failed — are weighted heavily; ratings and thumbs are used only when nothing
better exists, and never raise a reward above what a verified failure says. The weights
are fixed and printed, not learned: a learned reward model on top of a learned policy is
two optimisers colluding against one person's attention.
"""

from __future__ import annotations

WEIGHTS = {"corrected": 0.35, "reasked": 0.20, "edit_ratio": 0.40}


def reward_from_signals(signals: dict) -> float | None:
    """Map feedback signals to a reward in [0, 1], or None if there is nothing to learn from.

    signals:
      task_success: bool        verifiable outcome — overrides everything below
      rating: int 1..5          explicit rating
      thumbs: +1 | -1           explicit thumbs
      corrected: bool           the person had to correct the response
      reasked: bool             the person asked again shortly after
      edit_ratio: float 0..1    fraction of the response the person changed before using it
    """
    if "task_success" in signals and signals["task_success"] is not None:
        r = 1.0 if signals["task_success"] else 0.0
        # A verified success can still have needed correction on the way.
        if signals.get("corrected"):
            r -= WEIGHTS["corrected"] / 2
        return max(0.0, min(1.0, r))
    if "rating" in signals and signals["rating"] is not None:
        r = (float(signals["rating"]) - 1.0) / 4.0
    elif "thumbs" in signals and signals["thumbs"] is not None:
        r = 1.0 if signals["thumbs"] > 0 else 0.0
    elif any(k in signals for k in WEIGHTS):
        r = 1.0  # only failure evidence was given; start from "fine" and subtract it
    else:
        return None
    if signals.get("corrected"):
        r -= WEIGHTS["corrected"]
    if signals.get("reasked"):
        r -= WEIGHTS["reasked"]
    if signals.get("edit_ratio") is not None:
        r -= WEIGHTS["edit_ratio"] * float(signals["edit_ratio"])
    return max(0.0, min(1.0, r))
