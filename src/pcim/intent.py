"""Layer C — intent and goal modeling.

Two separable pieces:

1. `classify_task` — a distribution over task types from a lexical scorer. Crude by
   design: the context policy only needs p(type) to weight its type-specific values, and
   a soft distribution degrades gracefully where a wrong hard label would not. Swapping
   in an LLM or embedding classifier changes nothing downstream. Status: heuristic.

2. `should_clarify` — the decision rule for asking a question before acting. Ask iff the
   expected value of information exceeds the cost of interrupting:

       VOI = E_s[ max_a U(a, s) ] − max_a E_s[ U(a, s) ]

   over interpretations s with probabilities p(s) and candidate actions a. VOI is zero
   when one action is best under every interpretation — ambiguity that does not change
   what to do is not worth a question. `cost` is per user and learnable: some people
   would rather be asked, some would rather be guessed at. Status: the rule is exact;
   where interpretations and utilities come from (an LLM call) is not built yet.
"""

from __future__ import annotations

import math
import re

from .types import TASK_TYPES

_LEXICON: dict[str, tuple[str, ...]] = {
    "code": ("code", "bug", "function", "python", "error", "test", "refactor", "api", "compile", "stack trace",
             "debug", "script", "repo", "commit", "typescript", "sql", "regex", "class", "deploy"),
    "writing": ("write", "draft", "edit", "email", "essay", "paragraph", "tone", "rewrite", "letter", "post",
                "proofread", "summary", "abstract", "cover letter", "blog", "wording"),
    "research": ("paper", "literature", "evidence", "study", "cite", "citation", "hypothesis", "experiment",
                 "dataset", "analysis", "compare", "survey", "sources", "find out", "investigate"),
    "planning": ("plan", "schedule", "deadline", "itinerary", "roadmap", "week", "calendar", "budget",
                 "prioritize", "trip", "meeting", "organize", "timeline", "book", "dinner", "goal"),
    "learning": ("learn", "explain", "understand", "teach", "quiz", "practice", "concept", "intuition",
                 "study for", "exam", "tutor", "why does", "how does", "lecture", "homework"),
}


def classify_task(text: str, temperature: float = 0.7) -> dict[str, float]:
    t = text.lower()
    scores = {}
    for tt in TASK_TYPES:
        s = 0.0
        for w in _LEXICON[tt]:
            if (" " in w and w in t) or re.search(rf"\b{re.escape(w)}", t):
                s += 1.0
        scores[tt] = s
    z = max(scores.values())
    exps = {k: math.exp((v - z) / temperature) for k, v in scores.items()}
    tot = sum(exps.values())
    return {k: v / tot for k, v in exps.items()}


def entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)


def value_of_information(probs: list[float], utilities: list[list[float]]) -> float:
    """VOI for interpretations s (rows of `utilities`) and actions a (columns).

    utilities[s][a] = utility of action a if the user meant interpretation s.
    """
    if not probs:
        return 0.0
    z = sum(probs)
    p = [x / z for x in probs]
    n_actions = len(utilities[0])
    informed = sum(ps * max(row) for ps, row in zip(p, utilities))
    uninformed = max(sum(ps * row[a] for ps, row in zip(p, utilities)) for a in range(n_actions))
    return informed - uninformed


def should_clarify(probs: list[float], utilities: list[list[float]], cost: float) -> tuple[bool, float]:
    voi = value_of_information(probs, utilities)
    return voi > cost, voi
