# G0b pre-registration — the revised method, on a fresh seed

Written 2026-10-15, after the G0 confirmatory run (`experiments/g0/results/`, verdict
**FAIL**, 4 of 7 gates) and one post-hoc diagnostic, and before any G0b code change or
run. G0b exists because G0 failed in ways that point at two specific, fixable method
defects. It does not replace G0: both results are reported, in order, and this file says
what changed and why.

## What G0 found, and the two changes

1. **The learned policy ignored the budget's shadow price.** In the negative control the
   frozen population policy D lost to tuned B2 by 6.8pp. Diagnostic
   (`experiments/g0/diagnose_negctrl.py`, confirmatory seed, 60 users, post hoc): D took
   8.7 items averaging 71 tokens, B2 12.0 items averaging 46. Ranking D's *own scores* by
   value per token took 13.2 items and scored 0.741 against B2's 0.721 and the oracle's
   0.744. Ranking by value rather than value density is the textbook greedy-knapsack
   error; B2 could partly escape it through a tuned length weight, a per-item regression
   cannot, because the cost of a long item is the items it crowds out, not its own effect
   on reward.
   **Change: every learned policy ranks candidates by score per token** (include while the
   score is positive and the item fits). **B2's tuner gets the same option**: each random
   configuration ranks by density with probability ½, so the baseline can use the fix too.

2. **Thompson sampling's exploration cost exceeded the personalization gain.** E2 was
   below D in 9 of 12 conditions; the exploratory posterior-mean variant E2g beat E2 in all
   12 (by +1.2 to +5.3pp, interactions 101–200).
   **Change: the primary personalization system acts on the posterior mean** (exploration
   comes from natural variation in tasks and candidates). Thompson sampling is kept as the
   secondary system `E2ts`.

Nothing else changes: same world parameters, conditions, systems A–D and E1, analysis,
windows, and **identical gates applied to the primary system E2**. The D2 calibration
(near-duplicate threshold 0.993) and bug fixes carry over. Config: `config.json` here,
generated from G0's with exactly the keys `policy.ranking`, `policy.e2_mode`,
`policy.secondary`, `b2_tuning.allow_density`, `seeds`, `schema`, `experiment` changed.

## Seeds

Confirmatory seed 20261097, chosen before any G0b run, run once. One mechanics check on
dev seed 10 with a single condition, to confirm the code runs; its numbers are not used.

## What would count as what

- All 7 gates pass: the revised method clears G0. That licenses G1, nothing more — G0b was
  designed after seeing G0, so its pass is weaker evidence than a first-try pass would
  have been, and G1 must not reuse any G0/G0b tuning on real data.
- G0-2 passes but personalization gates (G0-3, G0-5a/b) still fail: population learning is
  the product; per-user adaptation is a research problem, not a feature.
- G0-2 still fails: the value model itself is wrong for similarity-dominated worlds;
  stop and redesign features before any further run.

## Deviations

**D1 — before the confirmatory run, caught by the dev-seed-10 mechanics check.** The code
made E1 inherit E2's new posterior-mean mode, contradicting this file ("systems A–D and E1
unchanged"). A posterior-mean learner on a zero prior includes nothing and never learns
(E1 scored exactly the no-context value, 0.432), which would have made G0-4 pass
trivially. Fixed so E1 is Thompson sampling as registered (with density ranking, which
applies to every learned policy). The text of this pre-registration did not change; only
the code was brought into line with it. G0's results reproduce exactly after the refactor
(dev seed 1, drift and transfer, every system's mean identical).
