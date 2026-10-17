# G0 pre-registration — learned context selection on synthetic users

Written 2026-10-15, **before** the simulator, the policies, or any run existed. The world
parameters are frozen in `config.json` next to this file; `g0.py` refuses to run a
confirmatory pass if either file's SHA-256 differs from the values it records in the
result. Anything changed after this point is listed under *Deviations* at the bottom,
with the reason, in the order it happened.

## What G0 is and is not

G0 runs the shipped policy code against synthetic users whose ground truth is known. It
answers one question: **if** a user's world contains context-relevance structure that
similarity retrieval cannot see, can PCIM's learned policy find it from realistic,
noisy, sparse feedback in a number of interactions a product could survive?

G0 cannot show that real users have that structure, or that real LLMs reward context
the way this simulator does. Both are G1 questions. A G0 pass is necessary and nowhere
near sufficient. A G0 failure is close to decisive, because the structure was planted
on purpose and every system got the same candidates, budget, and tasks.

## World (summary; exact values in config.json)

- Each user has 120 memory items across 6 kinds (fact, preference, goal, constraint,
  episode, skill), 6 topic clusters in a 16-d semantic space, ages, confidences, token
  lengths, near-duplicate pairs, and stale versions of current items.
- A task has a semantic vector, a type (code, writing, research, planning, learning) and
  a difficulty. 20% of tasks come from topics the user has never had a task on.
- The true utility of an item is `relevance(similarity, kind) × value(task type, kind,
  user)`, minus a harm term for stale items. `value` = base + population affinity
  `A[type, kind]` (strength `lambda_shared`) + a rank-2 user deviation
  `Σ_k z_ik P_k[type, kind]` (strength `lambda_user`) + a small idiosyncratic term.
  Constraints and preferences keep part of their relevance on dissimilar tasks.
- Response quality `Q = sigmoid(difficulty + Σ utilities − γ_M · tokens/1000)`, with
  duplicates counted once. Model M1 has a small length penalty; M2 a large one and
  more sensitivity to stale items.
- Feedback: **dense** `Q + N(0, 0.15²)`; **binary** `Bernoulli(Q)`; **sparse** binary
  observed with probability 0.3, otherwise nothing.

## Systems (all get the same candidate pool, the same 600-token budget, the same tasks)

| id | system | what it can use |
|---|---|---|
| A | no context | — |
| B1 | naive vector memory | similarity top-k until the budget is full; ignores temporal status |
| B2 | strong temporal heuristic memory | similarity + recency + confidence + kind bonus, superseded items excluded, near-duplicate suppression, inclusion threshold. **Its 9 weights and threshold are tuned on the training users against noiseless true quality** — more information than any learned system gets |
| C | static profile + retrieval (native-memory proxy) | top-5 current preferences/constraints always on, similarity for the rest |
| D | learned population policy, frozen | linear value model fit on training users' logged binary/dense/sparse feedback |
| E1 | per-user learning, no population prior | Bayesian linear Thompson sampling from a weak prior |
| E2 | **PCIM: population prior + per-user Bayesian adaptation** | EM-fit low-rank-plus-diagonal prior over user deviations, Thompson sampling, posterior updates from the user's own feedback |
| O | greedy oracle | true utilities; approximate upper bound on what any selection can achieve |

Training: 400 users × 200 interactions per condition, logged under an exploratory
policy (B2 with random perturbation). Test: 200 *different* users × 200 interactions.

## Conditions

| condition | lambda_shared | lambda_user | feedback | other |
|---|---|---|---|---|
| main-u0-{dense,binary,sparse} | 1.0 | 0.0 | each | |
| main-u05-{dense,binary,sparse} | 1.0 | 0.5 | each | |
| main-u1-{dense,binary,sparse} | 1.0 | 1.0 | each | |
| negctrl | 0 | 0 | binary | no stale items, no kind effects: relevance is similarity alone |
| drift | 1.0 | 1.0 | binary | every test user's `z_i` is redrawn at interaction 100 |
| transfer | 1.0 | 1.0 | binary | prior fit under model M1, test users served by M2 |

Model M1 everywhere except `transfer`.

## Outcome and analysis

Primary outcome: the noiseless expected quality `Q` of the context each system chose
(the simulator knows it; the policies never see it). For each user and window, average
`Q`; compare systems **paired within user** (common random numbers: the same tasks and
candidates for every system). 95% percentile bootstrap over users, 2,000 resamples.
Windows: interactions 1–25, 26–50, 51–100, 101–200.

Capture fraction `CF = (Q_E2 − Q_B2) / (Q_O − Q_B2)`. Personalization onset `n*`: the
end of the first 25-interaction window whose `E2 − D` lower 95% bound is above zero.

## Gates (all must pass for "G0 PASS")

| gate | condition | criterion |
|---|---|---|
| G0-1 positive control | main-u1-binary | `E2 − B2` on 101–200: lower 95% bound > 0, and CF ≥ 0.25 |
| G0-2 negative control | negctrl | `E2 − B2` on 1–200: lower 95% bound ≥ −0.01 (non-inferiority, 1pp margin) |
| G0-3 sample efficiency | main-u1-sparse | `n* ≤ 100` interactions (≈ 30 labels) |
| G0-4 population prior earns its place | main-u1-binary | `E2 − E1` on 1–50: lower 95% bound > 0 |
| G0-5a personalization exists when it should | main-u1-binary | `E2 − D` on 101–200: lower 95% bound > 0 |
| G0-5b and costs nothing when it should not | main-u0-binary | `E2 − D` on 101–200: lower 95% bound ≥ −0.01 |
| G0-6 identifiability | main-u1-dense | median over users of corr(E2 posterior-mean deviation, true deviation) across the 30 type×kind cells ≥ 0.5 at t = 200 |

Reported without a gate: drift (does stabilized forgetting recover after the change
point), transfer (gap between M1-fit prior and M2), dense/binary/sparse learning curves,
tokens sent per interaction, and B1 vs B2 (what temporal memory alone is worth).

## Decision rule

- G0-1 or G0-2 fails: stop. The learned-selection method does not work even where it
  was handed the answer; do not spend on G1b.
- G0-3 fails while G0-1 passes: per-user learning is too slow for sparse explicit
  feedback. G1 must test implicit signals, or ship population-level learning (D) only.
- G0-5a fails: the "personal" in personal intelligence model is not earned by this
  method; the defensible product is a population-learned retriever.
- G0-6 fails: the per-user posterior must not be shown to the user as a description of
  them ("Intelligence DNA" latent section is withheld) even if selection improves.

## Seeds

Development and debugging: seeds 0–9, small configurations only. Confirmatory: seed
20261015, run once. If a bug forces a rerun, both results are kept and the rerun is
listed under Deviations.

## Deviations

**D1 — 2026-10-15, before any system was run. Semantics the frozen text left open.**
Writing the simulator exposed choices the pre-registration did not pin down. They were
fixed as below using only world statistics (similarity distributions), never system
outcomes; `experiments/g0/world_check.py` output is quoted.

1. `topic_noise` is the RMS norm of the isotropic Gaussian noise added to a unit topic
   centre (per-coordinate SD = topic_noise/√dim). The alternative reading, per-coordinate
   SD = topic_noise, gives same-topic cosine 0.34 ± 0.21 against cross-topic 0.01 ± 0.25:
   no clusters, contradicting "6 topic clusters". The chosen reading gives 0.90 ± 0.04
   against −0.01 ± 0.26 (relevance gate 0.98 vs 0.05). Consequence worth stating:
   similarity identifies the *topic* almost perfectly and says almost nothing about which
   items *within* a topic matter. That is the world G0 tests.
2. Tasks are generated like items (topic centre + the same noise); novel-topic tasks use
   a fresh random centre.
3. `A ~ N(0, affinity_sd²)·λ_shared`, `P_k ~ N(0, user_basis_sd²)`, `z_i ~ N(0, I)·λ_user`,
   idiosyncratic `ε_i ~ N(0, user_idio_sd²)·λ_user`.
4. A stale item's utility is replaced by `−stale_harm·g'`; silently outdated
   preferences/goals (older than `silent_stale_age_days`, with `silent_stale_prob`) are
   treated the same way but carry no flag.
5. Near-duplicates (`dup_prob`) share a group and count once (the best member).
   Stale copies (`stale_prob`) are re-drawn copies of a current item, older, flagged
   superseded, with near-identical embeddings.
6. The oracle is an exact 0/1 knapsack on the logit, not greedy — an exact upper bound
   under the simulator's additivity.
7. Logging policy: with probability `random_prob` an interaction uses a random subset
   (each current candidate included with `random_include_prob`, budget-truncated in
   random order); otherwise tuned B2.
8. B2 tuning evaluates the default parameters plus `n_configs` random draws on
   `n_users` × `n_tasks` fresh training cases and keeps the best by mean true Q.
9. Feedback uses common random numbers across systems (one uniform and one normal draw per
   user-interaction); Thompson sampling draws use per-system streams.
10. The G0-4 window (1–50) is added to the analysis windows; it is not in the primary
    window list but is the window the gate names.

**D2 — 2026-10-15, after one dev run (seed 0, reduced scale), before the confirmatory run.**
The dev run produced an anomaly — naive similarity (B1) beat the tuned heuristic (B2) in
the negative control — and chasing it found one simulator bug and two design problems.
Each fix below was chosen from world statistics or design reasoning, not from which
system it helps; each applies to baselines and learned policies alike.

1. *Bug.* Near-duplicate and stale copies were drawn from any item, so a source could
   later be overwritten by its own copy step: "duplicate" groups held unrelated items
   (measured pair cosine as low as −0.53) and the count-once rule zeroed genuinely
   distinct items. Copies are now drawn from original items only. After the fix,
   duplicate pairs have cosine ≥ 0.9977.
2. *Calibration.* `policy.dedup_cos` 0.92 → **0.993**. Under the D1 geometry, distinct
   same-topic items have cosine 0.889 on average, and 26.8% of their pairs exceed 0.92,
   so the frozen threshold suppressed legitimate items for B2, D, E1 and E2 (not for B1,
   C or the oracle). 0.993 is the midpoint between the largest distinct same-topic cosine
   (0.9876) and the smallest duplicate cosine (0.9977) — `experiments/g0/dedup_check.py`,
   dev seed 0. B2 now uses the same threshold as the learned policies (it previously used
   a hard-coded 0.92).
3. *Design.* Every condition now shares one seed sequence (common random numbers across
   conditions): the same raw world, users, tasks and noise draws, scaled by λ. With
   per-condition seeds, e.g. dense-vs-binary comparisons were confounded by different
   random worlds. Within-condition comparisons — all of the gates — were already paired.
4. *Exploratory system, added.* The dev run showed E2 below D in most conditions, with the
   gap largest early, consistent with Thompson sampling's exploration cost in 44
   dimensions. **E2g** (same prior and updates as E2, acting on the posterior mean) is added
   as an exploratory comparison. It replaces nothing: every gate is evaluated on E2 exactly
   as registered, and any claim about E2g needs its own pre-registered run.

5. *Bug, found by the brute-force unit test.* D1.6's oracle kept only each duplicate
   group's best-gain member, which is not optimal under a token budget (a shorter
   duplicate with nearly equal gain can free room for another item): one instance scored
   0.9148 against a brute-force optimum of 0.9281. The oracle is now an exact
   multiple-choice knapsack and matches brute force on 25 random instances
   (`tests/test_policy_and_sim.py::test_oracle_is_exact_against_brute_force`). This only
   changes the upper bound O, hence headroom and capture fraction; no policy saw it.

6. *Bug, found by reading the code.* In the negative control ("no stale items"), stale
   copies had their flag cleared but kept their own duplicate group, so near-identical
   copies (cosine ≈ 0.997) counted as distinct value for the oracle and the non-dedup
   baselines but not for dedup policies. They are now duplicates of their source in that
   condition only.

Nothing in the gates, thresholds, conditions, systems A–E, or world parameters changed.
