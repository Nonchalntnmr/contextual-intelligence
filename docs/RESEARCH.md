# CI research plan — feasibility, PCIM, experiments, evidence bar

Covers Steps 1, 2, 3, 6, 8 and 9 of the founding brief and Parts II, IV, V, VII and XIII.
Numbers in brackets refer to `docs/REFERENCES.md`. Every result below is from
**simulation** unless it says otherwise. No real user and no real model has been
evaluated yet.

---

## 0. Status

The core hypothesis — a learned personal representation improves a foundation model for
one person beyond strong memory — is **not established**. Two pre-registered simulation
studies support a narrower and different claim:

1. **Outcome-learned context selection works when it is learned across a population.**
   A population-fitted value model beat a tuned strong-memory heuristic by about +15pp of
   response quality in every simulated world that had population structure. It was
   non-inferior in a world where similarity is all that matters (G0b).
2. **Per-person adaptation on top of that is real, small and conditional.** It added
   +1.7pp (binary feedback) to +2.8pp (dense feedback) when people genuinely differ. It
   cost 0.8pp when they do not, and it lost 1.6pp under sparse feedback. Under the decision
   rule registered before the run, it is a research problem, not a product feature (G0b).

Whether real people and real models have the structure the simulator planted is the G1
question, and nothing here answers it.

---

## 1. Feasibility (Step 1)

### 1.1 The constraint that shapes everything

Closed frontier models accept tokens. They do not accept embeddings, adapters, or
gradients. So whatever CI learns about a person reaches GPT, Claude or Gemini through
exactly five channels:

1. the context it sends;
2. which tools it exposes;
3. which model it picks;
4. decoding settings, which are shrinking on current models;
5. what it does with the output (verification, best-of-n with a personal scorer).

That turns the founding equation into a concrete object:

    I_i(t) = M( render( π_θ(H_i(t), T, M) ) )

where π_θ is a **context compiler**: a small learned policy *around* the model that
chooses what to say. It is not a model *inside* it. Three consequences follow.

- **Portability is only achievable in token space.** The literature agrees, and the case
  for it is now strong:
  - Text user summaries learned with RL transfer zero-shot to GPT-4o [67].
  - Text preference summaries transfer across model families [82].
  - PEFT adapters degrade even when the *same* base model is updated [68].
  - A per-user neural representation can still exist. It just has to act by choosing text.
- **Per-user model weights are the wrong default.** Per-user PEFT gives +1.07% over a
  non-personalized model against +14.92% for retrieval [63]. The parametric methods that
  beat retrieval all share capacity across users: PPlug [11], User-LLM [12],
  Profile-to-PEFT [66], and LINEUP, with an 8-scalar per-user code [83]. PCIM therefore
  keeps the per-user state tiny and the learned machinery shared.
- **Selection headroom shrinks as models improve. Data minimization does not.**
  - Frontier models tolerate distractors and long context better each generation, which
    erodes the *quality* case for careful selection.
  - Prompt caching (cache reads at 0.025–0.1× of input price) erodes the *cost* case.
  - What never erodes is that sending your whole life to every provider on every call is
    a privacy cost. The objective therefore prices privacy exposure explicitly (§3).
    Minimal sufficient context stays valuable even if quality headroom goes to zero.

### 1.2 What is established, what is plausible, what is speculative

| claim | status | evidence |
|---|---|---|
| Retrieving from a user's profile improves personalized tasks | established | LaMP [1], LongMemEval [3], LoCoMo [4] |
| Frontier models are poor at long-term user modeling unaided | established | PersonaMem ~50% [5], preference following <10% after 10 turns [61], CUPID <50% precision on "which past context applies" [62] |
| Trained retrievers beat strong similarity retrieval for personalization | weak: small, inconsistent | [8] (gains on some LaMP tasks, none on others); larger margins only in authors' own setups [64,65,79,81] |
| Oracle evidence beats what similarity retrieves | plausible, one preprint | similarity at a 5% budget finds the needed evidence for 61% of questions; oracle evidence +15.6pt over full history [80] |
| Learned selection beats strong heuristics *in simulation* | shown here | G0b, §4 |
| Per-user adaptation beats population learning | conditional, small, simulation only | G0b: +1.7 to +2.8pp with strong individual structure, negative otherwise |
| Personalization transfers across providers | plausible for text, false for parameters | [67,82] vs [68] |
| Personalized assistance increases *unassisted* capability | open; the default effect is negative | unguarded AI help reduces later independent performance [24,69,78]; well-designed tutoring can help [70] |
| Optimizing for user feedback is safe | false by default | feedback-RL learns manipulation [73]; memory raises sycophancy [74,77]; sycophancy raises dependence [75] |
| Memory is a security boundary | established attack surface | indirect injection [23], persistent memory injection [46,71,72] |

### 1.3 The three strongest possible breakthroughs

1. **Outcome-learned, model-agnostic context compilation, learned collectively from
   content-free features.**
   - A shared value model over item×task features is fit by EM across many people. Each
     person keeps a few-dimensional deviation.
   - The population part can be fit by *federated* EM. Each device sends only the
     sufficient statistics of 44-dimensional feature vectors (no text), under secure
     aggregation [33] with differential-privacy noise [32,60].
   - If real-model experiments reproduce G0b's population effect, this would be a learned
     context layer that improves any provider's model for everyone without centralizing
     anyone's context. Nobody has shown that yet.
2. **A calibrated, bitemporal belief state about the person, plus value-of-information
   clarification.**
   - CI knows what it believes, since when, on whose authority, and how sure it is.
   - It asks only when the answer would change what it does (VOI > interruption cost).
   - The measurable claim is fewer corrections at equal question rate against
     append-only memory.
   - Memory systems treat staleness as retrieval noise [3,77]. Treating it as belief
     revision is the gap.
3. **Measured gain in *unassisted* capability as an optimization target.**
   - The default effect of AI help on independent performance is negative [24,69,78].
   - A system shown, in a randomized longitudinal study, to raise what people can do
     *without it* would be scientifically significant and commercially rare.
   - It would also be the strongest answer to "won't the frontier labs just build this".

---

## 2. PCIM architecture (Step 2)

Implementation status: **W** working and tested · **E** experimental (implemented, not
validated on people) · **F** future.

| layer | what it is (code) | math | learned from | cost | main failure modes | status |
|---|---|---|---|---|---|---|
| **A** personal representation | stated profile + inferred items + learned deviation δ_i + skill state (`profile.py`, `policy/learned.py`, `growth.py`) | β_i = β_0 + δ_i, δ_i ~ N(0, UΛUᵀ + dI); z_i = Λ^-½ Uᵀ δ̂_i | feedback (δ), the person (stated), unassisted probes (skills) | O(d²) per user, d = 44 | stereotyping, staleness, a latent that looks like insight but is noise | stated **W**, learned **E** (only shown where |δ̂| > 2 SD) |
| **B** temporal memory | bitemporal store, supersession vs correction, conflicts, TTLs, lineage, crypto-shredding (`memory.py`, `store.py`) | valid time × system time; `as_of`, `valid_at` | writes, corrections | O(N) per user scan; ANN index for hosted | poisoning, contradiction storms, embedding-model migration | **W** |
| **C** intent | soft task-type distribution; VOI rule (`intent.py`) | VOI = E_s[max_a U] − max_a E_s[U] | — | negligible | lexical classifier is crude; interpretations need an LLM call | classifier **W** (heuristic), VOI **W**, interpretation generation **F** |
| **D** context selection | multi-channel candidates → egress filter → Bayesian linear value model → density-greedy knapsack (`policy/`) | include m iff E[gain_m] > λ·tok_m + ρ_s·tok_m; order by gain/token | population EM + per-user posterior | ~1 ms per request | non-additive item utility, budget blindness (found in G0), exploration cost (found in G0) | **W** (G0b-validated in simulation) |
| **E** cognitive adaptation | assistance-mode bandit; growth goals restrict modes (`assist.py`) | Beta-Bernoulli TS per (task, mode) | reward | negligible | learns "just do it" because it pleases; deskilling | **E** |
| **F** model interface | provider-neutral packet; trusted system section vs nonce-fenced data block; adapters (`adapters/`) | — | — | none | prompt injection; cache invalidation; provider drift | mock **W**; Anthropic **W** against recorded wire format via the official SDK, live call untested; OpenAI-compatible, Gemini same |
| **G** outcome evaluation | reward from signals, verified outcomes dominant (`feedback.py`); observatory (`observatory/`) | fixed weights, published | — | — | rewarding sycophancy [73–75] | **W** (sim), **F** (human evals) |
| **H** continual learning | versioned immutable snapshots, rollback, stabilised forgetting, influence caps, non-inferiority promotion gate (`learning.py`) | P ← γP + (1−γ)P₀ + xxᵀ/σ² | — | O(d²) | drift, poisoning of the shared prior, Goodhart | **W** (unit-tested), off-policy evaluation **F** |
| sovereignty | consent ledger, lineage-aware egress guard, hash-chained audit, envelope encryption, write gating (`sovereignty/`) | — | — | — | KEK on disk; backups predating deletion; heuristic injection detection | **W** |

**What is genuinely new here, and what is not.**

- Not new, each with prior art: a memory stream with recency, importance and relevance
  [17]; temporal knowledge graphs [18]; feedback-trained retrieval [8,64]; per-user
  latents [13]; hierarchical bandits [56,57]; federated bandits [58].
- Plausibly new, to be shown:
  - the combination of a *population-learned, provider-agnostic* value model over
    content-free item×task features with a few-dimensional per-user deviation, trained
    from outcome feedback;
  - consent that follows lineage into derived items;
  - selection under an explicit privacy-exposure price.
- A new name for an old component is not a contribution, and this doc does not claim one.

---

## 3. Mathematical foundation (Part V)

### 3.1 The candidate objective, term by term

    max_{π,θ}  E[ Q_i(Y_t) + α·H_i(t) − λ·C_t − μ·L_t − ρ·R_t ]

| term | measurable? | perverse incentive if optimised naively | treatment in PCIM |
|---|---|---|---|
| Q quality | partly: verifiable tasks yes, open-ended no | satisfaction ≠ quality; sycophancy wins [73–75] | verified outcomes dominate; ratings can never raise a verified failure (`feedback.py`) |
| H capability gain | slowly, via unassisted probes and delayed transfer tests | teaching to the probe; or, if assisted successes count, rewarding dependence | **constraint, not a weight**: growth goals restrict modes; only unassisted probes move mastery |
| C compute cost | yes | none worth noting | λ per token in the inclusion rule |
| L latency | yes | none worth noting | budget cap |
| R risk | privacy partly (sensitivity-weighted tokens disclosed); safety and reliability poorly | trading safety for quality at some λ | hard constraints (SECRET never egresses, consent, injection gating) plus a priced exposure ρ_s per sensitivity level |

So the operational form is a constrained problem, not a weighted sum:

    max_S  E[ΔQ(S)] − λ·tokens(S) − Σ_m ρ_{s(m)}·tokens(m)
    s.t.   S ⊆ allowed(consent, provider), tokens(S) ≤ B, mode ∈ eligible(growth goals)

### 3.2 Bayesian / meta-learning view

Hierarchical model: β_i = β_0 + δ_i, δ_i ~ N(0, Σ_δ), with EM for (β_0, Σ_δ, σ²) and the
low-rank-plus-diagonal projection Σ_δ = UΛUᵀ + dI.

In the linear-Gaussian case, **meta-learning a user-embedding space is learning Σ_δ.** The
prior N(0, I) on z pushed through UΛ^½ is exactly the prior on δ. "Billions of personal
representations" therefore costs O(r) floats per person plus one shared model. That is
consistent with LINEUP's result: 8 scalars per user beat a 4.19M-parameter private LoRA [83].

### 3.3 Information-theoretic view: when is compact context sufficient?

C* = argmin |C| s.t. Q(M, C, T) ≥ Q_target is budgeted subset selection.

- **Easy case.** If utility were monotone submodular, greedy would be (1 − 1/e)-optimal.
- **Why the easy case doesn't hold.** LLM context utility is neither monotone nor
  submodular:
  - distractors have negative value [21,22];
  - two facts can be useless apart and decisive together (supermodular pairs);
  - redundancy is submodular.
- **Information bottleneck.** Find Z = f(H) maximizing I(Z; Y*) − βI(Z; H) [42].
- **The catch.** M is not a Bayes-optimal reader of Z, so "sufficient for an ideal reader"
  is not "sufficient for M".

Predictions, each testable in G1:

| prediction | regime |
|---|---|
| (a) Compression helps | when M is distractible and the needed information is sparse in H |
| (b) Compression hurts | when the task needs verbatim detail (code, numbers, quotations) or interacting facts |
| (c) The gap between compact and full context shrinks with model capability | for quality; never for disclosed tokens |

### 3.4 Bandit / RL view: two lessons G0 forced

1. **Budget-blind ranking is a knapsack error.** Ranking by expected value ignores what a
   long item crowds out.
   - In G0's negative control, the learned model took 8.7 items averaging 71 tokens
     against the heuristic's 12.0 averaging 46, and lost 6.8pp.
   - Ranking the *same scores* by value per token closed the gap: 0.741 vs 0.721, oracle 0.744.
   - The density order is the LP-relaxation optimum, and is now the default.
2. **Thompson sampling over 44 dimensions explores more than it earns.**
   - Posterior-mean action, relying on natural variation in tasks and candidates, beat
     Thompson sampling in all 12 conditions (G0b, pre-registered secondary).
   - The cost: with deterministic action, the logged propensities are 0/1. Off-policy
     evaluation of alternative policies then needs a deliberate exploration fraction,
     whose price G0b puts at roughly 1–5pp.

### 3.5 Causal view: credit assignment

Feedback scores a response, not an item. Additive regression on x(S) = Σφ assigns credit,
which is identified only if S varies independently of what drives reward.

- That holds in the simulator by construction.
- It fails in real logs whenever the policy's choices correlate with task difficulty.
- Remedies, in order of cost:
  1. a small randomized inclusion fraction with logged propensities;
  2. paired leave-one-out ablations on a sample of tasks (G1);
  3. slate off-policy estimators [26,55], whose assumptions break under a hard budget,
     which is why they are listed as future work.

### 3.6 Multi-objective view

Q, H and R conflict: doing the work maximizes Q now and can minimize H later [24]. Weighted
sums hide that trade. PCIM makes the person choose it per task type (growth goals) and
reports the Pareto frontier in evaluation, not one blended score.

---

## 4. Evidence so far: G0 and G0b (Step 3, in simulation)

**Design.**

- 400 training users and 200 test users per condition, 200 interactions each.
- Planted structure: a population affinity A[type, kind] and a rank-2 personal deviation.
- Staleness, duplicates, novel topics.
- Feedback noise: dense, binary, or sparse at 30%.
- Every system gets the same candidates, the same 600-token budget and the same tasks.
- Paired within user; 95% user-cluster bootstrap.
- The strong baseline B2 has its 10 parameters tuned against *noiseless true quality*,
  which is more information than any learned policy gets.

Full pre-registrations, deviation logs and reports: `experiments/g0/`, `experiments/g0b/`.

| gate | G0 (as registered) | G0b (revised method, fresh seed) |
|---|---|---|
| G0-1 positive control: E2 beats B2, capture ≥ 0.25 | PASS (+16.6pp, CF 0.56) | PASS (+17.1pp, CF 0.72) |
| G0-2 negative control: E2 not worse than B2 by >1pp in a similarity-only world | **FAIL** (−8.0pp) | PASS (+0.1pp) |
| G0-3 personalization onset ≤ 100 interactions under sparse feedback | **FAIL** (never) | **FAIL** (never) |
| G0-4 population prior beats a weak prior early | PASS (+15.2pp) | PASS (+19.6pp) |
| G0-5a personalization helps when people differ | PASS (+1.4pp) | PASS (+1.7pp) |
| G0-5b personalization costs ≤1pp when they don't | **FAIL** (−4.0pp) | **FAIL** (−0.8pp, CI to −1.1) |
| G0-6 per-user posterior recovers the planted deviation (median r ≥ 0.5) | PASS (0.85) | PASS (0.73) |
| overall | FAIL 4/7 | FAIL 5/7 |

**What the series shows.**

- Learning *which context helps which kind of task*, from noisy outcome feedback across a
  population, works: +15pp over a tuned heuristic, about 72% of the oracle headroom.
- Learning what is special about *one person* works only with strong individual structure
  and dense feedback. It never paid off under sparse feedback within 200 interactions.
- Stabilised forgetting helped after a preference change (+2.3pp).
- A prior fit under one simulated model transferred to another with no measurable loss.
  The simulated models differed only in length and staleness penalties, so that last
  result is weak.

**What it cannot show.** The simulator's additivity, its relevance function and its
structure are assumptions. G0b passing would have licensed G1, nothing more.

**Process disclosures.**

- The series includes three bug fixes and one calibration change, all logged before the
  run they affected.
- One of the fixes, the exact oracle, was caught by a brute-force unit test.
- G0b was designed after seeing G0, so its partial pass is weaker evidence than a
  first-try pass.

---

## 5. Experimental programme (Part VII, Step 3)

### 5.1 The decomposition: Gain = Headroom × Capture

Any context policy's advantage over strong retrieval is bounded by the **headroom**: the
quality of the best context subset minus the quality of strong retrieval, at the same
token budget, for that model. Headroom is a property of the world (people × tasks ×
models). **Capture** is a property of the method. G0/G0b measured capture under planted
headroom. The cheapest experiment that could kill the selection thesis is to measure real
headroom.

### 5.2 G1a — the headroom test (the next experiment)

- **Tasks.** About 200 personalized tasks with verifiable answers (multiple choice where
  exactly one option is consistent with the person's *current* context). Each comes with
  a 40–80-item memory: current facts, superseded versions, distractors, irrelevant items.
- **Conditions per task.**
  - no context;
  - similarity top-k;
  - strong temporal retrieval (B2);
  - gold minimal set;
  - best-of-search over subsets (oracle).
- **Models.** A small and a frontier model from at least two providers, to test whether
  headroom shrinks with capability (§3.3c).
- **Pre-registered kill rule.** If oracle − B2 < 3pp of accuracy on frontier models at
  matched tokens, the learned-selection thesis is dead for quality. CI's selection layer
  would then be justified only by cost and privacy (fewer disclosed tokens at equal
  accuracy), and the docs and pitch would be rewritten to say so.
- **Budget.** About 200 tasks × 15 subset evaluations × 4 models ≈ 12k calls. Roughly
  $10–60 at the list prices in STRATEGY.md §3.
- **Code needed.** A generator plus a harness that runs the existing adapters. This is the
  single next engineering action (§9).

### 5.3 G1b — capture on real models

The G0b protocol is rerun with real model outcomes on public benchmarks:

- LongMemEval [3], PersonaMem / PersonaMem-v2 [5,81], CUPID [62], PersistBench [77].
- Reward is the benchmark's own metric, not an LLM judge where avoidable. Judges carry
  position, verbosity and self-preference biases [52].
- Population users are the benchmark's personas.

### 5.4 G2 — the n = 1 then n = 20 study ("Nishanth 2.0")

- **Data.** Real, permissioned data with verifiable outcomes. The natural domain is
  research engineering on the person's own repositories, where outcomes are tests passing
  and numbers reproducing.
- **Design.** Per-task randomization between systems A–E, blinded where the interface
  allows. Pre-registered metrics:
  - verified task success;
  - corrections per task;
  - tokens disclosed;
  - cost per success;
  - time to success.
- **Identifiability on real data.** The learned section of Intelligence DNA stays labelled
  experimental until a split-half reliability check passes on real data.
- **Population requirement.** n = 1 cannot show population learning, which G0 says is
  where the value is. G2 therefore needs at least about 20 people before it says anything
  about the core thesis.

### 5.5 G3 — capability

A randomized longitudinal study. Growth-mode assistance is compared against standard
assistance, with delayed *unassisted* tests and transfer tasks. The primary outcome is
unassisted performance at 4 weeks, following the [24,69] designs.

### 5.6 Systems A–E and controls for G1–G2

| system | definition |
|---|---|
| A | same model, standard prompting |
| B | same model plus strong retrieval/memory (B2-class; also an off-the-shelf memory layer such as mem0 or Zep as an external reference) |
| C | native personalization |
| D | CI population-learned selection |
| E | D plus per-user continual adaptation |

**C cannot be run at a controlled budget.** Native memory is a product feature, not an API
feature. Two handling options:

1. use the provider's API-level memory tool where it exists (e.g. Anthropic's memory tool),
   reported separately;
2. run a protocol in the consumer apps, labelled non-reproducible.

**Controls.** Model version pinned and recorded; identical token budgets; identical
information available; identical tools; tasks stratified by difficulty; cost and latency
reported; no task or persona used for tuning appears in evaluation.

**Statistics.**

- Paired within person and task.
- Cluster bootstrap over people.
- Holm correction across the pre-registered primary contrasts.
- Non-inferiority margins fixed in advance.
- Any sweep counted against a trial budget, as in `trialguard`.

### 5.7 Conditions under which the CI hypothesis is rejected

- G1a headroom below 3pp on frontier models.
- G1b capture indistinguishable from zero against B.
- G2 E ≤ B on verified success at matched tokens with 20 people.
- G3 showing growth modes do not improve unassisted performance.

Any one of these retires the corresponding claim. The rest of the system (sovereign
memory, portability, receipts) would then have to stand as a product on its own, and the
pitch would say so.

---

## 6. The ten critical scientific challenges (Part XIII) as falsifiable hypotheses

| # | challenge | hypothesis | experiment | refuted if |
|---|---|---|---|---|
| 1 | Do learned representations beat strong memory? | D > B at matched tokens | G1b, G2 | D − B ≤ 0 (95% CI) |
| 2 | Does personalization transfer across models? | text-space policy fit with model X keeps ≥50% of its gain with model Y | G1b cross-model arm | retained gain < 50% |
| 3 | Does continual learning forget? | snapshot posteriors never regress on a frozen regression suite | H-gate on G2 logs | promoted version loses > margin |
| 4 | Do inferred preferences become wrong or harmful? | inferred items' error rate (user-judged) < 10% at 30 days | G2 audit sample | ≥ 10%, or any harmful inference |
| 5 | Overfitting to history? | E's advantage holds on novel-topic tasks | G1b/G2 novel-topic stratum | gain only on familiar topics |
| 6 | Privacy and security vulnerabilities? | zero egress of ungranted or SECRET items; injection-planted preferences = 0 | red-team suite (MINJA- and SpAIware-style [46,71]) | any success |
| 7 | Does improvement persist on unseen tasks? | gain on held-out task families ≥ 50% of in-family | G1b | < 50% |
| 8 | Do the economics beat simpler alternatives? | cost per verified success (D) ≤ B's | G2 | higher at equal quality |
| 9 | Unhealthy dependency? | unassisted performance does not fall for heavy users | G3 | significant drop vs control |
| 10 | Do frontier improvements erase the differentiation? | headroom stays ≥ 3pp on the newest models | G1a repeated each model generation | headroom → 0; then sell only cost and privacy |

---

## 7. Research roadmap toward learned personal representations (Step 6)

1. **Now: population-learned linear value model** over content-free features (built).
2. **Months 1–3: G1a and G1b.** Real-model headroom and capture. Replace lexical
   embeddings with a real embedder. Add randomized inclusion for credit assignment.
3. **Months 3–6: neural-linear scorer.** A shared network maps (item text, task text,
   metadata) to features, and the per-user Bayesian linear head stays. Only justified if
   G1b shows the linear features leave headroom.
4. **Months 6–12: federated EM** for the population prior: secure aggregation of
   per-device sufficient statistics plus a DP accounting [32,33,58]. Measure the
   utility-privacy curve against central fitting.
5. **Year 2: learned text summarization of the person** (PLUS-style [67]) as a second
   portable artifact. Its success is judged by transfer to unseen providers.
6. **Years 2–5: open-weight path.** Hypernetwork-generated adapters from the text profile
   (Profile-to-PEFT [66], Text-to-LoRA [43]) for local models. These are justified only
   where they beat the context path at equal cost, and they must survive base-model
   updates [68].

---

## 8. Evidence required before any claim (Step 9)

| claim | minimum evidence |
|---|---|
| "learns what context helps you" | G1b capture > 0 against B on two public benchmarks and two providers, pre-registered |
| "better than ChatGPT/Claude/Gemini memory" | G2 E > C at matched tokens, ≥ 20 people, verified outcomes; never claimed from simulation |
| "portable across models" | §6 #2 retained gain ≥ 50% on ≥ 2 provider pairs |
| "makes you more capable" | G3 randomized, unassisted outcome at ≥ 4 weeks |
| "private by design" | external security review plus a published red-team suite with zero critical findings |
| "novel" | a related-work comparison reviewed by someone outside the team, against [8,11,12,18,64–67,79–83] at minimum |

---

## 9. The single next engineering action

**Build and run G1a**, the headroom test (§5.2). It is the cheapest experiment that could
kill the core quality claim, and every later investment depends on its answer.

- **What already exists:** adapters, budgets, rendering and statistics.
- **What's missing:** a task generator with gold minimal sets, and a subset-search harness.
- **What it needs from you:** provider API keys (none are configured on this machine) and
  approval to spend about $10–60.
