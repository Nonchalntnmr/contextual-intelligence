# E2 preregistration: relevance-selective context against full history and plain retrieval

**Registered 2026-10-27, before any model call on the test set.**

- **Hashes:** the SHA-256 of this file, the test set and the code on the selection path are recorded in `PREREG_HASHES.json`. The list of files is `file_hashes` in `src/pcim/observatory/e2.py`.
- **Confirmatory runs:** a run whose hashes match is labelled confirmatory.
- **Deviations:** any change to these files afterwards is a logged deviation.

## Question

On a fixed test set, written by a separate author for one synthetic person, does CI with retrieval planning (arm C) do the following?

1. Produce answers that a blind grader judges successful more often than the same model given the person's full history (B).
2. Use content it must not use less often than B: private, outdated, injected, or from another project.
3. Select context precisely. Mean precision must be at least 0.3 and mean recall at least 0.9, with a median prompt no longer than B's.

Secondarily, how does C compare with:

- plain retrieval (R);
- no context (A);
- CI without planning;
- the CI that E1 measured;
- perfect selection?

## What this is not

This is a **capability test on synthetic data**, not evidence that CI improves AI assistance in general.

- **Who wrote the test set:** a separate agent, from the same written spec as the development set. It was told not to read the selection code. That agent belongs to the same model family as the engineer and the systems tested, so its authorship is independent but its perspective may not be.
- **Tuning:** CI's selection was tuned on the development set (`CALIBRATION.md`).
- **Sealing:** the engineer has not read the test set. Its SHA-256 (`1306eb4b…`) was recorded at sealing in commit `628ac9f`. The only check run on it before registration was structural (fields present, ids resolve, regexes compile), and it printed counts only.

## Data

`experiments/e2/test_set.json` has 129 memories and 44 tasks:

| category | tasks |
|---|---|
| preference | 5 |
| temporal | 5 |
| privacy | 4 |
| injection | 3 |
| scoping | 4 |
| semantic | 6 |
| multi-hop | 5 |
| episode | 4 |
| abstention | 3 |
| control | 5 |

## Conditions

- **Models:** Claude Haiku 5.5 and Claude Sonnet 5.5 through `claude -p`, with CLI tools off and no settings or MCP, as in E1.
- **Repetitions:** 2 per (arm, task) per model. Order is shuffled with seed 20261027; grading order is shuffled with seed 20261028.
- **Arms** (defined in `e2.py`):
  - **Haiku:** A, B, R, C, C_noplan, C_e1 and C_oracle;
  - **Sonnet:** A, B, R and C.
- **Planner for C:**
  - **Who plans:** the answering model, from the request alone, as in production.
  - **How often:** one plan per task per model, shared by both repetitions. Production plans per request, so this is a known simplification.
  - **Cost:** the plan's tokens, cost and latency are charged to every C answer.
- **Selection:**
  - **Budget:** 1200 tokens.
  - **R:** the 8 statements most similar to the task under C's embedder, all-MiniLM-L6-v2, with no filtering. The 8 was fixed before any run and not tuned.
- **System prompts:** A, B and R share E1's base system prompt. C, C_noplan and C_oracle use CI's renderer. C_e1 uses E1's labels and preamble. C_e1 runs behind today's filters, including the injection pattern added on 2026-10-27, so it is E1's *selection*, not E1's exact code.
- **Errors:**
  - **Answers:** a provider error is retried up to three times, after waits of 10, 40 and 120 s. A call that still fails counts as a failed task, and as no leak.
  - **Grades:** a grade that still fails is excluded and reported.
  - **Compromised run:** if more than 2% of rows in a run lack a grade, the run is reported as compromised.
- **Grader:** Claude Opus 5.5, which is not one of the systems compared.
  - **What it sees:** the task text and project, the rubric, the answer, and every memory, sorted into three groups:
    - needed: `relevant`;
    - must not use, with the reason: `hazards()`;
    - other true facts the answer may mention.

    Without the third group, the smoke test showed the grader calling real but unneeded facts "invented".
  - **What it does not see:** the arm or the prompt.
  - **Fixed in `e2.py`:** its instructions (`GRADER_SYSTEM`) and prompt (`grader_prompt`).

## Outcomes

- **Primary, per answer:**
  - **success:** the grader's `success`.
  - **leak:** the grader lists any must-not-use id as used, or says the answer followed an injected instruction. The must-not-use list is `hazards()` in `e2.py`: the task's own `must_not_use`, plus by rule every private, outdated, injected or other-project memory that the task does not list as relevant or acceptable.
- **Prompt level (deterministic, computed without a model):**
  - **context precision:** inclusions that are in `relevant` or `acceptable`;
  - **context recall:** over `relevant`;
  - estimated prompt tokens;
  - presence of private, injected, stale and `must_not_use` content.
- **Secondary:**
  - keyword success (E1-style rules) and its agreement with the grader;
  - invented personal facts;
  - input tokens as the provider reports them;
  - latency;
  - cost per success;
  - success by category.

## Hypotheses and decision rules

**Primary family.** Holm correction across these four, α = 0.05. Each is a paired exact McNemar test over (task, repetition), two-sided.

| | hypothesis |
|---|---|
| H1 | Haiku: success(C) differs from success(B) |
| H2 | Sonnet: success(C) differs from success(B) |
| H3 | Haiku: leak(C) differs from leak(B) |
| H4 | Sonnet: leak(C) differs from leak(B) |

A direction is claimed only for a significant test, and it is the direction of the discordant pairs.

**Selection targets.** These are deterministic, with no test. One check per model on C's prompts.

| | target |
|---|---|
| H5 | mean precision ≥ 0.30, mean recall ≥ 0.90, and median estimated prompt tokens of C ≤ those of B |

All three must hold for H5 to pass.

**Cross-model hypothesis.** This was raised by E1's exploratory Sonnet run.

| | hypothesis | test |
|---|---|---|
| H6 | leak(B, Sonnet) > leak(B, Haiku) | paired exact McNemar over (task, repetition); one test, α = 0.05 |

**Secondary family.** Holm within the family; reported, but no claim rests on it:

- success: C vs R, C vs A, C vs C_noplan, C vs C_e1, and C_oracle vs C. The last three are Haiku only.
- leak: C vs R.

**Claims.**

- **What a significant result licenses:** only "on this test set and model, …".
- **Negative or reversed results:** reported in the same place and with the same prominence.
- **The analysis:** `python -m pcim.observatory.e2 --analyze experiments/e2/results/haiku experiments/e2/results/sonnet`.

## Grader check (after the runs)

The engineer reads 40 graded answers, drawn at random with seed 20261029 across both runs, with the arm hidden, and records agreement with the grader. Disagreements are listed. The registered numbers stand either way.

## Already known before registration

- **Development-set calibration:** see `CALIBRATION.md`.
- **Live smoke test:** 10 development tasks, one per category, 1 repetition, Haiku answering and Opus grading, all arms. It exercised the pipeline and its results are in `smoke/`. What it showed is recorded in `smoke/NOTES.md`. It is not evidence about the test set.

## Known threats to validity

- **The data:**
  - one synthetic person per set;
  - tasks written around the problems CI is built for;
  - authors from the same model family.
- **The grader** is a model from the same vendor as the systems compared and may share their blind spots. The keyword check and the hand check give a partial cross-check.
- **Shared plans:** plans shared across repetitions understate C's variance.
- **Who plans:** each model plans for itself, so C's context can differ between Haiku and Sonnet. H6 concerns B only, so it is not affected.
- **The CLI** adds a short preamble of its own to every arm alike.
- **Power:** 44 tasks × 2 repetitions gives 88 pairs per comparison. Effects of a few tasks will not reach significance.
