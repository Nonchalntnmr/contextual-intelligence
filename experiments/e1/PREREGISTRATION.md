# E1 preregistration: CI's context engine against no context and naive history

Registered 2026-10-26, before any model call on this dataset. The SHA-256 of this file,
`dataset.json` and `src/pcim/observatory/e1.py` are recorded in `PREREG_HASHES.json`. A run
whose hashes match is labelled confirmatory. Any change afterwards is a logged deviation.

## Question

On a fixed set of tasks for one synthetic person, does CI's context engine improve strictly
graded task success compared with:

- (A) the same model with no context;
- (B) the same model given the person's full history?

Which CI components account for the difference?

## What this is not

This is a **capability test**. The person, the memories and the tasks were written by the
author of CI, around the problems CI is designed to handle:

- corrected facts;
- private items without consent;
- an injected instruction in a document;
- two projects;
- irrelevant memories.

A positive result shows CI handles these cases on this set. It is **not** evidence that CI
improves AI assistance in general. The "unseen" category was written in the same session, not
by an independent party; it is held out only in the sense that nothing was tuned on it.

## Conditions

- **Model:** Claude Haiku 5.5 through the local Claude Code login (`claude -p`, CLI tools off, no settings or MCP).
- **Repetitions and order:** 2 repetitions per (arm, task). Execution order is shuffled with seed 20261026.
- **Errors:** a provider error is retried once. A call that still fails counts as a failed task and is reported.
- **Arms:** A, B, C, C_all, C_no_temporal, C_no_egress, C_no_scope, C_unlimited, as defined in `e1.py`.
- **System prompts:** A and B share one system prompt. C and its ablations use CI's renderer (preamble, labels, fenced notes) with no assistance directive.

## Outcomes

- **Primary:** strict success. Every `required` pattern matches and no `forbidden` pattern matches. Case-insensitive regexes, fixed in `dataset.json`.
- **Secondary, prompt level** (deterministic, computed without a model):
  - private content present in the prompt;
  - the injected instruction present in the prompt;
  - superseded statements present;
  - context precision and recall against the `relevant` labels;
  - estimated prompt tokens.
- **Secondary, output level:**
  - private content in the answer;
  - the injected vendor named on the injection tasks;
  - success by category;
  - input tokens as reported by the provider;
  - latency;
  - cost and cost per success.

## Already known before registration (deterministic, from `--prompts-only`)

Inspected to check the harness, before any model call:

- B and C_all put private content, the injected instruction, and superseded statements in 100% of prompts. C puts each in 0%.
- **C's selection is not relevance-selective at this memory size.**
  - Context precision is 3.3% for C and 2.4% for B. Recall is 1.0 for both.
  - The 1200-token budget does not bind: C_unlimited builds the same prompts as C.
  - C's prompts are longer than B's (median about 855 vs about 410 estimated tokens), because of the renderer's preamble and per-item labels.

This is recorded as a finding about the current system. It is **not** fixed before this run:
fixing it now would tune on the test set.

## Hypotheses and decision rules

| | hypothesis | test |
|---|---|---|
| H1 (primary) | success(C) > success(B) | paired exact McNemar over (task, rep), two-sided |
| H2 (primary) | success(C) > success(A) | same |
| H3 | private content in answers: C = 0, B > 0 | counts |
| H4 | injected vendor named on injection tasks: C = 0 | counts |
| H5 | control tasks (T18, T19): C not worse than A | reported; no test |

**Claims.**

- **Holm correction:** applied across H1 and H2 at α = 0.05.
- **What a significant result licenses:** only the claim "on this dataset and model, C's success rate exceeds B's (or A's)."
- **Negative results:** a non-significant or reversed primary is reported as such, in the same place and with the same prominence.
- **Ablations, categories, cost and latency:** exploratory. They are reported with p-values, but no claims rest on them.

**Exploratory secondary run.** Arms A, B and C, 1 repetition, with Claude Sonnet 5.5, to see
whether the pattern holds on a stronger model. It is labelled exploratory whatever it shows.

## Known threats to validity

- **Small sample:** 24 tasks × 2 reps per arm. Intervals will be wide.
- **Regex grading:** it can misgrade paraphrases. Rules were written before any run and are not changed after seeing answers. Misgrades found afterwards are listed, but the registered numbers stand.
- **One persona, one author.** Tasks are built around CI's mechanisms, which favors CI by design.
- **One vendor.** The CLI adds a small preamble of its own to every arm alike.
