# CI benchmarks

Every number here comes from a run recorded in the repository. Re-run commands are at the
end. Two kinds of measurement are kept apart:

- **CI's own overhead:** context selection, encryption, SQLite, the agent loop. Measured with a zero-latency mock model, so it is not hidden behind model latency.
- **Live behavior with real models:** latency, cost, task outcomes. It depends on the model and the day.

## 1. CI's own overhead

**Setup.** `scripts/bench_runtime.py`, 2026-10-26, Apple Silicon laptop, Python 3.14, a real
encrypted store with WAL. N is the number of memory items. Figures are median milliseconds
over 15–20 repetitions.

- **Before:** `experiments/bench/baseline-2026-10-26.json`, taken before any optimization.
- **After:** `experiments/bench/after-2026-10-26.json`, with all three changes below.

| N | metric | before | after |
|---|---|---|---|
| 100 | select_context | 11.4 | **0.9** |
| 100 | follow-up agent turn | 12.7 | **2.6** |
| 100 | first turn (new session, briefing) | 14.5 | **2.7** |
| 1000 | select_context | 28.3 | **3.0** |
| 1000 | follow-up agent turn | 30.0 | **5.1** |
| 1000 | first turn | 47.4 | **5.8** |
| 3000 | select_context (p95) | 62.6 (70.5) | **7.9 (10.4)** |
| 3000 | follow-up agent turn | 65.1 | **10.0** |
| 3000 | first turn | 100.9 | **12.2** |
| 3000 | SQL statements per follow-up turn | 3019 | **25** |
| 3000 | record decryptions per follow-up turn | 3002 | **5** |
| 3000 | select_context peak Python memory | 10.5 MiB | 7.8 MiB |
| any | one read_file tool call (governor + executor + audit) | 0.4 | 0.4 |
| any | briefing build and render (30 notes) | 1.3 | 1.4 |
| — | 8 threads × 10 turns, distinct sessions | 51.8 turns/s | **422 turns/s**, 0 errors |

**Which change bought what.** Each was measured on its own before the next was made.

| change | cause, found by profiling stages | effect at N=3000 |
|---|---|---|
| consent grants read once per egress decision | `allows()` queried SQLite per item | egress stage 18.4 → 4.7 ms; SQL 3019 → 25 |
| decrypted-item cache, invalidated on every write and on another process's commit | every request decrypted every item | memory stage 32.3 → 3.2 ms; decryptions 3002 → 5 |
| egress audit records a count and hash, not every permitted id | the audit payload grew with N (about 75 KB per request) | select 32.2 → 17.4 ms, and the audit log no longer grows with memory size |
| greedy propensities computed directly | 32 identical deterministic selections per request | the constant ~10 ms per request removed (select 17.4 → 7.9 ms; N=100: 10.2 → 0.9 ms) |

**What did not change.**

- The briefing got 0.1 ms slower: per-vendor lineage checks on notes. That's within noise, and expected.
- Tool calls are unchanged.

**What remains O(N).** The in-memory candidate scan and the per-item egress checks, about
2.6 ms per 1,000 items. At 3,000 items a full CI turn costs about 10 ms. A model step costs
2–30 seconds.

## 2. Live latency and cost

`scripts/milestone1_live.py`, run 4, 2026-10-26, through the Claude Code login. Cost figures
are the CLI's list prices: notional under a subscription.

| step | model | wall time | cost |
|---|---|---|---|
| first request: read 2 files, record 2 claims, 3 notes, propose a write | Opus 5.5 | 28.7 s | $0.160 |
| resume and answer from the briefing | Haiku 5.5 | 30.6 s | $0.012 |
| one delegated verification | Sonnet 5.5 | 4.8 s | — |
| write, approve, test, approve | Haiku 5.5 | 25.7 s | — |
| the whole demonstration, including two server starts | | 90.9 s | |

**Where the time goes.** Model steps take 2–30 s; CI's own work per step is 3–12 ms. CI's
overhead is under 1% of a turn. The remaining latency is the model, and the CLI's process
startup (about 0.3 s per step).

## 3. CI versus no context versus naive history (E1)

- **Report:** `experiments/e1/REPORT.md`.
- **Registration:** `experiments/e1/PREREGISTRATION.md`, committed before any model call.
- **Data:** 24 fixed tasks, one synthetic person, deterministic regex grading.

**Confirmatory, Claude Haiku 5.5, 2 repetitions:**

| arm | success | private data in prompt | stale facts in prompt | input tokens (median) | cost per success |
|---|---|---|---|---|---|
| A: no context | 18/48 | 0% | 0% | 656 | $0.00063 |
| B: naive history | 44/48 | 100% | 100% | 1226 | $0.00042 |
| **C: CI** | **46/48** | **0%** | **0%** | **1921** | $0.00042 |

| comparison | result |
|---|---|
| **C vs A** | 28–0 discordant, p ≈ 7×10⁻⁹. **Supported** |
| **C vs B** | 4–2, p = 0.69. **Not supported** |

**Ablations,** each removing one component from C. None is significant at this sample size.

| component removed | change vs C | p |
|---|---|---|
| temporal filtering | −5 | 0.06 |
| egress | −4 | 0.29 |
| scoping | −3 | 0.25 |
| the budget | no change; it does not bind | — |

**Exploratory, Claude Sonnet 5.5, 1 repetition:**

| arm | success |
|---|---|
| A | 9/24 |
| B | 18/24 |
| C | 24/24 |

C vs B: 6–0, p = 0.03. Exploratory. With the full history, Sonnet put the unconsented
salary and health detail into answers, and assumed the wrong project.

**What this licenses.** On this dataset:

- CI's context beats no context.
- On Haiku, CI ties naive history on success.
- CI keeps private, injected and stale content out of every prompt, where naive history sends all of it.

It does not license "CI makes assistants better" in general.

**The measured weakness.** At this memory size, selection is not relevance-selective
(precision 3.3%). CI sends 57% more input tokens than naive history. That makes it equal on
cost per success, not cheaper.

## Reproduction

```bash
cd ~/projects/contextual-intelligence
.venv/bin/python scripts/bench_runtime.py --sizes 100 1000 3000 --reps 20 --out /tmp/bench.json        # offline, ~1 min
.venv/bin/python -m pcim.observatory.e1 --prompts-only --out /tmp/e1-prompts                            # deterministic, no model
.venv/bin/python -m pcim.observatory.e1 --model haiku --reps 2 --out /tmp/e1-haiku                      # live: ~$0.17, ~5 min
.venv/bin/python scripts/milestone1_live.py --home /tmp/ci-live --out /tmp/ci-live/report               # live: ~$0.20, ~90 s
```

A confirmatory E1 rerun requires the three files hashed in `experiments/e1/PREREG_HASHES.json`
to be unchanged. The runner records whether they are.
