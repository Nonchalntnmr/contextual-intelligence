# Contextual Intelligence: Personal CI and PCIM

A research prototype of a user-owned context layer that sits in front of any foundation
model. It decides what each model should know about you for each request, learns from
outcomes which context actually helps, and can prove what it sent and what it deleted.

**Start here.** Read `docs/RESEARCH.md` §0 first; it is half a page. The short version:

- **Shown, in simulation only:** a value model learned *across a population* beats a
  tuned strong-memory heuristic by about 15pp.
- **Small and conditional:** per-person adaptation adds 1–3pp, and only when people
  really differ and give dense feedback.
- **Not yet tested:** nothing has been evaluated on real people or real models. The next
  experiment (G1a) is designed and needs API keys.

## What's here, and how real it is

| part | status |
|---|---|
| Temporal memory: bitemporal, supersession vs correction, conflicts, TTLs, lineage (`memory.py`) | working, tested |
| Sovereignty: per-record encryption with crypto-shredding, consent per source × provider, lineage-aware egress guard, hash-chained audit, injection write-gating (`sovereignty/`) | working, tested (including "plaintext never on disk" and "deleted key cannot decrypt") |
| Learned context policy: population EM prior, posterior-mean action, density-greedy knapsack (`policy/`) | working; validated in simulation only (G0b) |
| G0/G0b simulator, runner, pre-registrations, deviation logs, reports (`observatory/`, `experiments/`) | working; results in `experiments/g0*/results/REPORT.md` |
| Model adapters: mock (`adapters/`) | working |
| Model adapters: Anthropic (official SDK), OpenAI-compatible, Gemini (`adapters/`) | tested against recorded wire formats; **no live call made** (no keys configured) |
| Assistance-mode bandit, knowledge tracing (unassisted probes only), intent and VOI (`assist.py`, `growth.py`, `intent.py`) | experimental: implemented and unit-tested, not validated on people |
| Continual learning: immutable snapshots, rollback, non-inferiority promotion gate (`learning.py`) | working, tested; off-policy evaluation is future work |
| FastAPI backend + dashboard + `pcim` CLI | working (live-smoke-tested on localhost) |
| Postgres/pgvector backend, Redis, Next.js frontend, Docker deployment | **not built** — deliberately deferred; see below |

**Stack decisions**, which deviate from the brief on purpose:

- **SQLite instead of Postgres.** Local-first is the sovereignty argument made concrete:
  one encrypted file, no server.
- **No Redis.** One user on one machine does not need it.
- **A static dashboard instead of Next.js.** This repo's job is to test the thesis.
- All three are on the 90-day roadmap for the hosted, multi-tenant version.

## Run it

```bash
cd ~/projects/contextual-intelligence
.venv/bin/python -m pytest -q                       # 45 tests, ~1.5 s
export PCIM_HOME=~/.pcim                             # where your encrypted store lives
.venv/bin/pcim init --user nishanth --providers mock
.venv/bin/pcim remember "I prefer concise answers" --kind preference
.venv/bin/pcim ask "plan my week" --receipt
.venv/bin/pcim serve                                 # dashboard at http://127.0.0.1:8765
```

**Real models.** Set `ANTHROPIC_API_KEY`, then use `--providers mock anthropic` at init and
`--provider anthropic` when asking. The Anthropic adapter defaults to `claude-opus-5-5` at
effort `medium`, with prompt caching on the stable system section. It also turns on
**server-side refusal fallbacks** (`fallbacks: "default"`); pass `fallbacks=False` to
`AnthropicAdapter` to opt out.

**Re-running the experiments.** These are reproducible; the confirmatory runs refuse to
run twice.

```bash
.venv/bin/python -m pcim.observatory.g0 --exp g0  --dev --seed 0     # dev scale, ~25 s
.venv/bin/python -m pcim.observatory.g0 --exp g0b --dev --seed 10
```

## Docs

- `docs/RESEARCH.md` covers feasibility, the PCIM architecture with its math, the G0
  evidence, the G1–G3 programme, falsifiable hypotheses, the roadmap and the evidence bar.
- `docs/STRATEGY.md` covers the business model, unit economics, competition (as of
  2026-10), the pitch and deck outline, hiring, fundraising, the execution roadmap and the
  ten risks.
- `docs/REFERENCES.md` lists 83 sources, all checked by web search (not through LibKey).
- `experiments/g0/` and `experiments/g0b/` hold the pre-registrations, deviation logs,
  configs, reports and diagnostics.

## Personal data

Your store, its key and its token live in `$PCIM_HOME`, outside the repo. `.gitignore`
also excludes `data/`, `*.db` and `*.key`. This repo has its own `git init`, because
`~` is itself a git repository.
