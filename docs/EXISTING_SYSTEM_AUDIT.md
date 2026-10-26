# Existing system audit

> The current, complete issue list is `CI_ENGINEERING_AUDIT.md`. This file keeps the earlier
> audits as history. References here to `THREAT_MODEL.md` now point to `CI_SECURITY_MODEL.md`.

## Re-audit, 2026-10-24 (before milestone 1: cross-model exchange)

**State:**

- Branch `feat/workspace-autonomy`, one commit ahead of `main` (`71377f7`); not pushed.
- Two uncommitted changes, preserved:
  - `store.py` chmods the database to 0600;
  - `tools.py` refuses `git_status` inside the home-directory repository, and fixes a cancel race. This one was already staged.

**Broken at audit time:**

- **The whole store-backed suite failed:** 11 failed, 31 errors out of 66 tests.
- **Cause:** the uncommitted `chmod` in `store.py` used `os` without importing it.
- **Fix:** added the import, the smallest change that completes the edit's intent. The suite then passed: 66/66.

**Real, simulated or missing, against the brief's sixteen-step demonstration:**

| step | state at audit | gap |
|---|---|---|
| 1–3 sign in, project, goal | real (bearer token, projects, objective, success criteria) | no constraints field |
| 4 real request to Provider A | **never done**: no API keys on this machine | the `claude` CLI is installed and logged in, so it can serve as a real provider |
| 5 context with provenance | real, but notes recorded only "model", not which model | provider and model on every note |
| 6–7 restart and recover | real, tested | none |
| 8 resume with Provider B | only Anthropic had tool use | tool use for a second adapter |
| 9 B gets only authorized context | memory was egress-checked, but **the project briefing went to whatever provider the chat named**, with no check | per-project provider access; a disclosure record per message |
| 10–11 delegated verification, evidence, disagreement | **missing** | exchange gateway, `AgentMessage`, claim ledger |
| 12–14 reversible action, approval, execution | real (write with backup and undo, approvals, re-check at execution) | none |
| 15–16 revoke, prove blocked | consent revocation existed for memory, but nothing stopped project content | as for step 9 |

**Security findings at re-audit:**

1. Project content had no provider-level gate (step 9 above). This was the main finding.
2. A late approval on a closed session would restart that session's model loop. Found in a live run, fixed.
3. The session reflection described a proposed, unexecuted write as "changed". Found in a live run when the next model believed it, fixed.
4. Unchanged from before: `$HOME` is a git repository with sensitive untracked files. Every tool still refuses it.

**Decision:** keep the Python stack again. Nothing in the milestone needs TypeScript,
Postgres or Redis. The brief's fallback stack applies only "if no functional repository
exists", and one does.

## Original audit, 2026-10-19

Audited 2026-10-19 at commit `c478a94` on `main`, before the workspace and autonomy work.

## Repository

- **Location:** `~/projects/contextual-intelligence`
- **Remote:** `github.com/Nonchalntnmr/contextual-intelligence`, private.
- **Branches:** `main` (one commit), plus this work on `feat/workspace-autonomy`.
- **Working tree:** clean at audit time.
- **Stack:**
  - Python 3.14 (`.venv`), package `pcim`.
  - FastAPI + uvicorn for HTTP.
  - SQLite for storage.
  - Static HTML/JS dashboard, no build step.
  - numpy for the policy.
  - `cryptography` (AES-GCM) for encryption at rest.
  - Official `anthropic` SDK 1.x, plus httpx for the other providers.
- **Not present:** Node/TypeScript, Postgres, Redis, Docker, CI pipeline, authentication beyond a bearer token.

## What is real, simulated, broken or missing

| area | state | evidence |
|---|---|---|
| Temporal memory (bitemporal, supersession, correction, conflicts, TTL, lineage, cascade delete) | **real, tested** | `tests/test_memory_and_sovereignty.py` |
| Encryption at rest (per-record keys, crypto-shredding, secure_delete, WAL truncation) | **real, tested** | test that scans DB bytes for a plaintext canary; test that a deleted key cannot decrypt |
| Consent and egress (per source × provider, lineage-aware, SECRET never leaves) | **real, tested** | 5 tests |
| Audit log (hash-chained, content-free payloads) | **real, tested** | tamper-detection test |
| Injection defence (write gating real; pattern scan heuristic; nonce-fenced rendering) | **real, tested**, with stated limits | 3 tests |
| Learned context policy (population EM prior, density-greedy) | **real**; validated in **simulation only** | G0 FAIL 4/7, G0b FAIL 5/7 (`experiments/`) |
| Mock model | **real test double** | deterministic; reports what it was given |
| Anthropic adapter | implemented on the official SDK; **wire format tested, no live call** | no API key on this machine |
| OpenAI-compatible and Gemini adapters | implemented over httpx; **wire format tested, no live call** | same |
| Assistance-mode bandit, knowledge tracing, intent/VOI | implemented, unit-tested, **not validated on people** | |
| FastAPI API, dashboard, `pcim` CLI | **real**, live smoke test on localhost | `tests/test_api.py` |
| Projects / goals as first-class objects | **missing** | memory has `goal` items only |
| Conversation persistence across sessions | **missing**: interactions are single-shot `ask` calls | |
| Tool use / execution of any kind | **missing by design**: no side-effecting tools exist | |
| Autonomy modes, approvals, cancellation, durable task queue | **missing** | |
| Streaming | **missing** | |
| MCP server | **missing**: SDK installed (mcp 2.3), no server yet | |
| Multi-user authorization | single local user per store; bearer token; onboarding closes after the first user | |

## Security findings at audit time

1. **No secrets in the repository.** `.gitignore` covers `*.db`, `*.key` and `data/`. The store, key and token live in `$PCIM_HOME` outside the repo.
2. **The master key sits on disk** with 0600 permissions. A key held in the macOS Keychain is a roadmap item.
3. **No execution surface exists today, so there is no execution risk yet.** Adding tools is the single largest change to the threat model. `docs/THREAT_MODEL.md` is written with this work.
4. **Machine-specific hazard: `$HOME` is itself a git repository** with sensitive files untracked. Any tool layer must refuse `$HOME` as a workspace root and must never run git write operations.

## Mapping to the CI architecture in the build prompt

| prompt capability | existing component | gap |
|---|---|---|
| A. Persistent context | `memory.py`, `profile.py`, `policy/` | projects, sessions, decisions, open questions |
| B. Joint reasoning | none beyond the model call | a system prompt that requires critique; structured plans |
| C. Orchestration | `adapters/` (4 adapters, one packet format) | tool calling |
| D. Adaptation | `learning.py`, `feedback.py`, `assist.py`, `growth.py` | collaboration preferences that persist (Demo E) |
| E. Real execution | none | typed tools, governor, executor, approvals |
| F. Continuous availability | durable SQLite store | durable task queue, cancellation, restart semantics |

## Decision

Keep the stack. The prompt says not to migrate frameworks for fashion, and nothing missing here requires TypeScript, Postgres or Redis yet.

Build the missing parts as Python modules behind the same engine, so consent, egress, encryption and audit cover them automatically.
