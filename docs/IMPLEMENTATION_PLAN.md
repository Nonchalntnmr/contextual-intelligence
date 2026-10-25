# Implementation plan

## Milestone 1: the sixteen-step demonstration (2026-10-24)

**Goal:** begin a project with one model, close the application, reopen it, and continue
with a different model without re-explaining. Show what the second model was and was not
given, verify a claim independently, run an approved reversible action, and prove that a
revocation holds.

| step | code | offline proof | live (`claude-cli`) |
|---|---|---|---|
| 1 sign in | bearer token (`api.py`) | `test_milestone1_sixteen_steps` | yes |
| 2–3 project, goal, constraints | `Workspace.create_project` (constraints, provider access) | same | yes |
| 4 real request to Provider A | `Agent.send` → adapter | Anthropic on recorded wire | Opus 5.5 via Claude Code login |
| 5 context with provenance | notes `by`, `record_claim` author, sealed per project | same | yes |
| 6–7 restart, recover | new process on the same store; `Agent.briefing` | same | yes, the real `pcim serve` killed and restarted |
| 8 resume with Provider B | new session from the briefing | OpenAI on recorded wire | Haiku 5.5 |
| 9 B gets only authorized context | `require_provider`, egress per vendor, disclosures | canary and A-only memory absent; withheld item logged | yes |
| 10 delegated verification | `Gateway.verify_claim`, `AgentMessage` | OpenAI verifier refutes A's claim | routed to Sonnet 5.5 (same vendor, labelled weaker independence) |
| 11 evidence and disagreement | `ClaimLedger` statuses, exchange log | claim `contested`, counter-claim kept | verdict "supported", status `corroborated`; no disagreement live (the claim was true) |
| 12–14 reversible action, approval, real result | write with backup and undo, re-check at execution, `run_tests`, claim settled by test | file changed, test exit 0, claim `verified` by test | yes: SHA-256 changed, `check.py` exit 0, claim verified by test |
| 15–16 revoke, prove blocked | `revoke_provider`; checks on every path | 403 on chat and verify, zero further requests, `egress.blocked` audited | yes |

## Phase status against the brief

| phase | state |
|---|---|
| 0 Audit | done; re-audited 2026-10-24 (`EXISTING_SYSTEM_AUDIT.md`) |
| 1 Foundation: auth, database, projects, one real model | done. The real model is reached through the Claude Code login, because no API key exists |
| 2 Persistence: durable context, memory inspection, cross-session recovery | done (earlier slice) |
| 3 Coordination: a second provider, scoped context exchange | done in code. OpenAI is wire-tested only. Live runs use two Anthropic models |
| 4 Verification: claim ledger, evidence, independent verification | done (this milestone) |
| 5 Execution: approvals, safe tools, cancellation, logs | done (earlier slice) |
| 6 Product: premium UI, autonomy controls, billing, metering | partly. The dashboard has the brief's palette, model access, claims and a model network view. **Billing is not started** (see below) |
| 7 Distribution: SDK and MCP | MCP server exists (read-mostly, now provider-gated). **TypeScript SDK not started** |
| 8 Evaluation | **not started.** The harness design is in `RESEARCH.md` (G1–G3). Nothing has been measured |

## Next milestones, in order

1. **A second vendor, live.** Set `OPENAI_API_KEY` and `PCIM_OPENAI_MODEL`, and re-run
   `scripts/milestone1_live.py` with Provider B on OpenAI. That turns "cross-model" into
   "cross-vendor" and gives verification real independence. A Gemini adapter with tool use
   is the third.
2. **Intent engine.** Structured goal extraction (objective, constraints, success criteria,
   open questions) from the first message, shown to the person to confirm before it is saved.
3. **Evaluation harness, G1a.** The brief's four arms (single model, plus persistent
   context, fixed multi-model, full CI) on a fixed task set, with blinded grading. No
   claims about CI's value until this has run.
4. **Billing.** Stripe test mode. This needs product and legal decisions first:
   - how student eligibility is verified;
   - senior verification without collecting identity documents;
   - for under-15 users: COPPA-style parental consent, restricted integrations, and no purchases;
   - per-plan compute caps, so free tiers cannot run up unbounded model spend.

   None of this should be coded before those decisions are written down.
5. **TypeScript SDK** over the HTTP API (`createProject`, `recordContext`, `retrieveContext`,
   `proposePlan`, `requestApproval`, `executeAuthorizedTool`, `cancelTask`, `listActivity`,
   `deleteMemory`), generated from the OpenAPI schema FastAPI already serves.

## The first vertical slice (2026-10-19)

**Target loop:** open CI → discuss a project → close CI → reopen it → resume with the
right context → approve an action → watch it run → inspect what changed. Every stage
ships with tests. A stage is done only when its tests pass.

| stage | objective | files | done when |
|---|---|---|---|
| 1 | Durable project, session and task state | `store.py` (4 tables), `workspace.py` | a project and its notes survive a new process on the same store |
| 2 | Governor | `governor.py` | rule table of `CI_ARCHITECTURE.md` covered by tests, including path escape, symlink escape, `$HOME` root, secrets, revocation, expiry and limits |
| 3 | Executor and queue | `tools.py`, `tasks.py` | read/search/run/write work inside a temp workspace; timeout and cancel kill the process; restart marks running tasks interrupted; approvals expire |
| 4 | Agent loop | `agent.py`, `adapters/anthropic.py` (tool use), `adapters/mock_agent.py` | with a recorded Anthropic transcript: read executes automatically, run_tests pauses for approval, approve resumes the loop, deny returns an error result |
| 5 | Autonomy modes and halt | `governor.py`, `workspace.py` | mode changes, a scoped grant and its limits, and halt are enforced server-side |
| 6 | API and dashboard | `api.py`, `dashboard/` | the target loop works over HTTP; dashboard has Workspace and Autonomy views on desktop and mobile |
| 7 | MCP server | `mcp_server.py` | an in-memory MCP client can list tools, read a briefing, fetch context and record a memory |
| 8 | Docs | `THREAT_MODEL.md`, `README.md` | the docs match what was built, and untested parts are marked as such |

**Risks this plan accepts:**
- No live provider test (no key).
- No OS-level sandbox.
- No streaming to the UI.
- One user per store.

Each risk is listed again in the README status table when the slice lands.
