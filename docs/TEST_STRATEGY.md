# Test strategy

A claim in this repository counts as "working" only if a test exercises it, and as "live"
only if a run against a real model produced it. This page says which kind of evidence
backs each part, how to run it, and what is deliberately not tested.

## Four layers of evidence

| layer | what it proves | where | runs |
|---|---|---|---|
| **Unit and contract tests** | One component does what it says, including its refusals: governor rules, path jail, consent, egress, encryption, ledger status rules, message schema, adapter request shapes | `tests/test_*.py` | every change, about 5 s, no network |
| **Recorded wire formats** | An adapter builds the request the provider documents and parses the documented response, including tool calls | Anthropic through the official SDK on a mock transport; OpenAI Chat Completions on `httpx.MockTransport`; the Claude CLI through a fake `claude` binary that logs its argv, stdin, cwd and environment | every change |
| **The milestone scenario** | The sixteen steps of the build brief hold together over HTTP, across a real restart, with real file edits and test runs | `tests/test_milestone1_demo.py` | every change |
| **Live runs** | Real models follow CI's protocols and the loop works with them | `scripts/milestone1_live.py`, via the local Claude Code login | by hand. Each run spends real model usage |

A test double is never presented as a model. The mock agent prefixes every reply with
`[mock agent — no model connected]`, and routing never picks it unless asked to.

## What each security property is tested by

Every "controlled" row in `CI_SECURITY_MODEL.md` names its test, and every fixed finding in
`CI_ENGINEERING_AUDIT.md` names the test that fails without the fix. The test files, by
concern:

| file | concern |
|---|---|
| `test_runtime.py` | turn limits, durability, concurrency (the race test fails without the fix), idempotency, accounting, cache safety, equivalence of the propensity shortcut |
| `test_isolation.py` | another user (33 endpoints), authentication on every route, project memory scope, the token at rest |
| `test_hardening.py` | one test per finding of the independent review, plus the regex timeout and MCP input bounds |
| `test_exchange_and_providers.py` | provider access, the gateway, the ledger, routing, both bridge adapters |
| `test_milestone1_demo.py` | the sixteen steps over HTTP across a restart |
| `test_e1_harness.py` | the evaluation's prompt-level guarantees, grader and statistics |

A fifth layer sits beside these: **measurements.** `scripts/bench_runtime.py` records CI's
own overhead, and `pcim.observatory.e1` the preregistered evaluation. Neither is a pass/fail
test. Both are committed with their raw output, so a change can be compared against them.

The tests added with the exchange work:

- **Provider access is enforced before content leaves:**
  - `test_milestone1_sixteen_steps`, step 16: chat and verification both return 403, and no request reaches the provider's transport.
  - `test_revocation_stops_a_session_paused_on_an_approval`: an approval does not resume a revoked session.
  - `test_mcp_briefing_respects_project_provider_access`.
- **Provider B sees only what it may see:**
  - Step 9 of the milestone test: no canary from A's transcript reaches B, the A-only memory item is withheld, and the disclosure log names it.
  - `test_verification_sends_only_what_the_recipient_may_see`: it covers cross-project tasks, memory without an egress grant, and unknown references.
- **Model agreement is not verification:** `test_model_agreement_is_never_verification`.
- **A claim is settled only by a test or the person:**
  - `test_a_failing_linked_test_refutes_and_only_run_tests_tasks_count`.
  - Step 14 of the milestone test.
- **The CLI bridge can neither execute anything nor leak the prompt to `ps`:** `test_claude_cli_adapter_runs_tool_less_and_parses_the_json_protocol`. It checks that the CLI's tools are off, there are no MCP servers and no settings or hooks, the system prompt file is 0600 and deleted, there is no IDE attach variable, and the working directory is a sandbox.

## Live-run protocol

`scripts/milestone1_live.py` refuses a non-empty `--home`, uses synthetic project content
only, and acts as the person:

- it approves only the write to `src/sharpe.py` and the test run;
- it denies anything else;
- it denies work left over from a closed session.

It writes `report.json` with every model reply, tool call, routing decision, disclosure
and cost, and does not edit any of them. A step that does not go to plan is recorded as
failed, with the model's own words.

Live runs are not a benchmark. They show the loop works with real models on one scenario.
They say nothing about quality relative to a baseline (see below).

## Not tested, on purpose or for now

- **Cross-vendor live exchange.** There are no OpenAI, Gemini or Anthropic API keys on this machine, so the OpenAI adapter is wire-tested only.
- **Concurrency:** two browsers approving the same task, or a revocation racing a model step. The per-session lock covers the agent loop. Nothing covers the HTTP layer.
- **An OS sandbox for `run_tests`.** There is none, so there is nothing to test (CI_SECURITY_MODEL, accepted risks).
- **The dashboard.** No browser tests. It only displays server decisions, and every rule it shows is enforced and tested server-side.
- **Comparative evaluation.** Whether full CI beats a strong single-model assistant (arms A to D of the brief's §11) needs a pre-registered study with blinded grading and real tasks. `docs/RESEARCH.md` has the design (G1–G3). None of it has been run.

## Running

```bash
.venv/bin/python -m pytest -q                                   # everything offline
.venv/bin/python -m pytest -q tests/test_milestone1_demo.py      # the sixteen steps
.venv/bin/python scripts/milestone1_live.py --home /tmp/ci-live --out /tmp/ci-live/report   # live; spends usage
```
