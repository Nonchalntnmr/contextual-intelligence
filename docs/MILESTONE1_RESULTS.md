# Milestone 1 results (2026-10-24)

## Summary

- **Offline:** the sixteen-step demonstration passes over HTTP with a real restart. Provider A is Anthropic and Provider B is OpenAI, both on recorded wire formats.
- **Live:** it passed 16 of 16 steps on the third run against real models. Provider A was Claude Opus 5.5 and Provider B was Claude Haiku 5.5, both through the local Claude Code login, with a verifier routed to Claude Sonnet 5.5.
- **What the first two live runs found:** two real defects in CI and one bug in the script. All three were fixed before the third run.
- **Limit:** every live model was Anthropic. Exchange across vendors is shown only on recorded wire formats.

## How the live runs were set up

- **Script:** `scripts/milestone1_live.py` with a fresh `--home`.
- **Server:** the real `pcim serve` on 127.0.0.1, killed after step 6 and started again on the same encrypted store.
- **Project:** synthetic, a two-file toy repository. `src/sharpe.py` annualises with √365. `check.py` expects √252.
- **The person:** the script, which approves only the write to `src/sharpe.py` and the test run.
- **Models:** reached through `claude -p`, with its tools, MCP servers, settings and hooks off.

## Run 1: 12 of 16

- **Step 4, CI fine, script check wrong.** Opus did more than asked: it read both files, recorded two claims, and proposed the fix in the same turn. The script expected the turn to end without a pending approval.
- **Steps 12–14, a real defect in the adapter.** Haiku, as Provider B, tried to call CI's tools natively inside the CLI, got "No such tool available," and told the person the tools were broken. The cause was `--json-schema`: structured output runs through a tool inside the CLI, so a tool-calling mode was active.
  - A/B check: with the plain JSON protocol and no tool in the request, Haiku and Sonnet both answered correctly on the first turn.
  - The adapter was rewritten to use the plain protocol. A non-JSON reply gets one retry, then passes through as text with no tool calls.
- **Mislabelled verifier, a real defect.** The verifier was routed to Sonnet but recorded as Haiku. The CLI's `modelUsage` also lists its own auxiliary Haiku calls, and the adapter took the first entry. It now takes the model that was requested.

## Run 2: 12 of 16

- **Steps 8 and 12–14, a script bug.** Haiku now followed the protocol. While resuming, it read both files and proposed the fix unprompted. The script then sent its step-12 message into a session waiting on that approval, got 409, and stopped.
- **The reflection described a proposal as a change, a real defect.** It listed Opus's unexecuted write as "changed src/sharpe.py — awaiting_approval," and Haiku believed the change had been made. The reflection was also a snapshot taken when the session closed, so a later denial never showed.
  - Proposals now read as "proposed a change to …", with their status.
  - The briefing rebuilds the last-session lines from current task records.
- **Late approvals, a real defect found reading the logs.** Approving a task left over from a closed session would have restarted that session's model loop. It now settles the task only.

## Run 3: 16 of 16, in 85.4 s

Raw record: `experiments/milestone1/live-run3-2026-10-24.json`. It holds every reply, tool
call, routing decision, disclosure and cost.

| step | what happened |
|---|---|
| 4 | Opus 5.5 read `src/sharpe.py`, put the √365 finding on the claim ledger citing the read, and recorded the decision note |
| 7 | after the restart, decisions and claims came back identical |
| 8 | Haiku 5.5 resumed from the briefing. It re-read both files to check the claims, and recorded its own claims and notes. It made two malformed `record_note` calls (arguments outside `"input"`), which CI's validator denied; it retried correctly. The adapter now also accepts that slip, still validated |
| 9 | the canary from the person's message to Opus was not in Haiku's briefing. The Anthropic-unconsented memory item was withheld (`no-egress-grant:ingest:fund`), and the disclosure log names it |
| 10 | routing picked Sonnet 5.5: "same vendor as the author (anthropic), different model: weaker independence." No other vendor was available |
| 11 | Sonnet's verdict was **supported**, so the status is **corroborated**. The claim was true. Sonnet also said what would settle the separate question of whether √365 is right. **No disagreement occurred live**; the disagreement path is shown offline, where the OpenAI verifier refutes a wrong claim and the status becomes `contested` |
| 12–14 | asked for the change, Haiku proposed the write to `src/sharpe.py`. The person approved it, and the file's SHA-256 changed. Haiku proposed `run_tests`. The person approved it: exit 0, `1 passed`. A claim by the person, "check.py passes after the change," was settled `verified` **by the test** |
| 15–16 | Anthropic access was revoked. Chat and verification both returned 403, no new disclosure was written, two `egress.blocked` events were audited, and the hash chain was intact |

**Cost.** The list-price figures the CLI reported for the Opus turn, the first Haiku turn and
the verification add up to about **$0.13**. The write-and-test turns were not summed. Under
a Claude subscription these figures are notional.

## What this does not show

- **Exchange across vendors, live.** Needs `OPENAI_API_KEY` and `PCIM_OPENAI_MODEL`.
- **That CI helps.** One scenario, run once to completion, is a demonstration, not an
  evaluation. The comparative study (the brief's §11) has not been run.
- **Model disagreement on a live claim.** The claim Opus made was true, and the verifier agreed.
