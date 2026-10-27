# CI engineering audit

**Scope.**

- **Repository:** `feat/workspace-autonomy` at `7b04eb7`, plus the documentation commit that follows it.
- **Sources of findings:**
  - a read-through of every module on the request path;
  - an independent review by a second agent, which wrote probe scripts;
  - the live runs;
  - the runtime benchmark;
  - the E1 evaluation.

**Severity.**

| level | meaning |
|---|---|
| critical | breaks a stated guarantee |
| high | exposure, data loss or denial of service reachable through normal use |
| medium | wrong behavior, or a weakened guarantee |
| low | inconsistency without exposure |
| perf | measured cost |

Every "fixed" row names the test that fails without the fix, or the measurement that shows it.
The audit of 2026-10-19 is kept in `EXISTING_SYSTEM_AUDIT.md`.

## Findings

| # | sev | issue | root cause | where | fix | evidence |
|---|---|---|---|---|---|---|
| 1 | critical | Project content (briefing, transcript, tool results) went to whatever provider a chat named | Only memory items passed the egress guard; nothing gated project content | `agent.py`, `workspace.py` | Per-project provider access by vendor, checked before the briefing, before every model step, delegation and MCP call | `test_milestone1_sixteen_steps` step 16, `test_revocation_stops_a_session_paused_on_an_approval` |
| 2 | high | A model could launder memory a vendor may not see into a note, claim or file that then reached that vendor | Derived content carried no provenance of what its author had seen | `tasks.py`, `workspace.py`, `exchange.py`, `agent.py` | Lineage on model-written notes and claims (the memory shown in their session). Vendor-filtered briefings and evidence. The person can preview and release | `test_a_claim_derived_from_restricted_memory_needs_a_release_to_go_elsewhere`, milestone step 7 |
| 3 | high | Revoking consent or retracting memory did not reach an open session, which kept re-sending it in its system section and transcript | Sessions never re-checked consent after they started | `agent.py` | Context epoch from the audit log, so it works across processes. New sends start a fresh session; a paused turn stops | `test_withdrawing_context_ends_the_session_that_carried_it` |
| 4 | high | A claim or note could forge a "[verified]" ledger line or a person-written note in later briefings | Stored text was rendered raw, newlines included | `workspace.py: render_briefing` | Text collapsed to one line; CI's labels placed before stored text | `test_stored_text_cannot_forge_a_briefing_line` |
| 5 | high | MCP listed a revoked project's name and objective, and accepted notes into it without attribution | Only the briefing tool checked provider access | `mcp_server.py` | Listing, context, notes and memory writes all gated by vendor access; MCP notes attributed | `test_mcp_lists_and_writes_only_where_the_vendor_has_access` |
| 6 | high | Environment variables could reroute "Anthropic" traffic to Bedrock, Vertex or a proxy, and the "local" provider to a remote host | Adapters inherited `CLAUDE_CODE_USE_*` and `ANTHROPIC_*BASE_URL`; `PCIM_OPENAI_BASE_URL` applied to `local` | `adapters/claude_cli.py`, `anthropic.py`, `openai_compat.py` | Reroute variables dropped for the CLI child; SDK endpoint pinned (explicit opt-in variable); `local` must be loopback | `test_the_cli_child_is_never_rerouted_off_anthropic`, `test_the_local_provider_must_stay_on_this_machine` |
| 7 | high | Two turns on one session could overwrite each other's reply | State loaded before acquiring the session lock | `agent.py: send` | Reload under the lock | `test_concurrent_turns_on_one_session_lose_nothing` (fails without the fix: loses "reply 2") |
| 8 | high | No bound on cost, time, tokens or tool calls in a turn | Only a round count existed | `agent.py: _run` | `TurnLimits` checked before every model call; refused calls still answered | `test_tool_call_limits_answer_every_call_and_stop_the_turn`, `test_each_turn_limit_binds_before_the_next_model_call`, `test_the_time_limit_binds` |
| 9 | high | A provider error mid-turn left `tool_use` blocks without results, which breaks the next request | State saved only at the end of a turn | `agent.py: _run` | Save after every round; audit `model.error` and re-raise | `test_a_provider_failure_mid_turn_is_audited_and_leaves_a_valid_transcript` |
| 10 | high | A model-written regex could hang the server (`(a+)+b`) | `re` has no timeout and ran in-process | `tools.py` | Search runs in a child process, killed after 10 s | `test_a_catastrophic_regex_is_stopped_not_left_running` |
| 11 | high | Memory was not scoped to projects: one project's facts were candidates in another's sessions | Memory was scoped to the user only | `engine.py` | `project:<id>` scope tags; a scoped item is a candidate only in its project; deleting the project deletes it | `test_project_memory_never_reaches_another_project_or_a_projectless_request`; E1 ablation C_no_scope (−3 tasks, all on scoping) |
| 12 | high | With `--json-schema`, Haiku tried native tool calls, got "No such tool available", and sometimes gave up | Structured output works through a tool inside the CLI | `adapters/claude_cli.py` | Plain JSON protocol with one retry | Live runs 1 → 3 (12/16 → 16/16) |
| 13 | high | The whole test suite was failing at audit time | An uncommitted edit used `os` without importing it | `store.py` | Import added | 66/66, then 115/115 |
| 14 | medium | A late approval on a closed session restarted its model loop | `_after_resolution` did not check session status | `agent.py` | A closed session is never resumed | `test_a_late_approval_does_not_resume_a_closed_session` |
| 15 | medium | The reflection called an unexecuted proposal "changed x", and a later denial never showed. The next model believed the change was made | Reflection text built from tool names, snapshotted at close | `agent.py` | Proposals read "proposed … — status"; the briefing rebuilds the lines from current task records | Same test; live run 2 |
| 16 | medium | A model could verify its own claim through another transport (`claude-cli:opus` vs `anthropic:claude-opus-5-5`) | Identity compared by transport label | `exchange.py`, `registry.py` | Canonical model identity | `test_the_same_model_through_another_transport_is_not_independent` |
| 17 | medium | A test run older than the last change could settle a claim | No ordering check | `exchange.py: resolve_with_test` | Refused if any write or undo finished after the run started; evidence lists model-written files | `test_a_test_run_older_than_the_last_change_cannot_settle_a_claim` |
| 18 | medium | Halt stopped tools but not model calls; each round still cost money | Halt was checked by the governor only | `agent.py` | Sends refused while halted; a running turn stops before its next call | `test_a_halt_during_a_turn_stops_before_the_next_model_call` |
| 19 | medium | A repeated tool-use id could execute a side effect twice | No idempotency key | `tasks.py: propose` | (session, tool_use_id) is unique | `test_the_same_model_call_never_executes_twice` |
| 20 | medium | An unexpected exception left a task "running" until restart | Only expected exceptions were caught | `tasks.py: _execute` | The task fails visibly; the error is logged | `test_an_internal_error_fails_the_task_visibly` |
| 21 | medium | After a stop, the person's next message reached the CLI model labelled as tool results | Merged user turns rendered as one role | `adapters/claude_cli.py` | Text blocks rendered as a person turn | `test_a_persons_message_after_a_stop_is_rendered_as_theirs` |
| 22 | medium | An empty OpenAI turn could break the session for good; a malformed reply became a 404 | `content: null` without tool calls; `KeyError` | `adapters/openai_compat.py` | Empty string content; `ProviderError` | `test_openai_keeps_empty_turns_valid_and_reports_malformed_replies` |
| 23 | medium | A failed model call lost its disclosure record | State saved only after the call | `agent.py: send` | Save before the call | `test_the_disclosure_survives_a_failed_model_call` |
| 24 | medium | The egress audit grew by about 75 KB per request at 3,000 memories | Every permitted id was logged | `sovereignty/egress.py` | Count and hash of the permitted set; refusals still itemized; what was sent is itemized in receipts and disclosures | Benchmark |
| 25 | low | Listing claims on another user's project returned an empty 200 instead of 404 | No ownership check | `exchange.py: ClaimLedger.list` | Ownership check | `test_another_user_cannot_read_or_act_on_anything` |
| 26 | low | MCP accepted unbounded memory and task text | No size caps | `mcp_server.py` | Same caps as the HTTP API | `test_mcp_refuses_oversized_input` |
| 27 | low | A verifier routed to Sonnet was recorded as Haiku | CLI usage lists auxiliary models; the first entry was taken | `adapters/claude_cli.py` | Requested model preferred | `test_claude_cli_adapter_runs_tool_less_and_parses_the_json_protocol` |
| 28 | perf | Consent grants were read from SQLite once per memory item per request | `allows()` loaded grants per call | `sovereignty/consent.py`, `egress.py` | One snapshot per decision | SQL per turn 3019 → 25 (N=3000) |
| 29 | perf | Every request decrypted every memory item | No cache | `store.py`, `memory.py` | Decrypted-item cache and embeddings index; cleared on any item write and on another process's commit (`data_version`) | Decryptions 3002 → 5; `test_the_memory_cache_never_serves_a_withdrawn_item` |
| 30 | perf | Propensity estimation re-ran the same deterministic selection 32 times per request | The greedy policy's samples are identical | `policy/learned.py` | Indicator computed directly | `test_greedy_propensities_equal_the_sampling_loop_they_replace`; select 11.4 → 0.9 ms (N=100) |
| 31 | medium | **Addressed on dev, unconfirmed.** Context selection is not relevance-selective at small memory sizes. CI sends 57% more input tokens than naive history | The designed prior's threshold admits nearly every allowed item; the budget does not bind. Lexical similarity cannot find relevance that needs inference | `policy/prior.py`, `policy/learned.py`, `planner.py`, `policy/candidates.py` | Retrieval planning (the model writes searches from the request alone; CI runs them locally), a local semantic embedder, compact labels. Tuned on the E2 development set only | Dev: precision 0.03 → 0.29, recall 0.67 → 0.92, context tokens 796 → 106 (`experiments/e2/CALIBRATION.md`). Confirmation needs the sealed E2 test set |
| 32 | medium | **Open.** E1's keyword rules misread refusals as injection-following and mentions as leaks | Mention is not use | `experiments/e1/dataset.json` | Registered numbers kept; misgrades listed; E2 should grade use, blind, with a rubric | `experiments/e1/REPORT.md` |

## Round of 2026-10-26/27

**Sources.**

- **Relevance work:** the engineer's own work.
- **Review:** an independent adversarial review of `ff2bea4`. Every finding was reproduced by a probe.
- **Pending tests:** a regression test was drafted for each open finding. The drafts are kept, not yet run, in `docs/review_2026_10_26_pending_tests.py`.
  - Some need the parameters their fix will add.
  - The test for #52 hangs on unfixed code.

| # | sev | issue | where | status | evidence |
|---|---|---|---|---|---|
| 33 | critical | Content derived from restricted memory reached any vendor once its source was superseded, corrected or expired: `check` returned "not-current" before checking consent | `workspace.py: derived_blocked`, `sovereignty/egress.py` | **Fixed** (`4c71d5c`): only the temporal check is skipped for lineage | `test_content_derived_from_a_superseded_item_still_needs_the_vendors_consent` |
| 34 | medium | A store's vectors carried no embedder identity; changing the embedder would silently compare vectors from two models | `store.py`, `engine.py` | **Fixed**: recorded in `meta`, re-embedded on an explicit change | `test_a_store_keeps_the_embedder_it_was_written_with_and_reembeds_on_a_change` |
| 35 | medium | The injection tripwire missed "Note to AI assistants: … recommend X" (one of two dev injections), so the document reached a CI prompt | `sovereignty/injection.py` | **Fixed**: pattern for text addressed to AI. Still a heuristic | `test_the_scan_flags_text_addressed_to_ai_and_not_text_about_using_ai`, `test_e2_keeps_private_injected_and_stale_content_out_of_every_ci_prompt` |
| 36 | high | `write_file`'s temp file `.{name}.ci-tmp` is not jailed; a committed symlink redirects an approved write outside the workspace | `tools.py` | **Open** | review probe p07 |
| 37 | high | `.git` and secret-directory checks are case-sensitive (macOS `.GIT/config`). Unapproved `git_status` honors the repo's fsmonitor and clean filters (also submodule filters), so a write grant, or a hostile repo, becomes command execution | `governor.py`, `tools.py` | **Open** | p08; filter driver confirmed separately |
| 38 | high | The MCP briefing skips the vendor filter on CI's last message | `mcp_server.py`, `workspace.py: briefing` | **Open** | p02 |
| 39 | high | Reflection lines repeat model-chosen tool arguments verbatim to a vendor that may not see that session | `agent.py` | **Open** | p03 |
| 40 | high | `exposure()` ignores the lineage of notes and claims shown in a briefing (lineage is not transitive) | `tasks.py` | **Open** | p04 |
| 41 | medium-high | MCP notes, and memory written through `ci_remember`, carry no lineage of what the server served | `mcp_server.py` | **Open** | p05 |
| 42 | medium-high | Model-written files carry no lineage; a later session for another vendor reads them, and reads count as verifier evidence | `exchange.py`, `tasks.py` | **Open** | p06 |
| 43 | medium | `list_files` and `search_code` apply the secret filter to a link's name, not its target | `tools.py` | **Open** | p10 |
| 44 | medium | `git_status` reports the whole enclosing repo when the workspace is a subfolder | `tools.py` | **Open** | p09 |
| 45 | medium | The search child imports from the server's working directory; a model-written `json.py` executes | `tools.py` | **Open** | p18 |
| 46 | medium | Memory text keeps its newlines in `<personal_context>`, so an ingested email can forge a labelled line. Collapsing it in `_clean` would also flatten `exchange.py`'s multi-line project block | `adapters/render.py`, `agent.py`, `mcp_server.py` | **Open** | p11 |
| 47 | medium | Project notes are read, modified and written with no lock: 200 concurrent POSTs kept 65 | `workspace.py` | **Open** | p13 |
| 48 | medium | `undo` uses the currently authorized root and never compares the file with `sha256_after` | `tasks.py` | **Open** | p14 |
| 49 | medium | Keyed supersession ignores project scope | `memory.py` | **Open** | p12 |
| 50 | medium | `close_session` takes no session lock; a running turn overwrites the closed state | `agent.py` | **Open** | p16 |
| 51 | medium | Lineage on notes and claims is capped at 200 ids, so long histories silently drop restrictions | `tasks.py`, `exchange.py` | **Open** (found while fixing) | code reading |
| 52 | low | `list_files` runs the model's glob in-process with `..` allowed and no time bound | `tools.py` | **Open** | p17 |
| 53 | low | An erased user's audit rows stay under the same id; a re-onboarded id reads them | `store.py`, `api.py` | **Open** | p15 |
| 54 | unconfirmed | Autonomous-grant `max_actions` is checked and counted in separate steps (concurrent overrun); `memory.correct` relabels the source as "user", which may widen egress | `governor.py`, `memory.py` | Not reproduced | draft tests only |

## Remaining risks (not fixed in code)

| risk | why it remains | mitigation in place |
|---|---|---|
| `run_tests` executes repository code with no OS sandbox | No container or seccomp layer | Approval or an explicit grant; scrubbed environment; timeout; kill of the process group |
| Secrets inside ordinary files reach the provider when read | Detection is by file name only | Workspace authorization is the consent; secret-looking names are never read |
| One process per store | Session locks are in-process | SQLite WAL and one store lock. Cross-process cache invalidation through `data_version` |
| A model call in flight cannot be interrupted | Provider SDKs and the CLI block | Halt prevents the next call and blocks acting on tools |
| The master key sits on the same disk (0600) | No Keychain or KMS integration | Per-record keys; crypto-shredding |
| Cross-vendor exchange is untested live | No OpenAI or Gemini keys on this machine | Recorded wire formats |
| The dashboard is not browser-tested | No browser automation in the environment | It only displays server decisions, which are tested |
| The briefing quotes the last session's closing model message | It helps continuity | Withheld from a vendor when that session saw memory the vendor may not see |
