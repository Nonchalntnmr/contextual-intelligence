# CI security model

**Scope.** CI as built:

- a local server on 127.0.0.1;
- one encrypted SQLite store in `$PCIM_HOME`;
- one or more people, each with their own bearer token;
- model calls to several vendors:
  - Anthropic, through the API or the local `claude` CLI login;
  - OpenAI, wire-tested only;
  - a mock;
- governed tools on folders the person authorized;
- model-to-model delegation through the exchange gateway.

This document says "controlled" only where code enforces a rule *and* a named test exercises
it. Model instructions are never counted as a control. Everything below is enforced at the
application boundary.

## Assets

- the person's context: memory, projects, transcripts, claims;
- the authorized workspaces and the files in them;
- provider credentials and the person's Claude login;
- the master key;
- the integrity of the audit trail;
- the person's money: model spend.

## Trust boundaries

1. **Browser, CLI or MCP client → API.** Bearer token per user, stored hashed. Bound to localhost by default.
2. **CI → vendor.** Everything sent crosses the vendor's boundary.
3. **Model → tools.** The model is untrusted. Anything it proposes passes the governor.
4. **Workspace content → model.** Files, documents and tool output are untrusted input.
5. **Tests → machine.** `run_tests` executes repository code, so it is code execution.
6. **Model → model.** A verifier sees only what the gateway resolves for it. Its reply is untrusted.
7. **CI → `claude` CLI subprocess.** It runs with the person's Claude login. Its tools, MCP servers, settings and hooks are off, and it is never rerouted off Anthropic.

## Threats and controls

### Exposure of context

| threat | control | status |
|---|---|---|
| Project content reaches a vendor the person never chose, or revoked | Per-project provider access by vendor, checked before the briefing, before every model step, before every delegation, and on every MCP call that touches a project; refusals audited as `egress.blocked` | controlled (`test_milestone1_sixteen_steps` step 16, `test_revocation_stops_a_session_paused_on_an_approval`, `test_mcp_lists_and_writes_only_where_the_vendor_has_access`) |
| A memory item reaches a vendor without consent | Egress guard per item: SECRET never leaves; superseded, retracted or expired items never go; text flagged as instructions is held until confirmed; every source in the item's lineage needs an egress grant at or above its sensitivity | controlled (`test_memory_and_sovereignty.py`; E1 prompt-level: 0% of CI prompts contained private or injected content) |
| Memory a vendor may not see is laundered through a model-written note, claim or file | Lineage on model-written notes and claims (the memory their session was shown). Vendor-filtered briefings, evidence and closing messages. The person previews and releases | controlled (`test_a_claim_derived_from_restricted_memory_needs_a_release_to_go_elsewhere`, milestone step 7) |
| Consent is revoked but an open session keeps sending the old context | Context epoch: any revocation, retraction, deletion, unpin or correction ends the session that carried it, and stops a paused turn, across processes | controlled (`test_withdrawing_context_ends_the_session_that_carried_it`) |
| One project's memory contaminates another, or a request with no project | Scope tags; candidates filtered by `in_scope` before egress and selection | controlled (`test_project_memory_never_reaches_another_project_or_a_projectless_request`) |
| Another user on the same server reads or acts on someone's data | Every row carries the user id, and every read filters by it; project rows are sealed with the project key | controlled (`test_another_user_cannot_read_or_act_on_anything`: 33 endpoints return 404, and no id appears in any listing) |
| Provider B receives Provider A's transcript | B starts from a compiled briefing; disclosures record exactly what B was sent | controlled (milestone steps 8–9). **Not controlled:** the briefing quotes A's last message when allowed, so anything A repeated from the person's own words goes with it |
| A verifier sees evidence it may not see | Refs resolved for the recipient: this project only; memory through egress; derived content through lineage; withheld refs recorded | controlled (`test_verification_sends_only_what_the_recipient_may_see`) |
| Environment variables reroute "Anthropic" traffic elsewhere | CLI child drops `CLAUDE_CODE_USE_*` and `ANTHROPIC_*BASE_URL`; SDK endpoint pinned; `local` must be loopback | controlled (`test_the_cli_child_is_never_rerouted_off_anthropic`, `test_the_local_provider_must_stay_on_this_machine`) |
| Secrets in the workspace reach the provider | Secret-looking names are never read, listed or searched | controlled by name only (`test_read_list_search_and_secrets_never_listed`). **Not controlled:** a secret pasted into an ordinary file |
| The token is stored or logged in clear | Stored as a hash; audit payloads limited to ids, counts and hashes | controlled (`test_the_token_is_never_stored_or_logged_in_clear`) |
| Data at rest is read from disk | Per-record AES-GCM, crypto-shredding, `secure_delete`, WAL truncation, database file 0600 | controlled (`test_plaintext_never_reaches_the_database_file`, `test_delete_destroys_the_key_and_the_ciphertext_cannot_be_opened`). **Not controlled:** the master key is on the same disk (0600) |

### Integrity of actions and records

| threat | control | status |
|---|---|---|
| Prompt injection in a file makes the model act against the person | Every run or write needs approval or a narrow, expiring grant; content is fenced as data; no tool sends, publishes or spends | controlled for actions (`test_governor_rule_table`). **Not controlled:** a wrong *answer* |
| Injection plants a lasting preference | Untrusted text is stored as a fact with a proposed kind until confirmed | controlled (`test_untrusted_text_cannot_become_a_standing_preference`) |
| A claim or note forges a ledger line or a person-written note for later models | Stored text collapsed to one line; CI's labels first | controlled (`test_stored_text_cannot_forge_a_briefing_line`) |
| Consensus mistaken for truth | Model agreement gives at most `corroborated`; only a linked fresh test or the person settles a claim | controlled (`test_model_agreement_is_never_verification`) |
| A model verifies its own claim | Canonical model identity, so the same model through two transports is one | controlled (`test_the_same_model_through_another_transport_is_not_independent`) |
| A stale test settles a claim | Refused if the workspace changed after the run started | controlled (`test_a_test_run_older_than_the_last_change_cannot_settle_a_claim`) |
| A tool reads or writes outside the workspace | `jail()`; `$HOME`, `/` and ancestors cannot be roots; `.git` never written | controlled (`test_jail_blocks_traversal_absolute_symlink_secrets_and_git_writes`) |
| An autonomous task widens its own permissions | Only the person creates grants; checked on every call and again at execution; immediate revocation | controlled (`test_autonomous_grant_limits_revocation_and_expiry`, `test_revocation_after_approval_blocks_at_execution`) |
| A side effect runs twice | (session, tool_use_id) is unique | controlled (`test_the_same_model_call_never_executes_twice`) |
| Concurrent turns corrupt a transcript | Reload under the session lock | controlled (`test_concurrent_turns_on_one_session_lose_nothing`) |
| A forged tool result enters the transcript | Only CI writes `tool_result` blocks, from task records; CLI turns are nonce-delimited | controlled by construction; the CLI nonce lowers the rate of text-level forgery but does not bound it |
| Audit tampering | Hash chain, verified on read | evidence, not prevention (`test_audit_chain_detects_tampering`) |

### Availability and cost

| threat | control | status |
|---|---|---|
| A turn loops or overspends | `TurnLimits`: rounds, tool calls per step and per turn, seconds, cost, input tokens; checked before every model call | controlled (`test_each_turn_limit_binds_before_the_next_model_call` and siblings). The cost limit binds only for priced models |
| A model-written regex hangs the server | Search in a child process, killed at 10 s | controlled (`test_a_catastrophic_regex_is_stopped_not_left_running`) |
| Runaway tests | Timeout; cancel kills the process group; halt | controlled (`test_timeout_and_cancel_kill_the_process`, `test_halt_cancels_and_blocks_until_resume`) |
| Halt does not stop spending | Sends refused while halted; a running turn stops before its next model call | controlled (`test_a_halt_during_a_turn_stops_before_the_next_model_call`) |
| Oversized input | Size limits on every HTTP body and MCP text argument | controlled (`test_mcp_refuses_oversized_input`; pydantic limits) |

## Consent enforcement, in one place

| what | where it is decided | when |
|---|---|---|
| may CI store items from this source? | `ConsentLedger.allows(store)` in `memory.remember` | at write |
| may this vendor see this item? | `EgressGuard.check` (sensitivity, currency, instruction hold, lineage grants) | at every selection, against one grants snapshot |
| may this vendor see this project? | `Workspace.require_provider` | before every step and delegation |
| may this vendor see this derived note or claim? | `Workspace.derived_blocked` (its lineage, through the egress guard) | at every briefing and every resolution of evidence |
| does an open session still match current consent? | the context epoch | before every send and every model step |

Revocation takes effect at the next decision point. There is no window in which a revoked
grant still lets content out.

## MCP security

- **Identity.** `pcim-mcp` runs as the local user, reading the token from `$PCIM_HOME`. Any process running as that OS user can do the same.
- **Vendor.** `$PCIM_MCP_PROVIDER` names the vendor the calling agent forwards content to. The default is Anthropic.
  - Project listing, briefing, project-scoped context and project writes require that vendor's access to the project.
  - Personal context passes the egress guard for that vendor.
- **Writes.** Memories from an agent are stored as `agent:mcp`, trust `imported`, and need a store grant the person gives once. They can never become standing preferences without confirmation. Notes are attributed to `mcp:<vendor>`.
- **Inputs.** Validated by the SDK's typed signatures, and length-bounded as on the HTTP API.
- **Execution.** CI executes nothing on an MCP client's behalf. The client's own tools stay under its own permission system.

## Data lifecycle

| stage | rule |
|---|---|
| creation | Memory through `memory.remember`: consent check, injection scan, standing-kind gating, supersession or conflict, scope tags. Project content through `Workspace`. Model-written content records its provider, model, session and lineage |
| retention | Nothing expires silently, except goals (180-day valid time) and inferred items (confidence decays with a half-life). Approvals expire after an hour. Idle sessions close after two hours |
| modification | Content is immutable. A correction or supersession closes the old row's system time and links to the new one. Retraction is reversible. All of these end open sessions that carried the old version |
| deletion | `memory.delete` destroys the item's key and cascades through lineage. `delete_project` destroys the project key (transcripts, tasks, backups, claims, exchange records) and deletes memory scoped to the project. `erase` destroys every key the user has. `secure_delete` is on and the WAL is truncated after deletions |
| caches | The decrypted-item cache and the embeddings index are cleared on any item write in this process, and whenever another process commits (`data_version`). Unwrapped keys are dropped when their key is destroyed. Nothing is cached across restarts |
| what deletion does not reach | 1. What was already sent to a vendor (their retention terms apply; disclosures record what went where). 2. The audit log, which keeps ids, counts and hashes of what was erased as the record of the erasure. 3. Backups or copies of `$PCIM_HOME` made outside CI. 4. A derived note whose source was deleted, which is withheld from every vendor and stays visible to the person for review |

## Accepted risks

- **No OS sandbox for `run_tests`.** Approving a test run means trusting that repository's code as much as running it yourself.
- **Live calls bill the person's Claude account.** They go through the `claude` CLI login, under that account's terms. Anthropic API and OpenAI paths are wire-tested only.
- **New projects start with the onboarding providers.** This is an explicit, visible, revocable default; `providers: []` creates a project with none.
- **One process per store.** Session locks are in-process.
- **Fencing and nonces lower the rate of injection, but do not bound it.** No answer is trusted for an action without the governor.
