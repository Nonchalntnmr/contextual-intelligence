# Threat model: Personal CI with workspace execution

**Scope.** One person, one machine, a local server on 127.0.0.1, an encrypted SQLite store
in `$PCIM_HOME`, model calls to several providers, and tools that act on folders the person
authorized. The providers are:

- Anthropic, through the API or through the local `claude` CLI login;
- OpenAI, wire-tested only;
- a mock agent.

CI delegates work from one model to another through the exchange gateway.

This document says "controlled" only where code enforces it *and* a test exercises it. It
names the test.

## Assets

- the person's context: memory, projects, transcripts;
- the authorized workspaces and the files in them;
- provider API keys;
- the master key;
- the integrity of the audit trail.

## Trust boundaries

1. **Browser / CLI / MCP client → API.** Bearer token per user, stored hashed. Bound to
   localhost by default.
2. **CI → model provider.** Everything sent crosses the provider's boundary.
3. **Model → tools.** The model is treated as untrusted: anything it proposes passes the
   governor.
4. **Workspace content → model.** Repository files, documents and tool output are
   untrusted input.
5. **Tests → machine.** `run_tests` executes repository code, so it is code execution.
6. **Model → model.** A verifier sees only what the gateway resolves for it. Its reply is
   untrusted input, like any model output.
7. **CI → `claude` CLI subprocess.** It runs with the person's Claude login. The CLI's own
   tools, MCP servers, settings and hooks are switched off.

## Threats and controls

| threat | control | status |
|---|---|---|
| Prompt injection in a file makes the model act against the person | Every run or write needs approval unless a narrow, expiring grant covers it. Tool results and the briefing are fenced as data. No tool sends, publishes, deletes outside the workspace, or spends. | controlled for actions (`test_governor_rule_table`, `test_agent_reads_automatically_and_pauses_for_tests`). **Not controlled:** the model can still be misled into a wrong *answer*. |
| Injection plants a lasting "preference" | Memories from documents or agents are stored as facts with a proposed kind until the person confirms | controlled (`test_untrusted_text_cannot_become_a_standing_preference`, `test_mcp_context_is_consent_filtered_and_agent_memories_are_gated`) |
| A tool reads or writes outside the workspace | `jail()`: resolve with symlinks, refuse traversal, absolute paths and symlink escapes; `$HOME`, `/` and ancestors cannot be roots | controlled (`test_jail_blocks_traversal_absolute_symlink_secrets_and_git_writes`, `test_roots_that_are_too_broad_are_refused`) |
| Secrets in the workspace reach the provider | Secret-looking names are never read, listed or searched (`.env*`, `*.key`, `*.pem`, `id_*`, `*credentials*`, `*.db`, `.ssh/`, `.aws/` …) | controlled by name only (`test_read_list_search_and_secrets_never_listed`). **Not controlled:** secrets inside ordinary files, such as a key pasted into `config.py`. |
| Test code exfiltrates keys from the environment | `run_tests` gets an environment without `*KEY*`, `*TOKEN*`, `*SECRET*`, provider and cloud prefixes | controlled (`test_run_tests_scrubs_secrets_from_the_environment`). **Not controlled:** test code can still read files and use the network, because there is no OS sandbox. |
| Runaway or hung execution | Per-workspace timeout; cancel kills the whole process group; halt cancels everything and blocks execution until resume | controlled (`test_timeout_and_cancel_kill_the_process`, `test_halt_cancels_and_blocks_until_resume`) |
| An autonomous task widens its own permissions | Grants are created only by the person (API/UI), checked on every call and again at execution; limits on actions, duration and expiry; revocation is immediate | controlled (`test_autonomous_grant_limits_revocation_and_expiry`, `test_revocation_after_approval_blocks_at_execution`) |
| A stale approval fires after things changed | Approvals expire after 1 hour; execution re-checks halt, workspace and path; a crash never re-runs a running task | controlled (`test_approval_expires_and_restart_never_reruns`) |
| git publishes the home directory (a real hazard on the author's machine) | No git write tool exists; `git_status` refuses when git's actual top level is `$HOME` | controlled (`test_git_status_refuses_the_home_directory_repository`) |
| Cross-project or cross-user leakage | Every row is keyed by user; project content is sealed with a project key; tasks resolve their project from the row, never from the request | controlled by construction (single user per store in practice); **not tested** with concurrent users |
| Data at rest is read from disk | Per-record AES-GCM, crypto-shredding on delete, `secure_delete`, WAL truncation | controlled (`test_plaintext_never_reaches_the_database_file`, `test_delete_destroys_the_key_and_the_ciphertext_cannot_be_opened`). **Not controlled:** the master key sits on the same disk (0600). |
| Someone else on the machine calls the API | Bearer token required; onboarding closes after the first user | partly: any process running as the same OS user can read `$PCIM_HOME/token` |
| Provider keys reach the browser | Keys are read server-side from the environment only; no endpoint returns them | controlled by construction; **not tested** |
| Audit tampering | Hash chain; verification on every audit read | evidence, not prevention (`test_audit_chain_detects_tampering`) |
| Project content reaches a provider the person never chose, or one they revoked | Per-project provider access (by vendor), checked before the briefing is built, again before every model step, before every delegation, and before an MCP briefing; refusals are audited as `egress.blocked` | controlled (`test_milestone1_sixteen_steps` step 16, `test_revocation_stops_a_session_paused_on_an_approval`, `test_mcp_briefing_respects_project_provider_access`) |
| Provider B receives Provider A's transcript or A-only context | B starts from a compiled briefing, never A's transcript; memory passes the egress guard per vendor; every message logs a disclosure (briefing hash and text, memory sent and withheld) | controlled (`test_milestone1_sixteen_steps` steps 8–9). **Not controlled:** the briefing quotes A's last message (labelled model-written), so anything A repeated from the person goes with it |
| A verifier is shown evidence it may not see (another project's files, unconsented memory) | Each evidence reference is resolved for the recipient: this project's tasks, notes and claims only; memory through the egress guard; withheld refs are recorded with a reason | controlled (`test_verification_sends_only_what_the_recipient_may_see`) |
| Consensus mistaken for truth: two models agreeing marks a wrong claim "verified" | Status rules: model agreement gives at most `corroborated`; only a linked `run_tests` result or the person's decision gives `verified`/`refuted`; every assessment is kept | controlled (`test_model_agreement_is_never_verification`, `test_a_failing_linked_test_refutes_and_only_run_tests_tasks_count`) |
| A model verifies its own claim | Refused when the verifier is the author's provider and model; routing skips the author and prefers another vendor, and labels same-vendor verification as weaker independence | controlled (`test_verification_sends_only_what_the_recipient_may_see`, `test_routing_prefers_another_vendor_and_never_falls_back_to_the_mock`) |
| Injected text in evidence steers the verifier | Evidence, claim and project are fenced with a per-request nonce and fence-like tags are defanged; the verifier's only output is a typed assessment; it can call no tools | partly: lowers the rate, does not bound it. A misled verifier can still produce a wrong verdict, which is why verdicts never settle a claim |
| The `claude` CLI subprocess acts on the machine or leaks the prompt | `--tools ""`, `--strict-mcp-config` with no config, `--setting-sources ""` (so no hooks), `--no-session-persistence`, an empty 0700 sandbox cwd, the system prompt in a 0600 file deleted after the call (not in argv, so not in `ps`), IDE-attach variables removed | controlled for what CI passes (`test_claude_cli_adapter_runs_tool_less_and_parses_the_json_protocol`). **Not controlled:** the CLI's own behavior beyond those flags |
| A late approval restarts a model conversation the person closed | Resolving a task from a closed session settles the task only | controlled (`test_a_late_approval_does_not_resume_a_closed_session`) |

## Accepted risks in this slice

- There is no container, seccomp or network isolation for `run_tests`. Approving a test
  run means trusting that repository's code as much as running it yourself.
- Live model calls go through the person's Claude Code login (`claude-cli`). That bills or
  meters their Claude account, and the content goes to Anthropic under that account's
  terms. The Anthropic API and OpenAI paths are tested against recorded wire formats only;
  no key is configured.
- New projects grant access to the providers chosen at onboarding. That is an explicit,
  visible, revocable default, but it is a default. A project can be created with
  `providers: []`.
- File contents CI reads in a workspace are sent to the chosen provider. Authorizing the
  workspace is the consent for that, and the UI says so.
- Single-user local deployment. Multi-tenant hosting needs real authentication,
  per-tenant keys and row-level isolation tests before it exists.
