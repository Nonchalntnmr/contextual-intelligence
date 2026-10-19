# Threat model: Personal CI with workspace execution

**Scope.** One person, one machine, a local server on 127.0.0.1, an encrypted SQLite store
in `$PCIM_HOME`, model calls to one external provider, and tools that act on folders the
person authorized.

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

## Accepted risks in this slice

- There is no container, seccomp or network isolation for `run_tests`. Approving a test
  run means trusting that repository's code as much as running it yourself.
- There is no live provider test. The Anthropic tool loop is tested against recorded wire
  format only.
- File contents CI reads in a workspace are sent to the chosen provider. Authorizing the
  workspace is the consent for that, and the UI says so.
- Single-user local deployment. Multi-tenant hosting needs real authentication,
  per-tenant keys and row-level isolation tests before it exists.
