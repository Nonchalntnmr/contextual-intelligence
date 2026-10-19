# CI architecture: the workspace slice

This adds projects, sessions, governed tool execution and autonomy modes to the existing
PCIM core. Every new path goes through the same store (encrypted per record), consent and
egress checks, and audit log.

```
 dashboard / CLI / MCP client
            │  HTTP (bearer token, 127.0.0.1)          stdio (MCP)
            ▼                                            ▼
        api.py ───────────────► Workspace ◄────────── mcp_server.py
                                   │
          ┌────────────────┬───────┴────────┬─────────────────────┐
          ▼                ▼                ▼                     ▼
     projects &       agent loop       governor             task queue +
     sessions       (agent.py) ──────► (governor.py) ─────► executor (tools.py)
   (workspace.py)       │   tool_use     decide:             typed tools, timeouts,
          │             │                execute /           cancellation, path jail,
          │             ▼                approve / deny      secret-file denial
          │       model adapter                              │
          │   (Anthropic tools; mock                         ▼
          │    test double)                            results → tool_result
          ▼
   PersonalCI engine: memory · egress · policy · audit · encryption (unchanged)
```

## The six capabilities

1. **Persistent context.** A project record is sealed with the project's own key. It holds:
   - the objective and success criteria;
   - open questions, decisions and next steps, each with provenance (user or model) and a timestamp;
   - the authorized workspace roots;
   - the last session's reflection.

   Resuming starts a new model session from a **briefing** compiled from that state plus
   policy-selected memory. The raw transcript is not replayed. Continuity therefore comes
   from inspectable state rather than from a long, unreviewable history. Within a session
   the transcript is append-only and durable, so a server restart mid-session can continue.
2. **Joint reasoning.** The system directive requires the model to state assumptions, the
   strongest objection and the evidence that would change its mind, before agreeing with a
   plan. Notes written by the model are labelled as model-generated, never as fact.
3. **Orchestration.** One provider is real: Anthropic tool use through the official SDK,
   tested against recorded wire format only, since no key is configured. Others plug in
   through the same adapter boundary. Model selection is manual: no routing claim without
   benchmark evidence.
4. **Adaptation.** Collaboration preferences are explicit, user-visible memory items
   ("explanations: more technical"). They are written when the user gives that feedback and
   rendered into every later session (Demo E). This is an adaptation feature, not evidence
   of cognitive growth.
5. **Real execution.** Typed tools: `read_file`, `list_files`, `search_code`, `git_status`,
   `run_tests`, `write_file`, `record_note`.
   - Every call passes the governor.
   - Approved tasks are re-checked when they actually run (time-of-check vs time-of-use).
   - Writes keep a sealed backup, so they can be undone.
6. **Continuous availability.** A durable SQLite task queue with explicit states:
   `awaiting_approval → approved → running → succeeded | failed | cancelled | denied | interrupted | expired`.
   - A task that was running when the process died becomes `interrupted` and is never silently re-run.
   - Approvals expire after an hour.

## Governor rules

The governor is the backend authority. The UI only displays its decisions.

| risk | tools | copilot | collaborative (default) | autonomous |
|---|---|---|---|---|
| read | read_file, list_files, search_code, git_status | execute | execute | execute |
| note | record_note | execute | execute | execute |
| run | run_tests | approval | approval | execute only if an active authorization covers it, else approval |
| write | write_file | approval | approval | same as run |
| external | (none implemented) | deny | deny | deny |

These rules hold in every mode:

- Paths are resolved, symlinks included, and must stay inside an authorized root.
- `$HOME`, `/`, and any ancestor of `$HOME` cannot be roots.
- Secret-looking files are never read or sent: `.env*`, `*.key`, `*.pem`, `id_*`, `*credentials*`, `*.db`.
- The `.git` directory is read-only.
- No tool can call git with a write subcommand.
- `run_tests` runs only the argv the user configured for that workspace. It runs with no
  shell, a timeout, and an environment scrubbed of keys and tokens.

An autonomous authorization specifies:

- allowed tools and roots;
- maximum actions, maximum seconds and maximum spend;
- expiry.

It is checked on every call, so revoking it takes effect immediately. A task can never
raise its own authorization. **Halt** cancels every live task, kills running processes,
and blocks all execution until the user resumes.

## Reflection

When a session ends, a factual reflection is computed from task records:

- what ran, what changed (files, with hashes), what failed, what is still pending;
- the open questions.

It becomes the start of the next briefing. Any interpretation the model adds is stored
separately and labelled as such.

## What stays out of this slice

- **Streaming to the UI.** The SDK call itself is non-streaming within timeouts.
- **Scheduled background work.** The queue exists; a scheduler does not.
- **Sandboxing beyond path jail plus env scrub plus timeout.** No container and no seccomp.
  `run_tests` executes repository code, which is why it always needs approval or an
  explicit autonomous grant.
- **Multi-user hosting.**
