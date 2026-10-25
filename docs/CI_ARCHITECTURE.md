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
3. **Orchestration.** Tool use works through four adapters:
   - `claude-cli`: live, through the local Claude Code login;
   - Anthropic SDK and OpenAI Chat Completions: recorded wire formats only, since no key is configured;
   - the mock.

   The person picks the chat model. Verification is routed deterministically (see
   "Cross-model layer"). Nothing is routed on claimed quality.
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

## Cross-model layer (milestone 1, 2026-10-24)

```
                 person ── dashboard / CLI / HTTP ──┐
                                                    ▼
   registry.py ── route(role, allowed vendors, author) ──► which model, and why
        │
   Workspace.require_provider(project, provider)  ◄── checked before every send, every model
        │                                              step, every delegation, every MCP briefing
        ▼
   agent.py  session with Provider A ─┐                 ┌─ session with Provider B
   (briefing, not A's transcript)     │  project state  │  (briefing, not A's transcript)
                                      ▼                 ▼
                         notes · claims · tasks · disclosures   (sealed with the project key)
                                      │
   exchange.py  Gateway.verify_claim ─┴─► AgentMessage ci → verifier (scoped refs only)
                                         ◄─ AgentMessage verifier → ci (typed assessment)
                ClaimLedger: unverified → corroborated | contested → verified | refuted
                                                        (test or person only)
```

**Provider vs vendor.** A *provider* is how CI reaches a model:

- `anthropic`: API key, official SDK.
- `claude-cli`: the local Claude Code login.
- `openai`.
- `mock`.

A *vendor* is who receives the data. Consent, egress and project access are decided per
vendor, so revoking Anthropic for a project blocks the API and the CLI alike.

**Project provider access.**

- Each project holds revocable `provider` authorizations, one per vendor.
- New projects start with the providers chosen at onboarding, recorded as explicit grants.
- A project created before this existed is backfilled once, and the backfill is audited.
- Access is checked:
  - in `Agent.send` before the briefing is built;
  - in `Agent._run` before every model step, so a revocation also stops a session paused on an approval;
  - in `Gateway.verify_claim`;
  - in the MCP `ci_project_briefing`.
- Each refusal is audited as `egress.blocked`.

**Disclosures.** Every message records:

- the provider, vendor and model it went to;
- the hash and text of the briefing, if the message included it;
- the note and claim ids in the briefing;
- the memory items sent, and the ones withheld, with reasons.

`GET /v1/projects/{id}/disclosures` lists them. This is how step 9 of the brief ("Provider
B receives only the authorized context") is shown, not just asserted.

**Claims.**

- **Who records them:** a model, through the `record_claim` tool (note risk, runs without approval, provenance from its session), or the person.
- **Evidence:** ids of this project's tasks, notes and claims, or memory items. Every tool result carries its `task_id`, so a model can cite it.
- **Statuses:**
  - `unverified`: no assessment yet.
  - `corroborated`: another model agreed. This is **not** verification.
  - `contested`: any assessor disagreed. Every assessment is kept, with its rationale and counter-claim.
  - `verified` or `refuted`: only a `run_tests` result the person links, or the person's decision. Earlier resolutions stay in `history`.

**Context exchange.** `Gateway.verify_claim` runs these steps in order:

1. Check project access for the verifier's vendor.
2. Refuse when the verifier is the claim's author.
3. Resolve each evidence reference for that recipient:
   - tasks, notes and claims only from this project;
   - memory only through the egress guard for its vendor;
   - anything else is withheld, with a reason.
4. Render a nonce-fenced prompt.
5. Store the outbound `AgentMessage` with the rendered prompt's SHA-256 before sending.
6. Accept the reply only through `submit_assessment` (verdict, rationale, counter-claim, the check that would settle it).
7. Store the inbound message and the assessment.

The `AgentMessage` wire format is the brief's interface exactly (camelCase, strict: unknown
fields and scopes are rejected).

**Routing** is deterministic and explained. Verification never goes to the author's own
model, and prefers a model from a different vendor. A different model from the same vendor
is allowed, labelled "weaker independence." A test double is used only when asked for. Every
decision returns the candidates considered and why each was or was not eligible. The
registry reports each model's track record as a verifier (assessments later borne out or
contradicted by a test or the person). It does not route on it yet.

**The `claude-cli` adapter.**

- Each step is one stateless `claude -p` call, with:
  - the CLI's tools off, no MCP servers, no settings or hooks, no session persistence;
  - an empty sandbox as its working directory;
  - the system prompt passed in a 0600 file, never on the command line.
- The model answers in a plain JSON protocol, `{"text", "tool_calls"}`.
- CI validates and governs every requested call. The CLI executes nothing.
- An earlier version used `--json-schema` structured output. In live runs Haiku 5.5 then
  tried to call CI's tools natively ("No such tool available") and sometimes gave up. With
  no tool in the request at all, it follows the protocol on the first turn (see
  `adapters/claude_cli.py`).

## What stays out of this slice

- **Streaming to the UI.** The SDK call itself is non-streaming within timeouts.
- **Scheduled background work.** The queue exists; a scheduler does not.
- **Sandboxing beyond path jail plus env scrub plus timeout.** No container and no seccomp.
  `run_tests` executes repository code, which is why it always needs approval or an
  explicit autonomous grant.
- **Multi-user hosting.**
