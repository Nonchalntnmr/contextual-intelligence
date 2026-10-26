# CI architecture

CI is a local-first layer between one person and the AI models they use. It does four things:

- keeps what it knows about the person, encrypted;
- decides per request what each model may see;
- runs the person's projects with governed tools;
- coordinates more than one model through scoped, recorded exchanges.

**Status, measured, as of 2026-10-26.**

- **What holds:** the permission and isolation guarantees hold under test.
- **What helps, on a capability test (E1):** CI's context clearly beats no context. On Haiku 5.5 it does not beat naive history on task success. It does keep private, injected and stale content out of every prompt.
- **What does not work yet:** the context engine is not relevance-selective at small memory sizes (`experiments/e1/REPORT.md`).

## 1. Components and dependencies

Arrows mean "imports". Taken from the code, not from intent.

```
                       api.py (FastAPI, bearer token)        mcp_server.py (stdio, local token)
                          │                                        │
     ┌──────────┬─────────┼──────────┬───────────┐                 │
     ▼          ▼         ▼          ▼           ▼                 ▼
 exchange.py  agent.py  tasks.py  registry.py  engine.py ◄──── workspace.py ◄─┘
 (claims,     (sessions, (durable   (models,    (PersonalCI:     (projects, notes,
  gateway)     loop,      queue)     vendors,    memory, egress,  provider access,
     │         limits)      │        routing)    policy, audit)   autonomy, briefing)
     │            │         ▼                        │
     │            │     governor.py ─► tools.py      ├─ memory.py ─ store.py (SQLite, AES-GCM per record)
     │            ▼     (decide,       (jail, run,   ├─ sovereignty/ consent · egress · audit · crypto · injection
     │       adapters/   recheck)       timeouts)    ├─ policy/ candidates · features · learned · select
     └────►  anthropic · claude_cli · openai_compat · gemini · mock · mock_agent · render · pricing
```

| module | owns | does not own |
|---|---|---|
| `store.py` | SQLite schema, per-record encryption, crypto-shredding, the decrypted-item cache | any policy decision |
| `memory.py` | bitemporal memory: supersession, correction, conflicts, TTL, lineage, deletion cascade | egress |
| `sovereignty/consent.py`, `egress.py` | who may store what, and which vendor may see which item (lineage-aware) | project content |
| `engine.py` | context selection (scope → egress → candidates → policy) and one-shot `ask` | sessions, tools |
| `workspace.py` | projects, notes with provenance and lineage, per-project provider access, autonomy settings, briefings | the model loop |
| `governor.py`, `tools.py`, `tasks.py` | whether a tool call may run, running it inside the jail, its durable lifecycle | which model asked |
| `agent.py` | sessions, the tool loop, turn limits, disclosures, reflections | permission rules (it asks the governor and the workspace) |
| `exchange.py` | the claim ledger and the model-to-model gateway | routing policy |
| `registry.py` | provider and vendor mapping, model availability, deterministic routing | calling models |

## 2. The request lifecycle, and where the code differs from the intended one

The intended lifecycle runs: request → identity → permissions → context retrieval → context
selection → model routing → agent execution → tool invocation → memory updates → response
→ audit trail.

The table follows one chat message through the code.

| stage | where | what happens |
|---|---|---|
| identity | `api.py: user()` | The bearer token is hashed and looked up. Every route except onboarding requires it (`test_every_route_needs_a_token`). |
| permissions (project) | `agent.py: send()`, `workspace.py: require_provider()` | The project must allow this vendor. A halted user is refused before anything is read. |
| session | `send()` | Reuses the open session for this provider and model unless consent or memory changed since it began (the context epoch). State is reloaded under the session lock. |
| briefing | `Agent.briefing()` | Built from stored project state, filtered for the vendor: model-written notes and claims derived from memory the vendor may not see are withheld. Text is collapsed to one line, with CI's labels first. |
| context retrieval | `engine.py: select_context()` | Current memory from the cache. Out-of-scope project memory is dropped. Egress runs per item and per lineage against one grants snapshot. |
| context selection | `policy/candidates.py`, `policy/learned.py` | Multi-channel candidates (similarity, standing kinds, recency, pins). A density-greedy selection under the token budget acts on the posterior mean. Every candidate gets an explanation record. |
| model routing | the person picks the chat model; `registry.route()` picks verifiers | Deterministic, explained. Never the claim author's own model. Test doubles only on request. |
| agent execution | `agent.py: _run()` | Before every model call: provider access, context epoch, halt, and the turn limits. The model is called. The step is accounted (tokens, cost). |
| tool invocation | `tasks.py: propose()` → `governor.decide()` → `_execute()` → `governor.recheck()` → `tools.Executor` | Idempotent per tool-use id. Validated arguments. The governor decides execute, approve or deny, and decides again at execution. Path jail, timeouts, a scrubbed environment. |
| memory updates | `record_note` and `record_claim` tasks; the API | Model notes and claims carry provider, model and session, plus the lineage of memory the model had seen. Person-written memory goes through `memory.remember` (injection gating, supersession, conflicts). |
| response | `_turn()` | Status (done, awaiting_approval, stopped, blocked), stop reason, this turn's token usage, the session's cost. |
| audit | `sovereignty/audit.py` | A hash chain with content-free payloads: egress decisions, task lifecycle, disclosures, refusals, stops, model errors. |

**Differences from the intended lifecycle**, each one deliberate or recorded:

1. **Routing is not automatic for chat.** The person chooses the model. Automatic routing on quality would need measured quality per task type. That data does not exist yet (CI_ROADMAP).
2. **Memory updates from conversation are explicit, not inferred.** CI never turns chat into memory on its own. Only the person, or an approved tool call, writes it. This is deliberate: inferred memory is the main source of the stale and false context the E1 dataset is built from.
3. **Selection is not yet relevance-selective at small memory sizes.** E1 measured context precision of 3.3% (CI_BENCHMARKS). The architecture has the right inputs, but the designed prior's threshold admits nearly every allowed item.

## 3. Data model and where things live

| data | stored as | key | deleted by |
|---|---|---|---|
| memory item | `items` row; content, tags, embedding encrypted | per item | `memory.delete`: destroys the key, cascades to derived items |
| project, notes | `projects` row, sealed | per project | `delete_project`: destroys the key and deletes scoped memory |
| session transcript, disclosures | `sessions` row, sealed | project key | with the project |
| task arguments, results, write backups | `tasks` row, sealed | project key | with the project |
| claims, exchange messages | `claims`, `exchange_messages`, sealed | project key | with the project |
| consent grants, authorizations | plaintext rows (no content) | — | revocation marks them, never deletes |
| audit log | plaintext hash chain (no content: ids, counts, hashes) | — | never; an erasure is itself recorded |

**Scopes.**

- **User:** every row carries the user id, and every read filters by it (`test_another_user_cannot_read_or_act_on_anything`).
- **Project:** project content is sealed with the project key. Memory can be scoped to one project with a `project:<id>` tag, and is then a candidate only in that project (`in_scope`).
- **Vendor:** consent and provider access are decided per vendor, not per transport. Revoking Anthropic blocks both the API and the CLI.

**Caches.** Each one is invalidated on every relevant write. The first two are also invalidated
whenever SQLite's `data_version` shows another process wrote. None survives a restart.

- decrypted memory items;
- the embeddings index;
- unwrapped record keys (until the key is destroyed);
- model adapters.

## 4. Trust boundaries

Detail and controls are in CI_SECURITY_MODEL.

| boundary | what crosses | rule |
|---|---|---|
| person ↔ API / MCP | requests | token per user; localhost by default |
| CI → vendor | briefing, selected memory, transcript, tool results | project provider access, then egress per item and lineage, then derived-content checks, then a disclosure record |
| model → tools | proposed calls | the governor decides; the person approves risky ones; re-checked at execution |
| workspace content → model | file contents, test output | fenced as data; never instructions |
| model → model | one `AgentMessage` with resolved evidence | only refs the recipient may see; replies only through a typed assessment |
| CI → `claude` CLI | the prompt via a 0600 file and stdin | CLI tools, MCP, settings and hooks off; never rerouted off Anthropic |

## 5. The context engine

```
memory.current (cached) → in_scope(project) → egress (per item, lineage, one grants snapshot)
   → build_candidates (similarity top 30 ∪ standing kinds ∪ recent episodes ∪ pinned)
   → featurize (type × kind × relevance cells, staleness, confidence, length, similarity)
   → BayesLinearPolicy (posterior mean, density-greedy under the budget) → explain records
   → render: confirmed pinned profile in the system section; everything else fenced as data,
     labelled with kind, origin in words, date, confidence, conflict state
```

The pipeline separates the person's statements, confirmed statements, connector imports, CI
inferences and untrusted documents (`Trust`). It separates model hypotheses (claims, with
ledger status) from facts, and model-written notes from the person's notes. Inferred items
decay. Superseded and retracted items never reach a model. Conflicts are flagged in the label
and listed for the person.

**What is measured** (E1, deterministic):

- private, injected and stale content: **0%** of prompts;
- relevant items: **recall 1.0**;
- **precision 3.3%**.

## 6. The agent runtime

- **States:** `done`, `awaiting_approval`, `stopped` and `blocked`. A stop carries its reason:
  - `max_rounds`, `max_tool_calls`, `max_seconds`, `max_cost`, `max_tokens`;
  - `halted`, `context_changed`;
  - `provider_revoked`, which is the `blocked` status.
- **Limits** (`TurnLimits`) are checked before every model call. Every proposed tool call is answered, so the transcript stays valid even when calls are refused for a limit.
- **Durability:**
  - State is saved after every round and before the model call.
  - A provider error is audited and re-raised with the transcript intact.
  - A task running when the process died becomes `interrupted` and is never re-run.
  - An approval expires after an hour.
- **Concurrency:** one lock per session in the process, with state reloaded under it (`test_concurrent_turns_on_one_session_lose_nothing`). The store has one lock around SQLite. Two server processes on one store are not supported.
- **Cancellation and halt:**
  - Cancel kills a running test's whole process group.
  - Halt cancels live tasks, refuses new sends, and stops a running turn before its next model call.
  - A model call already in flight cannot be interrupted, but its result is not acted on after a halt.

## 7. Cross-model layer

**Provider and vendor.** A *provider* is how CI reaches a model:

- `anthropic`: API key, official SDK.
- `claude-cli`: the local Claude Code login.
- `openai`.
- `mock`.

A *vendor* is who receives the data.

**Project provider access.** Checked:

- before the briefing is built;
- before every model step;
- before every delegation;
- before every MCP briefing, listing, context request or write.

Each refusal is audited as `egress.blocked`.

**Disclosures.** Every message records:

- who received it;
- the briefing's hash and text, and what was withheld from it, and why;
- the memory sent and withheld, with the selection explanation;
- the memory in the system section.

**Claims.**

- **Who records them:** a model (`record_claim`) or the person. Each carries evidence refs and lineage.
- **Statuses:**
  - `unverified`;
  - `corroborated`: another model agreed, which is not verification;
  - `contested`: any disagreement. Every assessment is kept.
  - `verified` or `refuted`: only by a linked `run_tests` result that is newer than any workspace change, or by the person.

**Exchange gateway.** `Gateway.verify_claim` sends one `AgentMessage` in the brief's wire
format. Before it does:

- every evidence ref is resolved for the recipient;
- derived content is checked against the recipient's vendor;
- the outbound message is stored before sending, with the hash of the rendered prompt.

The reply comes back only through `submit_assessment`.

**Routing.**

- Never the author's own model. Identity is canonical, so `claude-cli:opus` and `anthropic:claude-opus-5-5` are the same model.
- A different vendor is preferred. A same-vendor verifier is labelled "weaker independence."
- Each model's verifier track record is reported; routing does not use it yet.

**The `claude-cli` adapter.**

- Plain JSON tool protocol. Structured output made Haiku try native tool calls.
- One retry on a non-JSON reply.
- The served model is taken from the requested entry, not from the CLI's auxiliary calls.

## 8. Governor rules

| risk | tools | copilot | collaborative (default) | autonomous |
|---|---|---|---|---|
| read | read_file, list_files, search_code, git_status | execute | execute | execute |
| note | record_note, record_claim | execute | execute | execute |
| run | run_tests | approval | approval | execute within an active grant, else approval |
| write | write_file | approval | approval | same as run |
| external | none exist | — | — | — |

These rules hold in every mode:

- **Paths:** resolved with symlinks, and kept inside an authorized root. `$HOME`, `/` and the ancestors of `$HOME` cannot be roots.
- **Files never touched:** secret-looking files are never read, listed or searched. Nothing is ever written inside `.git`, and there is no git write operation.
- **`run_tests`:** runs only the configured argv, with no shell, a timeout, and a scrubbed environment.
- **`search_code`:** runs in a child process, killed after 10 s.

## 9. Proposed evolution

In order of evidence:

1. **Relevance threshold for selection.** E1 shows the gap. Calibrate on a development set, then confirm on a fresh test set (E2).
2. **A second vendor, live.** That would give verification real independence.
3. **Automatic routing on measured verifier track records,** once there are enough settled claims.
4. **Multi-user hosting:**
   - Postgres with row-level security;
   - per-tenant keys in a KMS;
   - real authentication;
   - one process per tenant or a shared-nothing worker model, since the session lock is per process today.

Detail is in CI_ROADMAP.
