# Implementation plan: the first vertical slice

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
