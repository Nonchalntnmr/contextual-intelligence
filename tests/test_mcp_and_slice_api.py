"""The two outside surfaces of the workspace slice: MCP for terminal agents, HTTP for the dashboard."""

import asyncio
import sys

import pytest
from fastapi.testclient import TestClient

from pcim.api import create_app
from pcim.engine import PersonalCI
from pcim.workspace import Workspace

mcp = pytest.importorskip("mcp")
from pcim.mcp_server import build_server  # noqa: E402


def _call(server, name, args=None):
    async def go():
        async with mcp.Client(server) as c:
            r = await c.call_tool(name, args or {})
            return r.content[0].text if r.content else ""
    return asyncio.run(go())


def test_mcp_context_is_consent_filtered_and_agent_memories_are_gated(tmp_path):
    ci = PersonalCI(tmp_path, seed=0)
    ci.onboard("u", providers=("anthropic",))
    ci.remember("u", "Uses 25bps round-trip costs in every backtest", "constraint")
    ci.consent("u", "ingest:notes", "store")
    ci.ingest_text("u", "Private draft: fund terms we have not shared.", source="ingest:notes")  # no egress grant
    srv = build_server(ci, "u", provider="anthropic")

    out = _call(srv, "ci_personal_context", {"task": "set up costs for the backtest"})
    assert "25bps" in out and "fund terms" not in out and "withheld by consent: 1" in out
    iid = out.split("\n")[0].split(": ")[1]
    assert "recorded reward 1.00" == _call(srv, "ci_report_outcome", {"interaction_id": iid, "task_success": True})

    assert "not allowed yet" in _call(srv, "ci_remember", {"content": "the user prefers dark mode", "kind": "preference"})
    ci.consent("u", "agent:mcp", "store")
    msg = _call(srv, "ci_remember", {"content": "the user prefers dark mode", "kind": "preference"})
    assert "held as a fact until confirmed" in msg
    item = ci.store.get_item(msg.split()[1])
    assert item.kind == "fact" and item.trust.value == "imported"

    p = Workspace(ci).create_project("u", "Crowding")
    assert _call(srv, "ci_add_project_note", {"project_id": p.id, "kind": "decision", "text": "drop rho(J)"}).startswith("recorded")
    assert "drop rho(J) [model-written]" in _call(srv, "ci_project_briefing", {"project_id": p.id})


def test_vertical_slice_over_http(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "notes.md").write_text("v1\n")
    data = tmp_path / "data"
    c = TestClient(create_app(PersonalCI(data, seed=0)))
    tok = c.post("/v1/onboard", json={"user_id": "n", "providers": ["mock"]}).json()["api_token"]
    h = {"Authorization": f"Bearer {tok}"}
    pid = c.post("/v1/projects", headers=h, json={"name": "Paper", "objective": "finish section 4"}).json()["id"]
    c.post(f"/v1/projects/{pid}/workspace", headers=h,
           json={"root": str(repo), "run_argv": [sys.executable, "-c", "print('ok')"], "allow_write": True})
    c.post(f"/v1/projects/{pid}/notes", headers=h, json={"kind": "open_question", "text": "Is the proof of lemma 2 tight?"})

    t = c.post(f"/v1/projects/{pid}/chat", headers=h, json={"message": "run tests"}).json()
    assert t["status"] == "awaiting_approval"
    done = c.post(f"/v1/tasks/{t['pending'][0]}/approve", headers=h).json()
    assert done["status"] == "done" and "ok" in done["text"]
    c.post(f"/v1/projects/{pid}/session/close", headers=h)

    # reopen: a brand-new app on the same store
    c2 = TestClient(create_app(PersonalCI(data, seed=1)))
    b = c2.get(f"/v1/projects/{pid}", headers=h).json()["briefing"]
    assert b["open_questions"][0]["text"] == "Is the proof of lemma 2 tight?"
    assert any("ran the tests" in line for line in b["last_session"]["lines"])
    w = c2.post(f"/v1/projects/{pid}/chat", headers=h, json={"message": "write notes.md: v2"}).json()
    assert w["status"] == "awaiting_approval" and (repo / "notes.md").read_text() == "v1\n"
    c2.post(f"/v1/tasks/{w['pending'][0]}/approve", headers=h)
    assert (repo / "notes.md").read_text() == "v2"
    acts = c2.get(f"/v1/tasks?project_id={pid}", headers=h).json()
    write = next(a for a in acts if a["tool"] == "write_file")
    assert write["result"]["sha256_before"] and "backup" not in write["result"]
    c2.post(f"/v1/tasks/{write['id']}/undo", headers=h)
    assert (repo / "notes.md").read_text() == "v1\n"

    # autonomy is server-side
    assert c2.put("/v1/autonomy", headers=h, json={"mode": "reckless"}).status_code == 422
    c2.put("/v1/autonomy", headers=h, json={"mode": "autonomous"})
    c2.post("/v1/autonomy/grants", headers=h, json={"project_id": pid, "tools": ["run_tests"], "max_actions": 1})
    auto = c2.post(f"/v1/projects/{pid}/chat", headers=h, json={"message": "run tests"}).json()
    assert auto["status"] == "done" and auto["tasks"][0]["authorization_id"]
    assert c2.post("/v1/autonomy/halt", headers=h).json()["halted"]
    blocked = c2.post(f"/v1/projects/{pid}/chat", headers=h, json={"message": "read notes.md"}).json()
    assert blocked["tasks"][0]["status"] == "denied" and "halted" in blocked["tasks"][0]["reason"]
    assert c2.post(f"/v1/projects/{pid}/workspace", headers=h, json={"root": "/"}).status_code == 422
