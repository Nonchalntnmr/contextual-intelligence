"""Adversarial isolation: another user, another project, an unauthenticated caller."""

import re
import sys

import pytest
from fastapi.testclient import TestClient

from pcim.adapters.mock_agent import MockAgentAdapter
from pcim.api import create_app
from pcim.engine import PersonalCI

from test_exchange_and_providers import make

PY = sys.executable


# ---------------------------------------------------------------- projects as memory boundaries
def test_project_memory_never_reaches_another_project_or_a_projectless_request(tmp_path):
    ci, ws, ledger, q, agent, gw = make(tmp_path)
    x, y = ws.create_project("u", "X"), ws.create_project("u", "Y")
    scoped = ci.remember("u", "Project X budget constraint: spend at most 4k on compute", "constraint",
                         tags=(f"project:{x.id}",)).item
    shared = ci.remember("u", "I prefer metric units", "preference").item
    mock = MockAgentAdapter(script=[{"text": "x"}, {"text": "y"}])
    agent.set_adapter("mock", None, mock)
    agent.send("u", x.id, "what is the compute budget constraint?")
    agent.send("u", y.id, "what is the compute budget constraint?")
    to_x, to_y = (d for d in agent.disclosures("u", x.id) + agent.disclosures("u", y.id))
    x_candidates = {e["id"] for e in to_x["selection"]}
    y_candidates = {e["id"] for e in to_y["selection"]}
    assert scoped.id in x_candidates and scoped.id not in y_candidates and shared.id in y_candidates
    assert to_y["out_of_scope"] == 1
    assert "4k on compute" not in str(mock.requests[1])
    r = ci.ask("u", "what is the compute budget constraint?")  # one-shot: no project
    assert scoped.id not in {e["id"] for e in r.receipt["selection"]}
    counts = ws.delete_project("u", x.id)
    assert counts["memory_items"] == 1 and ci.store.get_item(scoped.id) is None
    assert ci.store.get_item(shared.id) is not None


def test_the_selection_record_explains_without_carrying_content(tmp_path):
    ci, ws, ledger, q, agent, gw = make(tmp_path)
    p = ws.create_project("u", "P")
    ci.remember("u", "Uses 25bps round-trip costs in every backtest", "constraint")
    agent.set_adapter("mock", None, MockAgentAdapter(script=[{"text": "ok"}]))
    agent.send("u", p.id, "set up backtest costs")
    sel = agent.disclosures("u", p.id)[0]["selection"]
    assert sel and set(sel[0]) == {"id", "selected", "kind", "pinned", "similarity", "age_days", "confidence",
                                   "tokens", "score"}
    assert "25bps" not in str(sel)
    assert "the person said this" in str(agent.adapter("mock", None).requests[-1])


# ---------------------------------------------------------------- another user on the same server
@pytest.fixture
def two_users(tmp_path, monkeypatch):
    monkeypatch.setenv("PCIM_ALLOW_ONBOARD", "1")
    c = TestClient(create_app(PersonalCI(tmp_path / "data", seed=0)))
    ha = {"Authorization": "Bearer " + c.post("/v1/onboard", json={"user_id": "alice", "providers": ["mock"]}).json()["api_token"]}
    hb = {"Authorization": "Bearer " + c.post("/v1/onboard", json={"user_id": "bob", "providers": ["mock"]}).json()["api_token"]}
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("alice's file\n")
    pid = c.post("/v1/projects", headers=ha, json={"name": "Alice's secret project", "objective": "alice-only"}).json()["id"]
    c.post(f"/v1/projects/{pid}/workspace", headers=ha, json={"root": str(repo), "allow_write": True})
    note = c.post(f"/v1/projects/{pid}/notes", headers=ha, json={"kind": "decision", "text": "alice decided"}).json()
    turn = c.post(f"/v1/projects/{pid}/chat", headers=ha, json={"message": "write a.txt: changed"}).json()
    claim = c.post(f"/v1/projects/{pid}/claims", headers=ha, json={"statement": "alice's claim"}).json()
    item = c.post("/v1/memory", headers=ha, json={"content": "alice's private memory"}).json()["item"]
    grant = c.post("/v1/consent", headers=ha, json={"source": "ingest:x", "purpose": "store"}).json()
    ask = c.post("/v1/ask", headers=ha, json={"task": "hello"}).json()
    return c, ha, hb, {"pid": pid, "note": note["id"], "task": turn["pending"][0], "claim": claim["id"],
                       "item": item["id"], "grant": grant["id"], "interaction": ask["interaction_id"]}


def test_another_user_cannot_read_or_act_on_anything(two_users):
    c, ha, hb, a = two_users
    p = a["pid"]
    for method, path, body in [
        ("GET", f"/v1/projects/{p}", None), ("PATCH", f"/v1/projects/{p}", {"name": "pwned"}),
        ("DELETE", f"/v1/projects/{p}", None), ("POST", f"/v1/projects/{p}/notes", {"kind": "decision", "text": "x"}),
        ("POST", f"/v1/projects/{p}/notes/{a['note']}/release", None), ("GET", f"/v1/projects/{p}/session", None),
        ("POST", f"/v1/projects/{p}/chat", {"message": "hi"}), ("GET", f"/v1/projects/{p}/providers", None),
        ("POST", f"/v1/projects/{p}/providers", {"vendor": "mock"}), ("DELETE", f"/v1/projects/{p}/providers/mock", None),
        ("GET", f"/v1/projects/{p}/disclosures", None), ("GET", f"/v1/projects/{p}/briefing?vendor=mock", None),
        ("GET", f"/v1/projects/{p}/claims", None), ("POST", f"/v1/projects/{p}/claims", {"statement": "x"}),
        ("POST", f"/v1/projects/{p}/claims/{a['claim']}/verify", {"provider": "mock"}),
        ("POST", f"/v1/projects/{p}/claims/{a['claim']}/resolve", {"decision": "verified"}),
        ("POST", f"/v1/projects/{p}/claims/{a['claim']}/release", None), ("GET", f"/v1/projects/{p}/exchange", None),
        ("POST", f"/v1/projects/{p}/workspace", {"root": "/tmp"}), ("GET", f"/v1/projects/{p}/route", None),
        ("GET", f"/v1/tasks/{a['task']}", None), ("POST", f"/v1/tasks/{a['task']}/approve", None),
        ("POST", f"/v1/tasks/{a['task']}/deny", {"reason": "x"}), ("POST", f"/v1/tasks/{a['task']}/cancel", None),
        ("POST", f"/v1/tasks/{a['task']}/undo", None),
        ("POST", f"/v1/memory/{a['item']}/correct", {"content": "x"}), ("POST", f"/v1/memory/{a['item']}/retract", None),
        ("POST", f"/v1/memory/{a['item']}/pin", None), ("DELETE", f"/v1/memory/{a['item']}", None),
        ("POST", "/v1/memory", {"content": "x", "project_id": p}),
        ("DELETE", f"/v1/consent/{a['grant']}", None), ("GET", f"/v1/interactions/{a['interaction']}", None),
        ("POST", f"/v1/feedback/{a['interaction']}", {"rating": 5}),
    ]:
        r = c.request(method, path, headers=hb, json=body)
        assert r.status_code == 404, (method, path, r.status_code, r.text[:200])
    assert c.delete(f"/v1/autonomy/grants/{a['grant']}", headers=hb).json() == {"revoked": False}
    everything = " ".join(c.get(path, headers=hb).text for path in (
        "/v1/projects", "/v1/memory?history=true", "/v1/consent", "/v1/tasks", "/v1/interactions", "/v1/export",
        "/v1/audit", "/v1/autonomy", "/v1/conflicts", "/v1/metrics"))
    for secret in ("alice", a["pid"], a["task"], a["claim"], a["item"], a["interaction"]):
        assert secret not in everything
    # and alice's things are intact
    assert c.get(f"/v1/projects/{p}", headers=ha).json()["project"]["name"] == "Alice's secret project"
    assert c.get(f"/v1/tasks/{a['task']}", headers=ha).json()["status"] == "awaiting_approval"


def test_every_route_needs_a_token(two_users):
    c, *_ = two_users
    public = {"/v1/onboard", "/", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    checked = 0
    for route in c.app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/v1") or path in public:
            continue
        concrete = re.sub(r"\{[^}]+\}", "x", path)
        for method in route.methods - {"HEAD", "OPTIONS"}:
            assert c.request(method, concrete).status_code == 401, (method, path)
            assert c.request(method, concrete, headers={"Authorization": "Bearer forged"}).status_code == 401
            checked += 1
    assert checked > 50


def test_the_token_is_never_stored_or_logged_in_clear(two_users, tmp_path):
    c, ha, hb, a = two_users
    token = ha["Authorization"].split()[1]
    raw = b"".join(p.read_bytes() for p in (tmp_path / "data").glob("pcim.db*"))
    assert token.encode() not in raw
    assert token not in c.get("/v1/audit", headers=ha).text and token not in c.get("/v1/export", headers=ha).text
