"""Milestone 1: the sixteen-step demonstration from the build brief, end to end over HTTP.

Provider A is Anthropic and Provider B is OpenAI, both on their recorded wire formats
(no keys on this machine; `scripts/milestone1_live.py` runs the same flow against real
models). The application really restarts between step 5 and step 7: a new app on a new
PersonalCI over the same encrypted store. File edits and test runs are real.
"""

import json
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from pcim.adapters.openai_compat import OpenAICompatAdapter
from pcim.api import create_app
from pcim.engine import PersonalCI

anthropic = pytest.importorskip("anthropic")
import httpx2  # noqa: E402  (anthropic 1.x's HTTP stack)
from pcim.adapters.anthropic import AnthropicAdapter  # noqa: E402

PY = sys.executable
CANARY = "ZEBRA-7741"
FUND_TERMS = "2 and 20 with a 6% hurdle"

SHARPE_365 = ("import math, statistics\n\n\ndef sharpe(daily):\n"
              "    return statistics.mean(daily) / statistics.stdev(daily) * math.sqrt(365)\n")
SHARPE_252 = SHARPE_365.replace("365", "252")
CHECK = ("import math, statistics, sys\nsys.path.insert(0, 'src')\nfrom sharpe import sharpe\n"
         "r = [0.01, -0.005, 0.002, 0.004]\n"
         "assert abs(sharpe(r) - statistics.mean(r) / statistics.stdev(r) * math.sqrt(252)) < 1e-12\n"
         "print('1 passed')\n")


def task_id_of(tool_result_content: str) -> str:
    return json.loads(tool_result_content)["task_id"]


def anthropic_wire(replies):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        content, stop = replies[len(seen) - 1](body)
        return httpx2.Response(200, json={"id": f"msg_{len(seen)}", "type": "message", "role": "assistant",
                                          "model": "claude-opus-5-5", "content": content, "stop_reason": stop,
                                          "stop_sequence": None, "usage": {"input_tokens": 1200, "output_tokens": 60}})
    client = anthropic.Anthropic(api_key="test", http_client=anthropic.DefaultHttpxClient(
        transport=httpx2.MockTransport(handler)))
    return AnthropicAdapter("claude-opus-5-5", client=client), seen


def openai_wire(replies):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        msg = replies[len(seen) - 1](body)
        return httpx.Response(200, json={"id": f"chatcmpl-{len(seen)}", "model": "gpt-test", "choices": [
            {"index": 0, "message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40}})
    return OpenAICompatAdapter("gpt-test", provider="openai", api_key="test",
                               transport=httpx.MockTransport(handler)), seen


def call(name, args, n):
    return {"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_milestone1_sixteen_steps(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "sharpe.py").write_text(SHARPE_365)
    (repo / "check.py").write_text(CHECK)
    data = tmp_path / "data"

    # ------------------------------------------------------------ process 1
    ci1 = PersonalCI(data, seed=0)
    c1 = TestClient(create_app(ci1))

    # 1. sign in
    tok = c1.post("/v1/onboard", json={"user_id": "n", "providers": ["anthropic", "openai"]}).json()["api_token"]
    h = {"Authorization": f"Bearer {tok}"}
    assert c1.get("/v1/me", headers=h).status_code == 200
    assert c1.get("/v1/me").status_code == 401

    # 2-3. a project with a goal and constraints
    p = c1.post("/v1/projects", headers=h, json={
        "name": "Sharpe audit", "objective": "Make the backtest's Sharpe ratio correct",
        "constraints": ["daily returns", "no new dependencies"], "success_criteria": "check.py passes"}).json()
    pid = p["id"]
    assert c1.get(f"/v1/projects/{pid}/providers", headers=h).json()["vendors"] == ["anthropic", "openai"]
    c1.post(f"/v1/projects/{pid}/workspace", headers=h,
            json={"root": str(repo), "run_argv": [PY, "check.py"], "allow_write": True})
    # private context that only Anthropic may see
    c1.post("/v1/consent", headers=h, json={"source": "ingest:fund", "purpose": "store"})
    c1.post("/v1/consent", headers=h, json={"source": "ingest:fund", "purpose": "egress", "provider": "anthropic"})
    fund = c1.post("/v1/ingest", headers=h, json={"text": f"Fund terms: {FUND_TERMS}.", "source": "ingest:fund"}).json()
    fund_id = fund["items"][0]["id"]

    # 4-5. a real request to Provider A; it reads the code, records a decision and a claim
    def a_step2(body):
        read_task = task_id_of(body["messages"][-1]["content"][0]["content"])
        return ([{"type": "text", "text": "It annualises with the trading-day convention."},
                 {"type": "tool_use", "id": "toolu_a2", "name": "record_note",
                  "input": {"kind": "decision", "text": "Report annualised Sharpe from daily returns"}},
                 {"type": "tool_use", "id": "toolu_a3", "name": "record_claim",
                  "input": {"statement": "sharpe() in src/sharpe.py annualises with sqrt(252)",
                            "evidence": read_task, "uncertainty": "none"}}], "tool_use")
    adapter_a, seen_a = anthropic_wire([
        lambda b: ([{"type": "text", "text": "Reading the code."},
                    {"type": "tool_use", "id": "toolu_a1", "name": "read_file", "input": {"path": "src/sharpe.py"}}],
                   "tool_use"),
        a_step2,
        lambda b: ([{"type": "text", "text": "Recorded."}], "end_turn"),
    ])
    c1.app.state.agent.set_adapter("anthropic", None, adapter_a)
    t = c1.post(f"/v1/projects/{pid}/chat", headers=h, json={
        "message": f"Is our Sharpe calculation right? (my canary phrase is {CANARY})", "provider": "anthropic"}).json()
    assert t["status"] == "done" and len(seen_a) == 3
    claims = c1.get(f"/v1/projects/{pid}/claims", headers=h).json()
    assert len(claims) == 1 and claims[0]["status"] == "unverified"
    claim_a = claims[0]
    assert claim_a["author"]["provider"] == "anthropic" and claim_a["evidence_refs"]
    decision = c1.get(f"/v1/projects/{pid}", headers=h).json()["briefing"]["decisions"][0]
    assert decision["source"] == "model" and decision["by"]["provider"] == "anthropic"

    # 6. close the application
    ci1.store.close()
    del c1, ci1

    # ------------------------------------------------------------ process 2
    c2 = TestClient(create_app(PersonalCI(data, seed=1)))

    # 7. the project state comes back from the store
    got = c2.get(f"/v1/projects/{pid}", headers=h).json()
    assert got["project"]["constraints"] == ["daily returns", "no new dependencies"]
    assert got["briefing"]["decisions"][0]["text"] == "Report annualised Sharpe from daily returns"
    assert got["briefing"]["claims"][0]["statement"].startswith("sharpe() in src/sharpe.py")

    # A had been shown the Anthropic-only fund terms, so what A wrote may carry them: OpenAI is
    # not given A's notes or claims until the person has read them and released them.
    preview = c2.get(f"/v1/projects/{pid}/briefing", headers=h, params={"vendor": "openai"}).json()
    held = {w["id"] for w in preview["withheld"]}
    assert decision["id"] in held and claim_a["id"] in held and "Report annualised" not in preview["text"]
    c2.post(f"/v1/projects/{pid}/notes/{decision['id']}/release", headers=h)
    c2.post(f"/v1/projects/{pid}/claims/{claim_a['id']}/release", headers=h)
    preview = c2.get(f"/v1/projects/{pid}/briefing", headers=h, params={"vendor": "openai"}).json()
    assert "Report annualised" in preview["text"] and FUND_TERMS not in preview["text"]

    # 8-9. resume with Provider B, which gets the briefing and nothing it is not allowed
    b_replies = [lambda b: {"role": "assistant", "content": "Resuming the Sharpe audit."}]
    adapter_b, seen_b = openai_wire(b_replies)
    c2.app.state.agent.set_adapter("openai", None, adapter_b)
    r = c2.post(f"/v1/projects/{pid}/chat", headers=h, json={"message": "Pick up where we left off.",
                                                             "provider": "openai"}).json()
    assert r["status"] == "done"
    first = json.dumps(seen_b[0])
    assert "<project_briefing" in first and "Report annualised Sharpe from daily returns" in first
    assert "annualises with sqrt(252)" in first and "no new dependencies" in first
    assert CANARY not in first and FUND_TERMS not in first  # A's transcript and A-only memory stay behind
    disc = c2.get(f"/v1/projects/{pid}/disclosures", headers=h).json()
    to_a = [d for d in disc if d["vendor"] == "anthropic"]
    to_b = [d for d in disc if d["vendor"] == "openai"]
    assert to_b[0]["briefing_sha256"] and claim_a["id"] in to_b[0]["briefing_claims"]
    assert any(i == fund_id and "no-egress-grant" in why for i, why in to_b[0]["memory_withheld"])
    assert fund_id in to_a[0]["memory_items"]  # A was shown it, which is why A's notes needed a release

    # 10. a narrow verification task delegated to B
    assert c2.post(f"/v1/projects/{pid}/claims/{claim_a['id']}/verify", headers=h,
                   json={"provider": "anthropic", "model": "claude-opus-5-5"}).status_code == 422  # its own claim
    b_replies.append(lambda b: {"role": "assistant", "content": None, "tool_calls": [call("submit_assessment", {
        "verdict": "refuted", "rationale": "The file multiplies by math.sqrt(365).",
        "counter_claim": "sharpe() annualises with sqrt(365)", "check": "run check.py"}, 1)]})
    v = c2.post(f"/v1/projects/{pid}/claims/{claim_a['id']}/verify", headers=h, json={"provider": "openai"}).json()
    sent = json.dumps(seen_b[-1])
    assert "math.sqrt(365)" in sent and "annualises with sqrt(252)" in sent  # the evidence and the claim
    assert CANARY not in sent and FUND_TERMS not in sent and "Pick up where we left off" not in sent

    # 11. evidence and disagreement are displayed, not averaged away
    assert v["claim"]["status"] == "contested" and v["assessment"]["independence"] == "different vendor"
    assert v["assessment"]["counter_claim"] == "sharpe() annualises with sqrt(365)"
    msgs = c2.get(f"/v1/projects/{pid}/exchange", headers=h).json()
    out_msg = msgs[0]["message"]
    assert set(out_msg) == {"taskId", "projectId", "sender", "recipient", "goal", "constraints", "contextRefs", "claim",
                            "evidenceRefs", "uncertainty", "dependencies", "requestedAction", "authorizationScope",
                            "timestamp", "schemaVersion"}
    assert out_msg["recipient"] == "openai:gpt-test" and out_msg["evidenceRefs"] == claim_a["evidence_refs"]
    assert msgs[1]["message"]["sender"] == "openai:gpt-test" and msgs[1]["message"]["recipient"] == "ci"

    # 12-14. a reversible action, approved, executed, and its result verified by a test
    b_replies.append(lambda b: {"role": "assistant", "content": "Here is the fix.", "tool_calls": [
        call("write_file", {"path": "src/sharpe.py", "content": SHARPE_252, "reason": "use 252 trading days"}, 2)]})

    def b_after_write(body):
        write_task = task_id_of(body["messages"][-1]["content"])
        return {"role": "assistant", "content": "Written; checking.", "tool_calls": [
            call("record_claim", {"statement": "sharpe() now annualises with sqrt(252)", "evidence": write_task,
                                  "uncertainty": ""}, 3),
            call("run_tests", {"reason": "confirm the fix"}, 4)]}
    b_replies += [b_after_write, lambda b: {"role": "assistant", "content": "check.py passes."}]
    w = c2.post(f"/v1/projects/{pid}/chat", headers=h, json={"message": "Fix the annualisation to 252 trading days.",
                                                             "provider": "openai"}).json()
    assert w["status"] == "awaiting_approval" and (repo / "src" / "sharpe.py").read_text() == SHARPE_365
    w2 = c2.post(f"/v1/tasks/{w['pending'][0]}/approve", headers=h).json()
    assert (repo / "src" / "sharpe.py").read_text() == SHARPE_252
    write = c2.get(f"/v1/tasks/{w['pending'][0]}", headers=h).json()
    assert write["status"] == "succeeded" and write["result"]["sha256_before"] != write["result"]["sha256_after"]
    assert w2["status"] == "awaiting_approval"  # run_tests next
    test_task = w2["pending"][0]
    done = c2.post(f"/v1/tasks/{test_task}/approve", headers=h).json()
    assert done["status"] == "done" and done["text"] == "check.py passes."
    assert c2.get(f"/v1/tasks/{test_task}", headers=h).json()["result"]["exit_code"] == 0
    claim_b = next(c for c in c2.get(f"/v1/projects/{pid}/claims", headers=h).json() if c["id"] != claim_a["id"])
    settled = c2.post(f"/v1/projects/{pid}/claims/{claim_b['id']}/resolve", headers=h,
                      json={"test_task_id": test_task}).json()
    assert settled["status"] == "verified" and settled["resolution"]["by"] == "test"

    # 15. revoke Provider B
    assert c2.delete(f"/v1/projects/{pid}/providers/openai", headers=h).json()["vendors"] == ["anthropic"]

    # 16. nothing more reaches B: chat, verification, and the audit trail shows the refusals
    n = len(seen_b)
    assert c2.post(f"/v1/projects/{pid}/chat", headers=h,
                   json={"message": "one more thing", "provider": "openai"}).status_code == 403
    assert c2.post(f"/v1/projects/{pid}/claims/{claim_a['id']}/verify", headers=h,
                   json={"provider": "openai"}).status_code == 403
    assert len(seen_b) == n
    audit = c2.get("/v1/audit", headers=h).json()
    assert audit["chain_ok"]
    blocked = [e for e in audit["entries"] if e["event"] == "egress.blocked"]
    assert len(blocked) == 2 and all(e["payload"]["vendor"] == "openai" for e in blocked)

    # and the person can take all of it with them
    exp = c2.get("/v1/export", headers=h).json()["projects"][0]
    assert exp["providers"] == ["anthropic"] and len(exp["claims"]) == 2 and len(exp["exchange"]) == 2
    assert {d["vendor"] for d in exp["disclosures"]} == {"anthropic", "openai"}
    assert all("backup" not in (t["result"] or {}) for t in exp["tasks"])
