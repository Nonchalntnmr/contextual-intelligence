"""Provider access, the context exchange gateway, the claim ledger, routing, and the two
bridge adapters (claude CLI, OpenAI tool use)."""

import asyncio
import json
import stat
import subprocess
import sys

import httpx
import pydantic
import pytest

from pcim import registry
from pcim.adapters.claude_cli import ClaudeCLIAdapter, ClaudeCLIError
from pcim.adapters.mock_agent import MockAgentAdapter
from pcim.adapters.openai_compat import OpenAICompatAdapter
from pcim.agent import Agent
from pcim.engine import PersonalCI
from pcim.exchange import AgentMessage, ClaimLedger, Gateway
from pcim.governor import Governor
from pcim.tasks import TaskQueue
from pcim.workspace import ProviderAccessDenied, Workspace

PY = sys.executable


def make(tmp_path, providers=("mock", "anthropic")):
    ci = PersonalCI(tmp_path / "data", seed=0)
    if not ci.store.list_user_ids():
        ci.onboard("u", providers=providers)
    ws = Workspace(ci)
    ledger = ClaimLedger(ws)
    q = TaskQueue(ws, Governor(ws), claims=ledger)
    agent = Agent(ci, ws, q, ledger)
    return ci, ws, ledger, q, agent, Gateway(ci, ws, q, ledger, agent.adapter)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "a.txt").write_text("alpha\n")
    return r


# ---------------------------------------------------------------- the message format
def test_agent_message_wire_format_is_the_documented_interface():
    m = AgentMessage(task_id="x1", project_id="p1", sender="ci", recipient="openai:gpt", goal="verify",
                     authorization_scope=["claim:read"], timestamp="2026-10-24T00:00:00+00:00")
    w = m.wire()
    assert w["taskId"] == "x1" and w["schemaVersion"] == "ci.agent-message/1" and w["authorizationScope"] == ["claim:read"]
    assert AgentMessage.model_validate(w) == m  # camelCase round-trips
    for bad in ({**w, "secretHistory": "..."}, {**w, "authorizationScope": ["tools:execute"]},
                {**w, "schemaVersion": "2"}, {k: v for k, v in w.items() if k != "goal"}):
        with pytest.raises(pydantic.ValidationError):
            AgentMessage.model_validate(bad)


# ---------------------------------------------------------------- the ledger
def test_model_agreement_is_never_verification(tmp_path):
    _, ws, ledger, *_ = make(tmp_path)
    p = ws.create_project("u", "P")
    c = ledger.add("u", p.id, "the backtest has no look-ahead", author={"kind": "model", "provider": "anthropic",
                                                                         "model": "m1"})
    for who in ("m2", "m3"):
        c = ledger.add_assessment("u", p.id, c["id"], {"by": {"provider": "openai", "model": who}, "verdict": "supported"})
    assert c["status"] == "corroborated"  # two models agree: still not verified
    c = ledger.add_assessment("u", p.id, c["id"], {"by": {"provider": "google", "model": "g"}, "verdict": "refuted"})
    assert c["status"] == "contested" and len(c["assessments"]) == 3  # disagreement is kept, not averaged
    with pytest.raises(ValueError):
        ledger.resolve("u", p.id, c["id"], "verified", by="model")
    c = ledger.resolve("u", p.id, c["id"], "verified", by="person", note="checked the fills by hand")
    assert c["status"] == "verified"
    c = ledger.resolve("u", p.id, c["id"], "refuted", by="person")
    assert c["status"] == "refuted" and c["history"][0]["decision"] == "verified"
    rec = ledger.track_record("u")
    assert rec["openai:m2"]["contradicted"] == 1 and rec["google:g"]["borne_out"] == 1


# ---------------------------------------------------------------- provider access
def test_revocation_stops_a_session_paused_on_an_approval(tmp_path, repo):
    ci, ws, ledger, q, agent, _ = make(tmp_path)
    p = ws.create_project("u", "P")
    ws.authorize_workspace("u", p.id, str(repo), allow_write=True)
    mock = MockAgentAdapter(script=[{"tool_uses": [{"id": "t1", "name": "write_file",
                                                    "input": {"path": "a.txt", "content": "beta\n", "reason": "r"}}]},
                                    {"text": "should never be produced"}])
    agent.set_adapter("mock", None, mock)
    turn = agent.send("u", p.id, "change a.txt")
    assert turn.status == "awaiting_approval"
    ws.revoke_provider("u", p.id, "mock")
    after = agent.resolve("u", turn.pending[0], approve=True)
    assert after.status == "blocked" and len(mock.requests) == 1  # the tool result never went back
    assert (repo / "a.txt").read_text() == "beta\n"  # the person's approved local action still ran
    with pytest.raises(ProviderAccessDenied):
        agent.send("u", p.id, "hello?")
    assert sum(e["event"] == "egress.blocked" for e in ci.audit.entries("u")) == 2
    ws.grant_provider("u", p.id, "mock")
    mock.script = [{"text": "back"}]
    assert agent.send("u", p.id, "and now?").text == "back"  # turns stay well-formed after the stop


def test_a_late_approval_does_not_resume_a_closed_session(tmp_path, repo):
    _, ws, _, q, agent, _ = make(tmp_path)
    p = ws.create_project("u", "P")
    ws.authorize_workspace("u", p.id, str(repo), allow_write=True)
    mock = MockAgentAdapter(script=[{"tool_uses": [{"id": "t1", "name": "write_file",
                                                    "input": {"path": "a.txt", "content": "beta\n", "reason": "r"}}]}])
    agent.set_adapter("mock", None, mock)
    turn = agent.send("u", p.id, "change a.txt")
    agent.close_session("u", p.id)
    assert "proposed a change to a.txt — awaiting approval" in agent.briefing("u", p.id)["last_session"]["lines"]
    out = agent.resolve("u", turn.pending[0], approve=True)
    assert out["task"]["status"] == "succeeded" and "closed" in out["session"]
    assert len(mock.requests) == 1 and (repo / "a.txt").read_text() == "beta\n"
    assert "changed a.txt" in agent.briefing("u", p.id)["last_session"]["lines"]  # the briefing follows the records


def test_new_projects_take_onboarding_providers_and_legacy_projects_are_backfilled_once(tmp_path):
    ci, ws, *_ = make(tmp_path)
    assert ws.provider_vendors("u", ws.create_project("u", "A").id) == ["anthropic", "mock"]
    assert ws.provider_vendors("u", ws.create_project("u", "B", providers=[]).id) == []  # explicit: none
    assert ws.provider_vendors("u", ws.create_project("u", "C", providers=["claude-cli"]).id) == ["anthropic"]
    legacy = ws.create_project("u", "Old", providers=[])
    row = ws._row("u", legacy.id)
    payload = ws._payload(row)
    payload.pop("provider_access")  # what a project saved before provider access looks like
    ws._save(row, payload)
    assert ws.provider_vendors("u", legacy.id) == ["anthropic", "mock"]
    ws.revoke_provider("u", legacy.id, "mock")
    ws.revoke_provider("u", legacy.id, "anthropic")
    assert ws.provider_vendors("u", legacy.id) == []  # revoking everything does not re-trigger the backfill


# ---------------------------------------------------------------- gateway scoping
def test_verification_sends_only_what_the_recipient_may_see(tmp_path, repo):
    ci, ws, ledger, q, agent, gw = make(tmp_path)
    p = ws.create_project("u", "P", objective="ship it")
    other = ws.create_project("u", "Other")
    ws.authorize_workspace("u", p.id, str(repo))
    ws.authorize_workspace("u", other.id, str(repo))
    mine = q.propose("u", p.id, "read_file", {"path": "a.txt"})
    theirs = q.propose("u", other.id, "read_file", {"path": "a.txt"})
    ci.consent("u", "ingest:diary", "store")  # stored, but no egress grant for anyone
    diary = ci.ingest_text("u", "Diary: I am interviewing elsewhere.", source="ingest:diary")[0].item
    note = ws.add_note("u", p.id, "decision", "alpha is the baseline")
    c = ledger.add("u", p.id, "a.txt says alpha",
                   evidence_refs=f"{mine.id}, {theirs.id}, {diary.id}, {note['id']}, task_00000000000000000000",
                   author={"kind": "model", "provider": "anthropic", "model": "claude-opus-5-5"})
    verifier = MockAgentAdapter(script=[{"tool_uses": [{"id": "v1", "name": "submit_assessment", "input": {
        "verdict": "supported", "rationale": "the read shows alpha", "counter_claim": "", "check": ""}}]}])
    agent.set_adapter("mock", None, verifier)
    out = gw.verify_claim("u", p.id, c["id"], "mock")
    sent = json.dumps(verifier.requests[0])
    assert "alpha" in sent and "alpha is the baseline" in sent and "interviewing" not in sent
    withheld = {w["ref"]: w["reason"] for w in out["assessment"]["evidence_withheld"]}
    assert withheld[theirs.id] == "belongs to another project" and withheld[diary.id].startswith("egress: no-egress-grant")
    assert withheld["task_00000000000000000000"] == "unknown task"
    assert out["assessment"]["evidence_sent"] == [mine.id, note["id"]]
    assert out["claim"]["status"] == "corroborated" and out["assessment"]["independence"] == "different vendor"
    with pytest.raises(ValueError):  # the author cannot verify itself
        agent.set_adapter("anthropic", "claude-opus-5-5", verifier)
        gw.verify_claim("u", p.id, c["id"], "anthropic", "claude-opus-5-5")


def test_deleting_a_project_removes_its_claims_and_exchange_records(tmp_path, repo):
    ci, ws, ledger, q, agent, gw = make(tmp_path)
    p = ws.create_project("u", "P")
    c = ledger.add("u", p.id, "x holds", author={"kind": "model", "provider": "anthropic", "model": "m"})
    gw.verify_claim("u", p.id, c["id"], "mock")
    assert ci.store.query("SELECT COUNT(*) AS n FROM exchange_messages")[0]["n"] == 2
    counts = ws.delete_project("u", p.id)
    assert counts["claims"] == 1 and counts["exchange_messages"] == 2
    assert ci.store.query("SELECT COUNT(*) AS n FROM claims")[0]["n"] == 0


def test_a_failing_linked_test_refutes_and_only_run_tests_tasks_count(tmp_path, repo):
    _, ws, ledger, q, _, gw = make(tmp_path)
    p = ws.create_project("u", "P")
    ws.authorize_workspace("u", p.id, str(repo), run_argv=[PY, "-c", "raise SystemExit(1)"])
    c = ledger.add("u", p.id, "the suite passes")
    read = q.propose("u", p.id, "read_file", {"path": "a.txt"})
    with pytest.raises(ValueError):
        gw.resolve_with_test("u", p.id, c["id"], read.id)
    run = q.approve("u", q.propose("u", p.id, "run_tests", {"reason": "r"}).id)
    out = gw.resolve_with_test("u", p.id, c["id"], run.id)
    assert out["status"] == "refuted" and out["resolution"]["evidence"]["exit_code"] == 1


# ---------------------------------------------------------------- routing
def test_routing_prefers_another_vendor_and_never_falls_back_to_the_mock(monkeypatch):
    up = {"claude-cli", "openai"}
    monkeypatch.setattr(registry, "availability", lambda e: (e.provider in up or e.provider == "mock", "test"))
    monkeypatch.setenv("PCIM_OPENAI_MODEL", "gpt-test")
    author = {"kind": "model", "provider": "claude-cli", "model": "opus"}
    r = registry.route("verify", {"anthropic", "openai"}, author)
    assert (r.entry.provider, r.entry.model) == ("openai", "gpt-test") and "different vendor" in r.reason
    r = registry.route("verify", {"anthropic"}, author)  # only the author's vendor is allowed
    assert (r.entry.provider, r.entry.model) == ("claude-cli", "sonnet") and "weaker independence" in r.reason
    up.clear()
    assert registry.route("verify", {"anthropic", "openai", "mock"}, author).entry is None
    assert registry.route("verify", {"mock"}, author, include_test_doubles=True).entry.provider == "mock"


# ---------------------------------------------------------------- the claude CLI bridge
def fake_claude(tmp_path, reply: dict, exit_code: int = 0):
    """A stand-in `claude` binary: logs how it was called and prints a canned result."""
    log, canned, script = tmp_path / "calls.jsonl", tmp_path / "reply.json", tmp_path / "claude"
    canned.write_text(json.dumps(reply))
    script.write_text(f"""#!{PY}
import json, os, sys
argv = sys.argv[1:]
sysfile = argv[argv.index('--system-prompt-file') + 1]
rec = {{"argv": argv, "stdin": sys.stdin.read(), "cwd": os.getcwd(), "system": open(sysfile).read(),
        "sysmode": oct(os.stat(sysfile).st_mode & 0o777), "ide": os.environ.get("CLAUDE_CODE_SSE_PORT")}}
open({str(log)!r}, "a").write(json.dumps(rec) + "\\n")
sys.stdout.write(open({str(canned)!r}).read())
sys.exit({exit_code})
""")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, log


def test_claude_cli_adapter_runs_tool_less_and_parses_the_json_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "12345")  # would attach the child to an IDE
    answer = {"text": "Reading it.", "tool_calls": [{"name": "read_file", "input": {"path": "a.txt"}},
                                                    {"name": "list_files", "input": "oops"},
                                                    {"name": "record_note", "kind": "decision", "text": "seen live"}]}
    reply = {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.0123,
             "result": "```json\n" + json.dumps(answer) + "\n```",
             "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 7},
             # the CLI also lists its own auxiliary Haiku calls; the served model is the one asked for
             "modelUsage": {"claude-haiku-5-5": {"costUSD": 0.0001}, "claude-sonnet-5-5": {"costUSD": 0.0122}}}
    binary, log = fake_claude(tmp_path, reply)
    ad = ClaudeCLIAdapter("sonnet", binary=str(binary), workdir=tmp_path / "sandbox")
    tools = [{"name": "read_file", "description": "d", "input_schema": {"type": "object"}},
             {"name": "list_files", "description": "d", "input_schema": {"type": "object"}}]
    msgs = [{"role": "user", "content": "please read a.txt"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                              {"type": "tool_use", "id": "x", "name": "read_file", "input": {"path": "a"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "alpha"}]}]
    step = ad.agent_step("SYSTEM-TEXT", msgs, tools)
    assert step.text == "Reading it." and step.model == "claude-sonnet-5-5" and step.cost_usd == 0.0123
    assert step.tool_uses[0]["input"] == {"path": "a.txt"} and step.tool_uses[0]["id"].startswith("toolu_cli_")
    assert "_unparseable" in step.tool_uses[1]["input"]  # left for CI's validator to reject
    assert step.tool_uses[2]["input"] == {"kind": "decision", "text": "seen live"}  # Haiku 5.5's slip, seen live
    rec = json.loads(log.read_text().splitlines()[0])
    a = rec["argv"]
    assert a[a.index("--tools") + 1] == "" and a[a.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in a and "--no-session-persistence" in a and a[a.index("--model") + 1] == "sonnet"
    assert "--json-schema" not in a  # no structured-output tool: nothing for the model to call natively
    assert rec["system"].startswith("You are connected to CI through a text relay") and "SYSTEM-TEXT" in rec["system"]
    assert rec["sysmode"] == "0o600" and "SYSTEM-TEXT" not in " ".join(a)  # the prompt never reaches argv
    assert rec["cwd"] == str((tmp_path / "sandbox").resolve()) and rec["ide"] is None
    assert "please read a.txt" in rec["stdin"] and "[x] result: alpha" in rec["stdin"] and "<ci_tools>" in rec["stdin"]
    assert '"tool_calls": [{"id": "x", "name": "read_file"' in rec["stdin"]  # its own past turns, in its own format
    assert not list((tmp_path / "sandbox").glob(".system-*"))  # system file removed after the call


def test_claude_cli_adapter_retries_once_when_the_reply_is_not_json(tmp_path):
    replies = ["Sure! I'll read the file now.", 'Here you go: {"text": "ok", "tool_calls": []}']
    seen = []

    def runner(argv, input, **kw):
        seen.append(input)
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({
            "subtype": "success", "is_error": False, "result": replies[len(seen) - 1], "total_cost_usd": 0.01,
            "usage": {"input_tokens": 5, "output_tokens": 2}, "modelUsage": {"claude-opus-5-5": {}}}), stderr="")
    step = ClaudeCLIAdapter(workdir=tmp_path / "s", runner=runner).agent_step("s", [{"role": "user", "content": "x"}], [])
    assert step.text == "ok" and len(seen) == 2 and "not a valid JSON object" in seen[1]
    assert step.cost_usd == pytest.approx(0.02) and step.usage.input_tokens == 10 and step.model == "claude-opus-5-5"
    replies[1] = "still prose"
    seen.clear()
    step = ClaudeCLIAdapter(workdir=tmp_path / "s", runner=runner).agent_step("s", [{"role": "user", "content": "x"}], [])
    assert step.text == "still prose" and step.tool_uses == []  # passed through as text, nothing executed


def test_claude_cli_adapter_reports_failures_as_provider_errors(tmp_path):
    binary, _ = fake_claude(tmp_path, {"type": "result", "subtype": "error_max_budget_usd", "is_error": True,
                                       "result": "budget exceeded"})
    with pytest.raises(ClaudeCLIError, match="budget"):
        ClaudeCLIAdapter(binary=str(binary), workdir=tmp_path / "s").agent_step("s", [{"role": "user", "content": "x"}], [])

    def garbage(*a, **k):
        return subprocess.CompletedProcess(a, 1, stdout="not json", stderr="Not logged in")
    with pytest.raises(ClaudeCLIError, match="Not logged in"):
        ClaudeCLIAdapter(workdir=tmp_path / "s", runner=garbage).agent_step("s", [{"role": "user", "content": "x"}], [])

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(a, 1)
    with pytest.raises(ClaudeCLIError, match="timed out"):
        ClaudeCLIAdapter(workdir=tmp_path / "s", runner=slow).agent_step("s", [{"role": "user", "content": "x"}], [])


# ---------------------------------------------------------------- OpenAI tool use
def test_openai_agent_step_translates_the_canonical_transcript():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"model": "gpt-test", "choices": [{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_9", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b"}'}},
                {"id": "call_10", "type": "function", "function": {"name": "read_file", "arguments": "{bad"}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
    ad = OpenAICompatAdapter("gpt-test", api_key="k", transport=httpx.MockTransport(handler))
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "reading"},
                                              {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "is_error": True, "content": "denied"},
                                         {"type": "text", "text": "and then?"}]}]
    step = ad.agent_step("SYS", msgs, [{"name": "read_file", "description": "d",
                                        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                                                         "required": ["path"], "additionalProperties": False}}])
    sent = seen[0]["messages"]
    assert sent[0] == {"role": "system", "content": "SYS"}
    assert sent[2]["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "a"}'}
    assert sent[3] == {"role": "tool", "tool_call_id": "call_1", "content": "did not run: denied"}
    assert sent[4] == {"role": "user", "content": "and then?"}
    assert seen[0]["tools"][0]["function"]["strict"] is True
    assert step.tool_uses[0] == {"id": "call_9", "name": "read_file", "input": {"path": "b"}}
    assert "_unparseable" in step.tool_uses[1]["input"] and step.cost_usd is None  # unpriced, never $0


# ---------------------------------------------------------------- MCP
def test_mcp_briefing_respects_project_provider_access(tmp_path):
    mcp = pytest.importorskip("mcp")
    from pcim.mcp_server import build_server

    ci, ws, *_ = make(tmp_path, providers=("anthropic",))
    p = ws.create_project("u", "Crowding", objective="decide on rho(J)")
    srv = build_server(ci, "u", provider="anthropic")

    def briefing():
        async def go():
            async with mcp.Client(srv) as c:
                r = await c.call_tool("ci_project_briefing", {"project_id": p.id})
                return r.content[0].text
        return asyncio.run(go())
    assert "decide on rho(J)" in briefing()
    ws.revoke_provider("u", p.id, "anthropic")
    out = briefing()
    assert out.startswith("error:") and "rho(J)" not in out
