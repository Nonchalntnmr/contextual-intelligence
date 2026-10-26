"""The agent runtime's guarantees: bounded turns, durable rounds, no lost updates, no
duplicate side effects, failures that are visible and leave a valid transcript."""

import json
import threading
import time

import pytest

from pcim.adapters.base import AgentStep, ProviderError, Usage
from pcim.agent import Agent, TurnLimits

from test_exchange_and_providers import make


class Scripted:
    provider, model = "mock", "scripted"

    def __init__(self, fn):
        self.fn, self.calls, self.requests = fn, 0, []

    def agent_step(self, system, messages, tools, max_tokens=None):
        self.calls += 1
        self.requests.append(json.loads(json.dumps(messages, default=str)))
        return self.fn(self.calls, messages)


def step(text="", calls=(), cost=0.0, inp=0):
    content = ([{"type": "text", "text": text}] if text else []) + [
        {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]} for c in calls]
    return AgentStep(content=content, text=text, tool_uses=list(calls), stop_reason="tool_use" if calls else "end_turn",
                     usage=Usage(input_tokens=inp, output_tokens=5), cost_usd=cost, model="scripted")


def notes(n, k):
    return [{"id": f"t{n}_{i}", "name": "record_note", "input": {"kind": "next_step", "text": f"step {n}.{i}"}}
            for i in range(k)]


def valid_transcript(messages):
    """Every assistant tool_use is answered by a tool_result in the next message."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            ids = {b["id"] for b in m["content"] if b.get("type") == "tool_use"}
            if ids:
                nxt = messages[i + 1]["content"] if i + 1 < len(messages) else []
                answered = {b.get("tool_use_id") for b in nxt if isinstance(b, dict)}
                if not ids <= answered:
                    return False
    return True


@pytest.fixture
def env(tmp_path):
    ci, ws, ledger, q, agent, gw = make(tmp_path)
    p = ws.create_project("u", "P")
    return ci, ws, q, agent, p


def with_limits(agent, **kw):
    a = Agent(agent.ci, agent.ws, agent.queue, agent.ledger, limits=TurnLimits(**kw))
    return a


def test_tool_call_limits_answer_every_call_and_stop_the_turn(env):
    ci, ws, q, agent, p = env
    a = with_limits(agent, max_tool_calls_per_step=3, max_tool_calls=5)
    ad = Scripted(lambda n, m: step(calls=notes(n, 4)))
    a.set_adapter("mock", None, ad)
    turn = a.send("u", p.id, "go")
    assert turn.status == "stopped" and turn.stop_reason == "max_tool_calls"
    assert len(q.list("u", p.id)) == 5 and ad.calls == 2
    msgs = a._load("u", p.id, turn.session_id)["messages"]
    assert valid_transcript(msgs)
    refused = [b for b in msgs[-1]["content"] if b.get("is_error")]
    assert len(refused) == 2 and "limit on tool calls" in refused[0]["content"]


@pytest.mark.parametrize("limits,fn,reason,calls", [
    ({"max_rounds": 3}, lambda n, m: step(calls=notes(n, 1)), "max_rounds", 3),
    ({"max_cost_usd": 1.0}, lambda n, m: step(calls=notes(n, 1), cost=0.6), "max_cost", 2),
    ({"max_input_tokens": 1500}, lambda n, m: step(calls=notes(n, 1), inp=1000), "max_tokens", 2),
])
def test_each_turn_limit_binds_before_the_next_model_call(env, limits, fn, reason, calls):
    ci, ws, q, agent, p = env
    a = with_limits(agent, **limits)
    ad = Scripted(fn)
    a.set_adapter("mock", None, ad)
    turn = a.send("u", p.id, "go")
    assert (turn.status, turn.stop_reason, ad.calls) == ("stopped", reason, calls)
    assert any(e["event"] == "session.stopped" and e["payload"]["why"] == reason for e in ci.audit.entries("u"))


def test_the_time_limit_binds(env):
    ci, ws, q, agent, p = env
    a = with_limits(agent, max_seconds=0.2)

    def slow(n, m):
        time.sleep(0.15)
        return step(calls=notes(n, 1))
    a.set_adapter("mock", None, Scripted(slow))
    assert a.send("u", p.id, "go").stop_reason == "max_seconds"


def test_a_halt_during_a_turn_stops_before_the_next_model_call(env):
    ci, ws, q, agent, p = env

    def halt_then_tool(n, m):
        ws.set_halted("u", True)  # as if the person pressed Stop all while the model was thinking
        return step(calls=notes(n, 1))
    ad = Scripted(halt_then_tool)
    agent.set_adapter("mock", None, ad)
    turn = agent.send("u", p.id, "go")
    assert turn.stop_reason == "halted" and ad.calls == 1
    assert turn.tasks[0]["status"] == "denied"  # the proposed tool was refused by the governor too
    with pytest.raises(RuntimeError, match="halted"):
        agent.send("u", p.id, "and now?")


def test_a_provider_failure_mid_turn_is_audited_and_leaves_a_valid_transcript(env):
    ci, ws, q, agent, p = env

    def fails_second(n, m):
        if n == 2:
            raise ProviderError("overloaded")
        return step(calls=notes(n, 1)) if n == 1 else step("recovered")
    ad = Scripted(fails_second)
    agent.set_adapter("mock", None, ad)
    with pytest.raises(ProviderError):
        agent.send("u", p.id, "go")
    sid = agent.open_session("u", p.id)
    msgs = agent._load("u", p.id, sid)["messages"]
    assert valid_transcript(msgs) and msgs[-1]["content"][0]["type"] == "tool_result"  # round 1 was saved
    assert any(e["event"] == "model.error" and e["payload"]["error"] == "ProviderError" for e in ci.audit.entries("u"))
    assert agent.send("u", p.id, "try again").text == "recovered"
    assert valid_transcript(ad.requests[-1])


def test_concurrent_turns_on_one_session_lose_nothing(env):
    """B arrives while A is waiting on the model. Without reloading state under the session
    lock, B would continue from the state it read before A's reply and overwrite that reply."""
    ci, ws, q, agent, p = env
    in_model = threading.Event()

    def slow(n, m):
        in_model.set()
        time.sleep(0.3)
        return step(f"reply {n}")
    ad = Scripted(slow)
    agent.set_adapter("mock", None, ad)
    agent.send("u", p.id, "open the session")
    in_model.clear()
    a = threading.Thread(target=agent.send, args=("u", p.id, "first"))
    a.start()
    assert in_model.wait(5)
    b = threading.Thread(target=agent.send, args=("u", p.id, "second"))
    b.start()
    a.join()
    b.join()
    sid = agent.open_session("u", p.id)
    replies = [e["text"] for e in agent.transcript("u", p.id)["events"] if e["role"] == "ci"]
    assert replies == ["reply 1", "reply 2", "reply 3"]
    roles = [m["role"] for m in agent._load("u", p.id, sid)["messages"]]
    assert roles == ["user", "assistant"] * 3  # strictly alternating, nothing merged or dropped


def test_the_same_model_call_never_executes_twice(env, tmp_path):
    ci, ws, q, agent, p = env
    repo = tmp_path / "r"
    repo.mkdir()
    ws.authorize_workspace("u", p.id, str(repo), allow_write=True)
    t1 = q.propose("u", p.id, "write_file", {"path": "x.txt", "content": "1", "reason": "r"}, session_id="ses_x",
                   tool_use_id="toolu_same")
    t2 = q.propose("u", p.id, "write_file", {"path": "x.txt", "content": "2", "reason": "r"}, session_id="ses_x",
                   tool_use_id="toolu_same")
    assert t1.id == t2.id and len(q.list("u", p.id)) == 1


def test_an_internal_error_fails_the_task_visibly(env, monkeypatch):
    ci, ws, q, agent, p = env

    def broken(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(ws, "add_note", broken)
    t = q.propose("u", p.id, "record_note", {"kind": "decision", "text": "x"})
    assert t.status == "failed" and t.reason == "internal error (RuntimeError)"


def test_the_memory_cache_never_serves_a_withdrawn_item(tmp_path):
    from pcim.engine import PersonalCI
    server = PersonalCI(tmp_path / "d", seed=0)
    server.onboard("u", providers=("mock",))
    keep = server.remember("u", "uses 25bps costs", "constraint").item
    drop = server.remember("u", "private note about the fund", "fact").item
    assert {i.id for i in server.memory.current("u")} == {keep.id, drop.id}  # now cached
    cli = PersonalCI(tmp_path / "d", seed=1)  # another process on the same store, e.g. `pcim delete`
    cli.memory.retract("u", drop.id)
    assert {i.id for i in server.memory.current("u")} == {keep.id}
    ids, _ = server.memory.embeddings("u")
    newer = cli.remember("u", "a fact written elsewhere", "fact").item
    assert newer.id in server.memory.embeddings("u")[0] and newer.id not in ids  # the index follows too
    server.memory.delete("u", keep.id)
    assert keep.id not in {i.id for i in server.memory.current("u")}
    server.memory.current("u")[0].content = "mutated"
    assert server.memory.current("u")[0].content != "mutated"  # callers get copies


def test_greedy_propensities_equal_the_sampling_loop_they_replace():
    import numpy as np
    from pcim.policy.candidates import CandidateSet
    from pcim.policy.features import featurize
    from pcim.policy.learned import BayesLinearPolicy
    from pcim.policy.prior import designed_prior
    rng = np.random.default_rng(0)
    pol = BayesLinearPolicy(designed_prior(), greedy=True, rank="density")
    for _ in range(200):
        n = int(rng.integers(1, 60))
        emb = rng.normal(size=(n, 16))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cs = CandidateSet([f"m{i}" for i in range(n)], rng.uniform(-0.2, 1, n), rng.integers(0, 6, n),
                          rng.uniform(0, 900, n), rng.uniform(0.3, 1, n), rng.integers(5, 120, n), np.zeros(n, bool),
                          rng.random(n) < 0.1, rng.integers(0, 3, n), emb)
        p, budget = rng.dirichlet(np.ones(5)), int(rng.integers(50, 1500))
        sel = pol.select(cs, p, budget, np.random.default_rng(1), propensity_samples=32)
        F, beta, old = featurize(cs, p), pol.mean(), np.zeros(n)
        for _ in range(32):  # what the loop computed before
            old[pol._choose(beta, F, cs, budget)[0]] += 1
        assert np.array_equal(sel.propensities, old / 32)


def test_usage_is_accounted_per_turn_and_per_session(env):
    ci, ws, q, agent, p = env
    agent.set_adapter("mock", None, Scripted(lambda n, m: step(calls=notes(n, 1), inp=100) if n % 2 else step("ok", inp=50)))
    t1 = agent.send("u", p.id, "one")
    assert t1.usage == {"input_tokens": 150, "output_tokens": 10, "steps": 2}
    agent.send("u", p.id, "two")
    assert agent.transcript("u", p.id)["usage"] == {"input_tokens": 300, "output_tokens": 20, "steps": 4}
