"""Adapters against recorded wire shapes (no network), and the learning loop end to end."""

import json

import httpx
import numpy as np
import pytest

from pcim.adapters import ContextPacket, PacketItem, render
from pcim.adapters.gemini import GeminiAdapter
from pcim.adapters.openai_compat import OpenAICompatAdapter
from pcim.adapters.pricing import cost_usd
from pcim.adapters.base import Usage
from pcim.assist import GROWTH_MODES, AssistPolicy
from pcim.engine import PersonalCI
from pcim.feedback import reward_from_signals
from pcim.growth import BKT, next_challenge
from pcim.intent import classify_task, should_clarify, value_of_information
from pcim.learning import noninferiority_gate


def _packet():
    item = PacketItem("mem_1", "constraint", "Vegetarian since May 2026", "user", "user_stated", "2026-05-02", 1.0)
    pin = PacketItem("mem_2", "preference", "Answers in metric units", "user", "user_confirmed", "2025-01-01", 1.0)
    return ContextPacket("int_1", "Plan dinner", ["Be concise."], [pin], [item], nonce="abc123")


# ---------------------------------------------------------------- rendering
def test_render_places_trusted_and_untrusted_content_apart():
    p = render(_packet())
    assert "Answers in metric units" in p.system and "Vegetarian" not in p.system
    assert '<personal_context nonce="abc123">' in p.user and p.user.endswith("Request:\nPlan dinner")
    assert len(p.digest()) == 64


# ---------------------------------------------------------------- anthropic (official SDK)
def test_anthropic_adapter_wire_format_caching_and_fallbacks():
    anthropic = pytest.importorskip("anthropic")
    import httpx2
    from pcim.adapters.anthropic import AnthropicAdapter

    seen = {}

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen["url"], seen["headers"], seen["body"] = req.url.path, dict(req.headers), json.loads(req.content)
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "Try the lentil place."}], "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 30, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 600},
        })

    client = anthropic.Anthropic(api_key="test", http_client=anthropic.DefaultHttpxClient(
        transport=httpx2.MockTransport(handler)))
    ad = AnthropicAdapter("claude-opus-5-5", client=client)
    resp = ad.generate(render(_packet()))
    body = seen["body"]
    assert seen["url"].endswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in seen["headers"].get("anthropic-beta", "")
    assert body["fallbacks"] == "default"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"] == {"effort": "medium"}
    assert "temperature" not in body and "top_p" not in body
    assert body["messages"][0]["role"] == "user" and "Vegetarian" in body["messages"][0]["content"]
    assert resp.text == "Try the lentil place." and resp.usage.cache_read_tokens == 600
    assert resp.cost_usd == pytest.approx((120 * 4 + 30 * 20 + 600 * 0.20) / 1e6)


def test_anthropic_refusal_is_checked_before_content_and_haiku_has_no_fallback():
    anthropic = pytest.importorskip("anthropic")
    import httpx2
    from pcim.adapters.anthropic import AnthropicAdapter

    seen = {}

    def handler(req):
        seen["body"], seen["headers"] = json.loads(req.content), dict(req.headers)
        return httpx2.Response(200, json={
            "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "content": [],
            "stop_reason": "refusal", "stop_sequence": None,
            "stop_details": {"type": "refusal", "category": "cyber", "explanation": "x"},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        })

    client = anthropic.Anthropic(api_key="t", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    resp = AnthropicAdapter("claude-haiku-5-5", client=client).generate(render(_packet()))
    assert resp.refusal and "declined" in resp.text
    assert "fallbacks" not in seen["body"] and "server-side-fallback" not in seen["headers"].get("anthropic-beta", "")


# ---------------------------------------------------------------- openai-compatible, gemini
def test_openai_compat_wire_format():
    seen = {}

    def handler(req):
        seen["path"], seen["body"], seen["auth"] = req.url.path, json.loads(req.content), req.headers.get("authorization")
        return httpx.Response(200, json={"id": "c1", "model": "local-model", "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 20}}})

    ad = OpenAICompatAdapter("local-model", provider="local", base_url="http://127.0.0.1:9/v1",
                             transport=httpx.MockTransport(handler))
    r = ad.generate(render(_packet()))
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] is None
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]
    assert seen["body"]["max_tokens"] == 4096
    assert r.text == "ok" and r.usage.input_tokens == 30 and r.usage.cache_read_tokens == 20
    assert r.cost_usd is None  # unpriced model: unknown, not free


def test_gemini_wire_format_and_block():
    seen = {}

    def handler(req):
        seen["path"], seen["body"], seen["key"] = req.url.path, json.loads(req.content), req.headers.get("x-goog-api-key")
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 3, "thoughtsTokenCount": 7}})

    ad = GeminiAdapter("gemini-test", api_key="k", transport=httpx.MockTransport(handler))
    r = ad.generate(render(_packet()))
    assert seen["path"].endswith("/models/gemini-test:generateContent") and seen["key"] == "k"
    assert "systemInstruction" in seen["body"] and seen["body"]["contents"][0]["role"] == "user"
    assert r.text == "hi" and r.usage.output_tokens == 10

    def blocked(req):
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {}})
    r2 = GeminiAdapter("gemini-test", api_key="k", transport=httpx.MockTransport(blocked)).generate(render(_packet()))
    assert r2.refusal


def test_pricing_unknown_is_none_known_is_exact():
    assert cost_usd("no-such-model", Usage(100, 100)) is None
    assert cost_usd("claude-sonnet-5-5", Usage(1_000_000, 0)) == pytest.approx(2.0)


# ---------------------------------------------------------------- reward, assist, growth, intent
def test_verified_outcome_dominates_satisfaction():
    assert reward_from_signals({"task_success": False, "rating": 5}) == 0.0
    assert reward_from_signals({"rating": 5, "corrected": True}) < reward_from_signals({"rating": 5})
    assert reward_from_signals({}) is None
    assert reward_from_signals({"reasked": True}) == pytest.approx(0.8)


def test_growth_goals_restrict_assistance_modes():
    ap = AssistPolicy(growth_tasks={"learning"})
    rng = np.random.default_rng(0)
    for _ in range(50):
        ap.update("learning", "direct", 1.0)  # even if "direct" looks great...
    assert all(ap.select("learning", rng) in GROWTH_MODES for _ in range(50))
    assert ap.select("learning", rng, requested="direct") == "direct"  # explicit request wins


def test_bkt_learns_and_assisted_attempts_do_not_count(tmp_path):
    s = BKT()
    for _ in range(8):
        s.observe(True)
    assert s.p_mastery > 0.9
    ci = PersonalCI(tmp_path, seed=0)
    ci.onboard("u")
    ci.probe("u", "bayes rule", correct=True, assisted=True)
    assert ci.store.list_skills("u") == {}
    ci.probe("u", "bayes rule", correct=True)
    assert ci.store.list_skills("u")["bayes rule"]["n_probes"] == 1
    assert next_challenge({"a": BKT(p_mastery=0.95), "b": BKT(p_mastery=0.6)}) in ("a", "b")


def test_voi_is_zero_when_ambiguity_does_not_change_the_action():
    assert value_of_information([0.5, 0.5], [[1.0, 0.0], [1.0, 0.0]]) == 0.0
    ask, voi = should_clarify([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], cost=0.2)
    assert ask and voi == pytest.approx(0.5)
    assert max(classify_task("fix this python bug in my test"), key=classify_task("fix this python bug in my test").get) == "code"


def test_noninferiority_gate():
    rng = np.random.default_rng(0)
    inc = rng.uniform(0.5, 0.7, 200)
    assert noninferiority_gate(inc + 0.02, inc).promote
    assert not noninferiority_gate(inc - 0.05, inc).promote


# ---------------------------------------------------------------- the loop
def test_feedback_updates_the_policy_versions_and_rollback(tmp_path):
    ci = PersonalCI(tmp_path, seed=1, propensity_samples=4)
    ci.onboard("u")
    for k in range(6):
        ci.remember("u", f"constraint {k}: no meetings on Fridays", "constraint")
    pol0 = ci.learner.policy_for("u").mean().copy()
    for n in range(12):
        res = ci.ask("u", "plan my week of meetings")
        ci.feedback("u", res.interaction_id, rating=5)
    pol = ci.learner.policy_for("u")
    assert pol.n_updates == 12 and not np.allclose(pol.mean(), pol0)
    versions = ci.learner.versions("u")
    assert [v["version"] for v in versions] == [1, 2]  # snapshot at 1st and 10th update
    ci.learner.rollback("u", 1)
    assert ci.learner.policy_for("u").n_updates == 1
    m = ci.metrics("u")
    assert m["interactions"] == 12 and m["with_feedback"] == 12 and m["audit_chain_ok"]
    r = ci.receipt("u", res.interaction_id)
    assert r["sent"] and r["task"] == "plan my week of meetings"
