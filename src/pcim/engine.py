"""Personal CI — the orchestration loop over every PCIM layer.

    ask(task)
      C  classify the task (soft type distribution)
      B  current memory → multi-channel candidates (similarity, standing items, recency, pins)
      ⊕  egress guard: drop everything this provider may not see — BEFORE the policy runs,
         so the policy can never learn to rely on context it is not allowed to send
      D  learned policy selects under the token budget, logging inclusion propensities
      E  assistance mode (bandit; growth goals restrict it)
      F  ContextPacket → render → provider adapter
      G  interaction logged: ids, propensities, x(S), tokens, cost, latency, hash of the
         exact prompt (the prompt text itself is not stored)
    feedback(signals)
      G  signals → reward (feedback.py; verified outcomes dominate, satisfaction is weak)
      H  posterior update, versioned and auditable; assistance-mode update

Every personal datum touched along the way is encrypted at rest under a per-record key,
and every egress decision is in the hash-chained audit log.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np

from .adapters import ContextPacket, PacketItem, get_adapter, render
from .assist import DIRECTIVES, AssistPolicy
from .embed import Embedder, HashingEmbedder
from .feedback import reward_from_signals
from .growth import BKT, next_challenge
from .intent import classify_task
from .learning import ContinualLearner
from .memory import TemporalMemory, effective_confidence
from .policy.candidates import build_candidates
from .policy.learned import PopulationPrior
from .policy.prior import designed_prior
from .profile import intelligence_dna
from .registry import vendor_of
from .sovereignty.audit import AuditLog
from .sovereignty.consent import ConsentLedger
from .sovereignty.crypto import KeyRing
from .sovereignty.egress import CONFIRMED_FLAG, INSTRUCTION_FLAG, EgressGuard
from .store import Store
from .types import KINDS, TASK_TYPES, Sensitivity, Trust, new_id, utcnow

ASSIST_STATE = "pcim-assist"


@dataclasses.dataclass
class AskResult:
    interaction_id: str
    text: str
    receipt: dict


@dataclasses.dataclass
class ContextSelection:
    probs: dict
    task_type: str
    chosen: list
    sel: object
    policy: object
    decision: object
    n_candidates: int
    profile: list
    memories: list
    explain: list = dataclasses.field(default_factory=list)  # per candidate: ids and numbers, no content
    out_of_scope: int = 0  # current items scoped to another project, never candidates here


SCOPE_PREFIX = "project:"


def in_scope(item, project_id: str | None) -> bool:
    """An item tagged `project:<id>` belongs to that project only; an untagged item is user-wide."""
    scopes = [t[len(SCOPE_PREFIX):] for t in item.tags if t.startswith(SCOPE_PREFIX)]
    return not scopes or project_id in scopes


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class PersonalCI:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        store: Store | None = None,
        embedder: Embedder | None = None,
        prior: PopulationPrior | None = None,
        budget_tokens: int = 1200,
        propensity_samples: int = 32,
        seed: int | None = None,
    ):
        if store is None:
            if data_dir is None:
                store = Store(":memory:", KeyRing.ephemeral())
            else:
                d = Path(data_dir)
                d.mkdir(parents=True, exist_ok=True)
                store = Store(d / "pcim.db", KeyRing.load_or_create(d / "master.key"))
        self.store = store
        self.embedder = embedder or HashingEmbedder()
        self.audit = AuditLog(store)
        self.ledger = ConsentLedger(store, self.audit)
        self.egress = EgressGuard(store, self.ledger, self.audit)
        self.memory = TemporalMemory(store, self.embedder, self.audit, self.ledger)
        self.learner = ContinualLearner(store, self.audit, prior or designed_prior())
        self.budget_tokens = budget_tokens
        self.propensity_samples = propensity_samples
        self.rng = np.random.default_rng(seed)
        self._adapters: dict[tuple[str, str | None], object] = {}

    # ------------------------------------------------------------------ onboarding
    def onboard(self, user_id: str, display_name: str = "", providers: tuple[str, ...] = ("mock",),
                max_sensitivity: Sensitivity = Sensitivity.PERSONAL) -> dict:
        """Create a user and grant egress of *their own statements* to the chosen providers.

        Nothing else is granted. Every connector and document source needs its own store
        and egress grants, given explicitly (consent()).
        """
        vendors = list(dict.fromkeys(p if p == "*" else vendor_of(p) for p in providers))  # ValueError if unknown
        self.store.create_user(user_id, display_name, settings={"providers": list(providers)})
        for v in vendors:
            self.ledger.grant(user_id, "user", "egress", v, max_sensitivity, note="onboarding: own statements")
        token = secrets.token_urlsafe(32)
        self.store.set_token_hash(user_id, _hash_token(token))
        self.audit.append(user_id, "user.onboard", providers=list(providers))
        return {"user_id": user_id, "api_token": token}

    def user_for_token(self, token: str) -> str | None:
        return self.store.user_for_token_hash(_hash_token(token))

    def consent(self, user_id: str, source: str, purpose: str, provider: str = "*",
                max_sensitivity: Sensitivity = Sensitivity.PERSONAL, note: str = ""):
        vendor = provider if provider == "*" else vendor_of(provider)  # consent is per vendor
        return self.ledger.grant(user_id, source, purpose, vendor, max_sensitivity, note=note)

    def revoke(self, user_id: str, grant_id: str, delete_stored: bool = False) -> dict:
        grants = {g.id: g for g in self.ledger.grants(user_id)}
        if grant_id not in grants:
            raise KeyError(grant_id)
        ok = self.ledger.revoke(user_id, grant_id)
        deleted = []
        g = grants[grant_id]
        if delete_stored and g.purpose == "store":
            import fnmatch
            for it in self.memory.all(user_id):
                if fnmatch.fnmatchcase(it.source, g.source) and self.store.get_item(it.id) is not None:
                    deleted += self.memory.delete(user_id, it.id, cascade=True)
        return {"revoked": ok, "deleted_items": deleted}

    # ------------------------------------------------------------------ memory
    def remember(self, user_id: str, content: str, kind: str = "fact", **kw):
        return self.memory.remember(user_id, content, kind, **kw)

    def ingest_text(self, user_id: str, text: str, source: str, title: str | None = None,
                    sensitivity: Sensitivity = Sensitivity.PERSONAL, kind: str = "fact",
                    tags: tuple[str, ...] = ()) -> list:
        """Split a document into paragraph items. Documents are UNTRUSTED: they can carry
        injected instructions, so they are stored as data and can never become standing
        preferences without the person's confirmation."""
        paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
        out = []
        for n, p in enumerate(paras):
            out.append(self.memory.remember(
                user_id, p, kind, source=source, trust=Trust.UNTRUSTED, sensitivity=sensitivity,
                confidence=0.7, provenance={"title": title or "", "paragraph": n}, tags=tags,
            ))
        return out

    # ------------------------------------------------------------------ the loop
    def _adapter(self, provider: str, model: str | None):
        key = (provider, model)
        if key not in self._adapters:
            self._adapters[key] = get_adapter(provider, model)
        return self._adapters[key]

    def set_adapter(self, provider: str, model: str | None, adapter) -> None:
        """Inject an adapter (tests, or a pre-configured client)."""
        self._adapters[(provider, model)] = adapter

    def _assist(self, user_id: str) -> AssistPolicy:
        loaded = self.store.load_policy_state(user_id, ASSIST_STATE)
        return AssistPolicy.from_dict(json.loads(loaded[1])) if loaded else AssistPolicy()

    def _save_assist(self, user_id: str, ap: AssistPolicy) -> None:
        loaded = self.store.load_policy_state(user_id, ASSIST_STATE)
        blob = json.dumps(ap.to_dict()).encode()
        if loaded:
            self.store.overwrite_policy_state(user_id, ASSIST_STATE, loaded[0], blob)
        else:
            self.store.save_policy_state(user_id, ASSIST_STATE, blob, note="assist policy")

    def set_growth_goals(self, user_id: str, task_types: list[str]) -> None:
        bad = set(task_types) - set(TASK_TYPES)
        if bad:
            raise ValueError(f"unknown task types {sorted(bad)}")
        ap = self._assist(user_id)
        ap.growth_tasks = set(task_types)
        self._save_assist(user_id, ap)
        self.audit.append(user_id, "assist.growth_goals", task_types=sorted(task_types))

    def select_context(self, user_id: str, task: str, provider: str, model_label: str, budget: int,
                       ref: str, project_id: str | None = None) -> "ContextSelection":
        """C → scope → B → egress → D: what this provider may see, and what the policy chose from it.

        Memory scoped to a project is a candidate only for that project; with no project
        (a one-shot ask, an MCP request without one) only user-wide memory is."""
        if self.store.get_user(user_id) is None:
            raise KeyError(f"unknown user {user_id}")
        now = utcnow()
        probs = classify_task(task)
        p_vec = np.array([probs[t] for t in TASK_TYPES])
        everything = self.memory.current(user_id, now)
        current = [i for i in everything if in_scope(i, project_id)]
        ids, mat = self.memory.embeddings(user_id)
        emb = {i: mat[j] for j, i in enumerate(ids)} if mat is not None else {}
        by_id = {i.id: i for i in current}
        task_emb = self.embedder.embed([task])[0]
        # egress before policy: the policy never learns to rely on what it may not send.
        # Consent is per vendor (who receives the data), not per transport.
        decision = self.egress.filter(user_id, current, vendor_of(provider), model_label, ref=ref)
        cs = build_candidates(decision.allowed, emb, task_emb, effective_confidence, now)
        pol = self.learner.policy_for(user_id)
        sel = pol.select(cs, p_vec, budget, self.rng, propensity_samples=self.propensity_samples)
        chosen = [by_id[cs.ids[j]] for j in sel.idx]

        def pitem(i):
            return PacketItem(
                id=i.id, kind=i.kind, content=i.content, source=i.source, trust=i.trust.value,
                valid_from=i.valid_from.isoformat(), confidence=round(effective_confidence(i, now), 2),
                conflicted="conflicted" in i.flags,
                confirmed_instructions=INSTRUCTION_FLAG in i.flags and CONFIRMED_FLAG in i.flags,
            )
        trusted = lambda i: i.pinned and i.trust.rank >= Trust.USER_STATED.rank  # noqa: E731
        picked = {int(j) for j in sel.idx}
        explain = [{"id": cs.ids[j], "selected": j in picked, "kind": KINDS[int(cs.kind[j])],
                    "pinned": bool(cs.pinned[j]), "similarity": round(float(cs.sim[j]), 3),
                    "age_days": round(float(cs.age_days[j]), 1), "confidence": round(float(cs.confidence[j]), 2),
                    "tokens": int(cs.tokens[j]), "score": round(float(sel.scores[j]), 4)} for j in range(len(cs))]
        explain.sort(key=lambda e: (not e["selected"], -e["score"]))
        return ContextSelection(
            probs=probs, task_type=max(probs, key=probs.get), chosen=chosen, sel=sel, policy=pol,
            decision=decision, n_candidates=len(cs),
            profile=[pitem(i) for i in chosen if trusted(i)], memories=[pitem(i) for i in chosen if not trusted(i)],
            explain=explain, out_of_scope=len(everything) - len(current),
        )

    def ask(self, user_id: str, task: str, provider: str = "mock", model: str | None = None,
            budget_tokens: int | None = None, assist_mode: str | None = None) -> AskResult:
        budget = budget_tokens or self.budget_tokens
        now = utcnow()
        adapter = self._adapter(provider, model)
        interaction_id = new_id("int")
        cx = self.select_context(user_id, task, provider, adapter.model, budget, ref=interaction_id)
        probs, task_type, chosen, sel, pol, decision = cx.probs, cx.task_type, cx.chosen, cx.sel, cx.policy, cx.decision
        profile, memories = cx.profile, cx.memories
        cs_len = cx.n_candidates

        ap = self._assist(user_id)
        mode = ap.select(task_type, self.rng, requested=assist_mode)
        packet = ContextPacket(
            packet_id=interaction_id, task=task, directives=[DIRECTIVES[mode]], profile=profile,
            memories=memories, nonce=secrets.token_hex(8),
            meta={"policy": pol.name, "version": pol.version, "budget": budget},
        )
        prompt = render(packet)
        resp = adapter.generate(prompt)

        # G — log
        props = sel.propensities.tolist() if sel.propensities is not None else []
        receipt = {
            "provider": provider, "model": resp.model, "assist_mode": mode, "task_type_probs": probs,
            "sent": [{"id": i.id, "kind": i.kind, "source": i.source, "trust": i.trust.value,
                      "where": "profile" if i.pinned and i.trust.rank >= Trust.USER_STATED.rank else "notes"}
                     for i in chosen],
            "withheld_by_egress": [{"id": i, "reason": r} for i, r in decision.blocked],
            "candidates_considered": cs_len,
            "selection": cx.explain,
            "out_of_scope": cx.out_of_scope,
            "context_tokens": int(sum(i.tokens for i in chosen)),
            "budget": budget, "policy": pol.name, "policy_version": pol.version,
            "usage": dataclasses.asdict(resp.usage), "cost_usd": resp.cost_usd, "latency_ms": round(resp.latency_ms, 1),
            "stop_reason": resp.stop_reason, "refusal": resp.refusal, "prompt_sha256": prompt.digest(),
            "egress_audit_hash": decision.audit_hash,
        }
        self.store.insert_interaction(
            {
                "id": interaction_id, "user_id": user_id, "ts": now.isoformat(), "task_type": task_type,
                "provider": provider, "model": resp.model, "policy": pol.name, "policy_version": pol.version,
                "assist_mode": mode, "selected": [i.id for i in chosen], "propensities": props,
                "x_vector": sel.x.tolist() if sel.x is not None else None,
                "tokens_context": receipt["context_tokens"], "tokens_in": resp.usage.input_tokens,
                "tokens_out": resp.usage.output_tokens, "cache_read": resp.usage.cache_read_tokens,
                "cost_usd": resp.cost_usd, "latency_ms": resp.latency_ms, "stop_reason": resp.stop_reason,
                "packet_hash": prompt.digest(),
            },
            task_text=task, response_text=resp.text, receipt=receipt,
        )
        self.audit.append(user_id, "model.call", interaction=interaction_id, provider=provider, model=resp.model,
                          items=len(chosen), prompt_sha256=prompt.digest(), cost_usd=resp.cost_usd)
        return AskResult(interaction_id, resp.text, receipt)

    def feedback(self, user_id: str, interaction_id: str, **signals) -> float | None:
        rec = self.store.get_interaction(interaction_id, with_payload=False)
        if rec is None or rec["user_id"] != user_id:
            raise KeyError(interaction_id)
        r = reward_from_signals(signals)
        self.store.set_feedback(interaction_id, signals, r)
        if r is not None and rec.get("x_vector") is not None:
            version = self.learner.update(user_id, np.array(rec["x_vector"]), r)
            ap = self._assist(user_id)
            ap.update(rec["task_type"], rec["assist_mode"], r)
            self._save_assist(user_id, ap)
            self.audit.append(user_id, "feedback", interaction=interaction_id, reward=round(r, 4),
                              policy_version=version)
        else:
            self.audit.append(user_id, "feedback", interaction=interaction_id, reward=None)
        return r

    # ------------------------------------------------------------------ growth
    def probe(self, user_id: str, skill: str, correct: bool, assisted: bool = False) -> dict:
        """Record an attempt at `skill`. Assisted attempts are logged but never move mastery."""
        skills = self.store.list_skills(user_id)
        state = BKT(**skills[skill]) if skill in skills else BKT()
        if not assisted:
            state.observe(bool(correct))
            self.store.put_skill(user_id, skill, state.to_dict())
        self.audit.append(user_id, "growth.probe", skill=skill[:80], correct=bool(correct), assisted=assisted)
        return {"skill": skill, "p_mastery": state.p_mastery, "counted": not assisted}

    def next_challenge(self, user_id: str) -> str | None:
        return next_challenge({k: BKT(**v) for k, v in self.store.list_skills(user_id).items()})

    # ------------------------------------------------------------------ inspection
    def receipt(self, user_id: str, interaction_id: str) -> dict:
        rec = self.store.get_interaction(interaction_id)
        if rec is None or rec["user_id"] != user_id:
            raise KeyError(interaction_id)
        items = []
        for s in rec["receipt"]["sent"]:
            it = self.store.get_item(s["id"])
            items.append({**s, "content": it.content if it else "[deleted since — the key is destroyed]"})
        return {**rec["receipt"], "sent": items, "task": rec["task"], "response": rec["response"],
                "feedback": rec.get("feedback"), "reward": rec.get("reward")}

    def dna(self, user_id: str) -> dict:
        return intelligence_dna(self.memory, user_id, self.learner.policy_for(user_id),
                                self.store.list_skills(user_id))

    def metrics(self, user_id: str) -> dict:
        rows = self.store.list_interactions(user_id)
        by_provider: dict[str, dict] = {}
        for r in rows:
            p = by_provider.setdefault(f"{r['provider']}/{r['model']}", {"calls": 0, "cost_usd": 0.0, "cost_unknown": 0,
                                                                          "tokens_in": 0, "tokens_out": 0, "latency_ms": []})
            p["calls"] += 1
            if r["cost_usd"] is None:
                p["cost_unknown"] += 1
            else:
                p["cost_usd"] += r["cost_usd"]
            p["tokens_in"] += r["tokens_in"] or 0
            p["tokens_out"] += r["tokens_out"] or 0
            p["latency_ms"].append(r["latency_ms"] or 0)
        for p in by_provider.values():
            lat = p.pop("latency_ms")
            p["latency_ms_p50"] = float(np.median(lat)) if lat else None
        rewards = [r["reward"] for r in rows if r["reward"] is not None]
        window = 10
        curve = [float(np.mean(rewards[i:i + window])) for i in range(0, len(rewards), window)]
        return {
            "interactions": len(rows), "with_feedback": len(rewards),
            "mean_reward": float(np.mean(rewards)) if rewards else None,
            "reward_curve_per_10": curve, "context_tokens_mean": float(np.mean([r["tokens_context"] or 0 for r in rows])) if rows else 0.0,
            "by_provider": by_provider, "policy_versions": self.learner.versions(user_id),
            "audit_chain_ok": self.audit.verify()[0],
            "caveat": "Reward is from your own feedback; a rising curve is not evidence CI is better than a "
                      "baseline. That needs the paired comparison in `pcim compare` / docs/04 G2.",
        }

    def export(self, user_id: str) -> dict:
        return {
            "format": "pcim-export/1",
            "user": self.store.get_user(user_id),
            "memory": self.memory.export(user_id),
            "consent": [dataclasses.asdict(g) | {"max_sensitivity": g.max_sensitivity.value}
                        for g in self.ledger.grants(user_id)],
            "interactions": self.store.list_interactions(user_id, with_payload=True),
            "dna": self.dna(user_id),
            "audit": self.audit.entries(user_id),
            "audit_head": self.audit.head(),
        }

    def erase(self, user_id: str) -> dict:
        """Delete the user and everything about them; destroys every key. The audit log keeps
        pseudonymous ids of what was erased (no content), as the record of the erasure."""
        counts = self.store.delete_user(user_id)
        self.learner._cache.pop(user_id, None)
        self.memory._index.pop(user_id, None)
        self.audit.append(user_id, "user.erase", **{k: int(v) for k, v in counts.items()})
        return counts
