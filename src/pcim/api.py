"""HTTP API and dashboard for Personal CI.

Local-first: `pcim serve` binds 127.0.0.1. Every route except onboarding requires the
bearer token issued at onboarding (stored hashed). Every route goes through PersonalCI,
so there is no HTTP path around consent, egress, or the audit log.

Onboarding over HTTP is allowed only while the store has no users, or when
PCIM_ALLOW_ONBOARD=1 — otherwise anyone who can reach the port could mint an account.
"""

from __future__ import annotations

import dataclasses
import os
from importlib import resources
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import PersonalCI
from .memory import ConsentRequired
from .types import KINDS, Sensitivity


class OnboardIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    display_name: str = ""
    providers: list[str] = ["mock"]


class MemoryIn(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    kind: str = "fact"
    key: str | None = None
    value: str | int | float | bool | None = None
    sensitivity: Sensitivity = Sensitivity.PERSONAL
    pinned: bool = False


class CorrectIn(BaseModel):
    content: str
    value: str | int | float | bool | None = None


class IngestIn(BaseModel):
    text: str = Field(min_length=1, max_length=500_000)
    source: str = Field(pattern=r"^ingest:[A-Za-z0-9_./-]+$")
    title: str | None = None
    sensitivity: Sensitivity = Sensitivity.PERSONAL


class ConsentIn(BaseModel):
    source: str
    purpose: str
    provider: str = "*"
    max_sensitivity: Sensitivity = Sensitivity.PERSONAL
    note: str = ""


class AskIn(BaseModel):
    task: str = Field(min_length=1, max_length=50_000)
    provider: str = "mock"
    model: str | None = None
    budget_tokens: int | None = Field(default=None, ge=0, le=200_000)
    assist_mode: str | None = None


class FeedbackIn(BaseModel):
    task_success: bool | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    thumbs: int | None = Field(default=None, ge=-1, le=1)
    corrected: bool | None = None
    reasked: bool | None = None
    edit_ratio: float | None = Field(default=None, ge=0, le=1)
    comment: str | None = Field(default=None, max_length=2000)


class GoalsIn(BaseModel):
    task_types: list[str]


class ProbeIn(BaseModel):
    skill: str = Field(min_length=1, max_length=200)
    correct: bool
    assisted: bool = False


def _item_out(i) -> dict:
    d = dataclasses.asdict(i)
    for k in ("valid_from", "valid_to", "recorded_at", "superseded_at"):
        d[k] = d[k].isoformat() if d[k] else None
    d["trust"], d["sensitivity"], d["current"] = i.trust.value, i.sensitivity.value, i.is_current()
    return d


def create_app(ci: PersonalCI) -> FastAPI:
    app = FastAPI(title="Personal CI", version="0.1.0",
                  description="Local API for Personal CI. Every call is consent-checked and audited.")

    def user(authorization: str = Header(default="")) -> str:
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        uid = ci.user_for_token(authorization.split(" ", 1)[1].strip())
        if uid is None:
            raise HTTPException(401, "unknown token")
        return uid

    def owned(fn, *a, **k):
        try:
            return fn(*a, **k)
        except KeyError:
            raise HTTPException(404, "not found")

    @app.post("/v1/onboard")
    def onboard(body: OnboardIn):
        if ci.store.list_user_ids() and os.environ.get("PCIM_ALLOW_ONBOARD") != "1":
            raise HTTPException(403, "onboarding is closed once a user exists (set PCIM_ALLOW_ONBOARD=1)")
        if ci.store.get_user(body.user_id):
            raise HTTPException(409, "user exists")
        return ci.onboard(body.user_id, body.display_name, tuple(body.providers))

    @app.get("/v1/me")
    def me(uid: str = Depends(user)):
        return ci.store.get_user(uid)

    @app.delete("/v1/me")
    def erase(uid: str = Depends(user)):
        return ci.erase(uid)

    @app.get("/v1/memory")
    def memory(uid: str = Depends(user), history: bool = False):
        items = ci.memory.all(uid) if history else ci.memory.current(uid)
        return [_item_out(i) for i in items]

    @app.post("/v1/memory")
    def remember(body: MemoryIn, uid: str = Depends(user)):
        if body.kind not in KINDS:
            raise HTTPException(422, f"kind must be one of {KINDS}")
        r = ci.remember(uid, body.content, body.kind, key=body.key, value=body.value,
                        sensitivity=body.sensitivity, pinned=body.pinned)
        return {"item": _item_out(r.item), "superseded": r.superseded, "conflicts": r.conflicts,
                "gated_from": r.gated_from, "reaffirmed": r.reaffirmed}

    @app.post("/v1/memory/{item_id}/correct")
    def correct(item_id: str, body: CorrectIn, uid: str = Depends(user)):
        r = owned(ci.memory.correct, uid, item_id, body.content, value=body.value)
        return {"item": _item_out(r.item), "superseded": r.superseded}

    @app.post("/v1/memory/{item_id}/confirm")
    def confirm(item_id: str, promote: bool = True, uid: str = Depends(user)):
        return _item_out(owned(ci.memory.confirm, uid, item_id, promote=promote))

    @app.post("/v1/memory/{item_id}/retract")
    def retract(item_id: str, uid: str = Depends(user)):
        owned(ci.memory.retract, uid, item_id)
        return {"ok": True}

    @app.post("/v1/memory/{item_id}/restore")
    def restore(item_id: str, uid: str = Depends(user)):
        owned(ci.memory.restore, uid, item_id)
        return {"ok": True}

    @app.post("/v1/memory/{item_id}/pin")
    def pin(item_id: str, pinned: bool = True, uid: str = Depends(user)):
        owned(ci.memory.pin, uid, item_id, pinned)
        return {"ok": True}

    @app.delete("/v1/memory/{item_id}")
    def delete(item_id: str, cascade: bool = True, uid: str = Depends(user)):
        return {"deleted": owned(ci.memory.delete, uid, item_id, cascade=cascade)}

    @app.get("/v1/conflicts")
    def conflicts(uid: str = Depends(user)):
        return ci.store.list_conflicts(uid)

    @app.post("/v1/ingest")
    def ingest(body: IngestIn, uid: str = Depends(user)):
        try:
            rs = ci.ingest_text(uid, body.text, body.source, body.title, body.sensitivity)
        except ConsentRequired as e:
            raise HTTPException(403, str(e))
        return {"items": [_item_out(r.item) for r in rs],
                "flagged": [r.item.id for r in rs if "contains_instructions" in r.item.flags]}

    @app.get("/v1/consent")
    def consent(uid: str = Depends(user)):
        return [dataclasses.asdict(g) | {"max_sensitivity": g.max_sensitivity.value,
                                         "active": g.revoked_at is None} for g in ci.ledger.grants(uid)]

    @app.post("/v1/consent")
    def grant(body: ConsentIn, uid: str = Depends(user)):
        try:
            g = ci.consent(uid, body.source, body.purpose, body.provider, body.max_sensitivity, body.note)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return dataclasses.asdict(g) | {"max_sensitivity": g.max_sensitivity.value}

    @app.delete("/v1/consent/{grant_id}")
    def revoke(grant_id: str, delete_stored: bool = False, uid: str = Depends(user)):
        return owned(ci.revoke, uid, grant_id, delete_stored)

    @app.post("/v1/ask")
    def ask(body: AskIn, uid: str = Depends(user)):
        try:
            r = ci.ask(uid, body.task, body.provider, body.model, body.budget_tokens, body.assist_mode)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(422, str(e))
        return {"interaction_id": r.interaction_id, "text": r.text, "receipt": r.receipt}

    @app.post("/v1/feedback/{interaction_id}")
    def feedback(interaction_id: str, body: FeedbackIn, uid: str = Depends(user)):
        signals = {k: v for k, v in body.model_dump().items() if v is not None}
        return {"reward": owned(ci.feedback, uid, interaction_id, **signals)}

    @app.get("/v1/interactions")
    def interactions(uid: str = Depends(user)):
        return ci.store.list_interactions(uid)

    @app.get("/v1/interactions/{interaction_id}")
    def receipt(interaction_id: str, uid: str = Depends(user)):
        return owned(ci.receipt, uid, interaction_id)

    @app.get("/v1/dna")
    def dna(uid: str = Depends(user)):
        return ci.dna(uid)

    @app.get("/v1/metrics")
    def metrics(uid: str = Depends(user)):
        return ci.metrics(uid)

    @app.get("/v1/audit")
    def audit(uid: str = Depends(user), limit: int = 500):
        ok, bad = ci.audit.verify()
        return {"chain_ok": ok, "first_bad_seq": bad, "head": ci.audit.head(),
                "entries": ci.audit.entries(uid)[-limit:]}

    @app.get("/v1/export")
    def export(uid: str = Depends(user)):
        return ci.export(uid)

    @app.post("/v1/growth/goals")
    def goals(body: GoalsIn, uid: str = Depends(user)):
        try:
            ci.set_growth_goals(uid, body.task_types)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"ok": True}

    @app.post("/v1/growth/probe")
    def probe(body: ProbeIn, uid: str = Depends(user)):
        return ci.probe(uid, body.skill, body.correct, body.assisted)

    dash = Path(str(resources.files("pcim") / "dashboard"))
    if dash.exists():
        app.mount("/static", StaticFiles(directory=dash), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return (dash / "index.html").read_text()

    return app
