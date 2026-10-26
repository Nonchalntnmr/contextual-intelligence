"""Consent ledger: who may store what, and which provider may see it.

A grant is (source pattern, purpose, provider, sensitivity ceiling, expiry). Purposes:

- ``store``  — CI may keep items from this source at all.
- ``egress`` — CI may send items from this source to this model provider.

There is no implicit consent. In particular, items CI *derives* (summaries, inferred
preferences) inherit the consent requirements of every source in their lineage — an
inference drawn from your health notes cannot reach a provider your health notes cannot.
That rule lives in egress.py, which is the only caller of `allows` for egress.

Hard rules no grant can override: SECRET items never egress, and an egress grant's
ceiling cannot be SECRET.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import fnmatch

from ..store import Store
from ..types import Sensitivity, new_id, utcnow
from .audit import AuditLog

PURPOSES = ("store", "egress")
PROVIDERS = ("anthropic", "openai", "google", "local", "mock", "*")


@dataclasses.dataclass(frozen=True)
class Grant:
    id: str
    user_id: str
    source: str
    purpose: str
    provider: str
    max_sensitivity: Sensitivity
    granted_at: dt.datetime
    expires_at: dt.datetime | None = None
    revoked_at: dt.datetime | None = None
    note: str = ""

    def active(self, now: dt.datetime) -> bool:
        return self.revoked_at is None and (self.expires_at is None or self.expires_at > now)

    def covers(self, source: str, purpose: str, provider: str) -> bool:
        return (
            self.purpose == purpose
            and fnmatch.fnmatchcase(source, self.source)
            and (self.provider == "*" or self.provider == provider or purpose == "store")
        )


def _grant_from_row(r: dict) -> Grant:
    return Grant(
        id=r["id"], user_id=r["user_id"], source=r["source"], purpose=r["purpose"], provider=r["provider"],
        max_sensitivity=Sensitivity(r["max_sensitivity"]),
        granted_at=dt.datetime.fromisoformat(r["granted_at"]),
        expires_at=dt.datetime.fromisoformat(r["expires_at"]) if r["expires_at"] else None,
        revoked_at=dt.datetime.fromisoformat(r["revoked_at"]) if r["revoked_at"] else None,
        note=r["note"] or "",
    )


class ConsentLedger:
    def __init__(self, store: Store, audit: AuditLog):
        self.store = store
        self.audit = audit

    def grant(
        self,
        user_id: str,
        source: str,
        purpose: str,
        provider: str = "*",
        max_sensitivity: Sensitivity = Sensitivity.PERSONAL,
        expires_at: dt.datetime | None = None,
        note: str = "",
    ) -> Grant:
        if purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {PURPOSES}")
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}")
        if purpose == "egress" and max_sensitivity is Sensitivity.SECRET:
            raise ValueError("SECRET items never leave the device; no grant can allow it")
        g = Grant(
            id=new_id("grt"), user_id=user_id, source=source, purpose=purpose, provider=provider,
            max_sensitivity=max_sensitivity, granted_at=utcnow(), expires_at=expires_at, note=note,
        )
        self.store.insert_grant(
            {
                **dataclasses.asdict(g),
                "max_sensitivity": g.max_sensitivity.value,
                "granted_at": g.granted_at.isoformat(),
                "expires_at": g.expires_at.isoformat() if g.expires_at else None,
            }
        )
        self.audit.append(
            user_id, "consent.grant", grant=g.id, source=source, purpose=purpose, provider=provider,
            max_sensitivity=max_sensitivity.value,
        )
        return g

    def revoke(self, user_id: str, grant_id: str) -> bool:
        ok = self.store.revoke_grant(grant_id, utcnow())
        if ok:
            self.audit.append(user_id, "consent.revoke", grant=grant_id)
        return ok

    def grants(self, user_id: str) -> list[Grant]:
        return [_grant_from_row(r) for r in self.store.list_grants(user_id)]

    def allows(
        self,
        user_id: str,
        source: str,
        purpose: str,
        provider: str,
        sensitivity: Sensitivity,
        now: dt.datetime | None = None,
        grants: list[Grant] | None = None,
    ) -> tuple[bool, str]:
        """`grants` lets a caller deciding many items at once read the ledger once."""
        now = now or utcnow()
        if purpose == "egress" and sensitivity is Sensitivity.SECRET:
            return False, "secret"
        grants = self.grants(user_id) if grants is None else grants
        matching = [g for g in grants if g.active(now) and g.covers(source, purpose, provider)]
        if not matching:
            return False, f"no-{purpose}-grant"
        if not any(sensitivity.rank <= g.max_sensitivity.rank for g in matching):
            return False, "above-sensitivity-ceiling"
        return True, "ok"
