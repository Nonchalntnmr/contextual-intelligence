"""Hash-chained, append-only audit log.

Every entry commits to the previous entry's hash, so editing or deleting any past entry
breaks verification from that point on. This gives tamper *evidence*, not tamper
*resistance*: someone with write access can rewrite the entire chain. Exporting the head
hash somewhere the attacker cannot write (an email to yourself, a git commit, a
transparency log) is what turns evidence into proof. `head()` exists for that.

Payloads must never contain personal content — ids, counts, provider names, reasons and
hashes only. `append` enforces a small allowlist of value types to make that hard to get
wrong by accident.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..store import Store
from ..types import utcnow

_ALLOWED_SCALARS = (str, int, float, bool, type(None))
_MAX_STR = 200  # long strings are almost always content leaking into the audit log


def _check(value: Any, path: str = "payload") -> None:
    if isinstance(value, str) and len(value) > _MAX_STR:
        raise ValueError(f"{path}: string of {len(value)} chars in an audit payload; log a hash instead")
    if isinstance(value, dict):
        for k, v in value.items():
            _check(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check(v, f"{path}[{i}]")
    elif not isinstance(value, _ALLOWED_SCALARS):
        raise TypeError(f"{path}: {type(value).__name__} not allowed in an audit payload")


def _digest(prev_hash: str, user_id: str, ts: str, event: str, payload: str) -> str:
    h = hashlib.sha256()
    for part in (prev_hash, user_id, ts, event, payload):
        h.update(part.encode())
        h.update(b"\x1f")
    return h.hexdigest()


class AuditLog:
    def __init__(self, store: Store):
        self.store = store

    def append(self, user_id: str, event: str, **payload: Any) -> str:
        _check(payload)
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self.store._lock:  # read-then-write of the chain head must be atomic
            prev = self.store.last_audit_hash()
            ts = utcnow().isoformat()
            h = _digest(prev, user_id, ts, event, body)
            self.store.insert_audit(user_id, ts, event, body, prev, h)
        return h

    def entries(self, user_id: str | None = None, limit: int | None = None) -> list[dict]:
        out = []
        for row in self.store.list_audit(user_id, limit):
            row = dict(row)
            row["payload"] = json.loads(row["payload"])
            out.append(row)
        return out

    def head(self) -> str:
        return self.store.last_audit_hash()

    def verify(self) -> tuple[bool, int | None]:
        """Recompute the whole chain. Returns (ok, seq of the first bad entry)."""
        prev = "0" * 64
        for row in self.store.list_audit():
            if row["prev_hash"] != prev:
                return False, row["seq"]
            if _digest(prev, row["user_id"], row["ts"], row["event"], row["payload"]) != row["hash"]:
                return False, row["seq"]
            prev = row["hash"]
        return True, None
