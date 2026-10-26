"""Layer B — temporal context memory.

Assertions are bitemporal (see types.MemoryItem). The rules that make this more than a
vector store with timestamps:

- **Keyed assertions supersede, they do not accumulate.** Writing `diet = vegetarian`
  closes the valid time of `diet = omnivore` and links the two. The old row stays
  queryable `as_of` any earlier system time; it never reaches a model again.
- **Authority decides who supersedes whom.** A new assertion supersedes an old one only
  if its trust is at least as high and its valid time starts no earlier. Otherwise both
  are kept and a conflict is opened for the user; the newer, weaker claim is flagged.
- **Corrections are user-confirmed supersessions**, so they are reversible by
  inspection; **deletion** is not reversible by design (crypto-shredding), and it
  cascades through lineage to anything derived from the deleted item.
- **Goals expire** after 180 days unless restated; inferred items lose confidence with a
  half-life. Facts, constraints and user-stated preferences do not decay on their own:
  a constraint ("allergic to peanuts") that silently expired would be a safety bug.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np

from .embed import Embedder
from .sovereignty.audit import AuditLog
from .sovereignty.consent import ConsentLedger
from .sovereignty.egress import CONFIRMED_FLAG, INSTRUCTION_FLAG
from .sovereignty.injection import gate_kind, scan
from .store import Store
from .types import KINDS, MemoryItem, Sensitivity, Trust, approx_tokens, new_id, utcnow

DEFAULT_TTL_DAYS = {"goal": 180}
INFERRED_HALF_LIFE_DAYS = {"preference": 365, "goal": 120, "skill": 365, "fact": 730, "episode": 365, "constraint": 730}
CONFLICT_FLAG = "conflicted"


class ConsentRequired(PermissionError):
    pass


@dataclasses.dataclass
class WriteResult:
    item: MemoryItem
    superseded: list[str]
    conflicts: list[str]
    gated_from: str | None  # standing kind withheld pending user confirmation
    reaffirmed: bool = False


def effective_confidence(item: MemoryItem, now: dt.datetime | None = None) -> float:
    if item.trust is not Trust.INFERRED:
        return item.confidence
    hl = INFERRED_HALF_LIFE_DAYS.get(item.kind, 365)
    return item.confidence * 0.5 ** (item.age_days(now) / hl)


class TemporalMemory:
    def __init__(self, store: Store, embedder: Embedder, audit: AuditLog, ledger: ConsentLedger):
        self.store = store
        self.embedder = embedder
        self.audit = audit
        self.ledger = ledger
        self._index: dict[str, tuple[list[str], np.ndarray | None]] = {}

    # ------------------------------------------------------------------ writes
    def remember(
        self,
        user_id: str,
        content: str,
        kind: str = "fact",
        *,
        key: str | None = None,
        value=None,
        source: str = "user",
        trust: Trust = Trust.USER_STATED,
        sensitivity: Sensitivity = Sensitivity.PERSONAL,
        confidence: float = 1.0,
        valid_from: dt.datetime | None = None,
        valid_to: dt.datetime | None = None,
        parents: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        provenance: dict | None = None,
        pinned: bool = False,
    ) -> WriteResult:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        now = utcnow()
        if source != "user":
            ok, reason = self.ledger.allows(user_id, source, "store", "*", sensitivity, now)
            if not ok:
                raise ConsentRequired(f"cannot store from {source!r}: {reason}")

        flags = []
        hits = scan(content)
        if hits:
            flags.append(INSTRUCTION_FLAG)
        stored_kind, gated_from = gate_kind(kind, trust)
        prov = dict(provenance or {})
        if hits:
            prov["instruction_patterns"] = hits
        if gated_from:
            prov["proposed_kind"] = gated_from

        valid_from = valid_from or now
        if valid_to is None and stored_kind in DEFAULT_TTL_DAYS and trust is not Trust.UNTRUSTED:
            valid_to = valid_from + dt.timedelta(days=DEFAULT_TTL_DAYS[stored_kind])

        item = MemoryItem(
            id=new_id("mem"), user_id=user_id, kind=stored_kind, content=content, source=source,
            trust=trust, sensitivity=sensitivity, confidence=confidence, valid_from=valid_from,
            recorded_at=now, key=key, value=value, valid_to=valid_to, parents=tuple(parents),
            tags=tuple(tags), flags=tuple(flags), provenance=prov, tokens=approx_tokens(content),
            pinned=pinned,
        )

        superseded, conflicts = [], []
        if key is not None:
            for old in self._current_with_key(user_id, key, now):
                if old.value == value and old.content.strip().lower() == content.strip().lower():
                    # The same claim again: reaffirm rather than duplicate.
                    self.store.update_item_meta(old.id, confidence=max(old.confidence, confidence))
                    self.audit.append(user_id, "memory.reaffirm", item=old.id)
                    old.confidence = max(old.confidence, confidence)
                    return WriteResult(old, [], [], gated_from, reaffirmed=True)
                if trust.rank >= old.trust.rank and valid_from >= old.valid_from:
                    # Close system time only. The old row's valid time is left as it was
                    # asserted, so as_of() can reconstruct exactly what was believed
                    # before this write; valid_at() derives the effective end from the
                    # superseding row.
                    self.store.update_item_meta(old.id, superseded_at=now, superseded_by=item.id)
                    superseded.append(old.id)
                else:
                    conflicts.append(self.store.add_conflict(user_id, old.id, item.id))
            if conflicts:
                item.flags = item.flags + (CONFLICT_FLAG,)

        emb = self.embedder.embed([content])[0]
        self.store.insert_item(item, emb)
        self._index.pop(user_id, None)
        self.audit.append(
            user_id, "memory.write", item=item.id, kind=item.kind, source=source, trust=trust.value,
            sensitivity=sensitivity.value, superseded=superseded, conflicts=conflicts, flags=list(item.flags),
            gated_from=gated_from,
        )
        return WriteResult(item, superseded, conflicts, gated_from)

    def correct(self, user_id: str, item_id: str, new_content: str, value=None) -> WriteResult:
        """User says an item was *wrong* (not that it changed) and gives the right version.

        A correction takes over the old row's valid time — the corrected claim applies to
        the same period — and valid_at() treats the old row as never having been true.
        Contrast with remember(key=...), which records that the world changed.
        """
        old = self._owned(user_id, item_id)
        if not old.is_current():
            raise ValueError("only a current item can be corrected; correct its successor instead")
        res = self.remember(
            user_id, new_content, old.kind, key=old.key, value=value, source="user",
            trust=Trust.USER_CONFIRMED, sensitivity=old.sensitivity, confidence=1.0,
            valid_from=old.valid_from, valid_to=old.valid_to, tags=old.tags,
            provenance={"corrects": old.id},
        )
        if old.id not in res.superseded:  # un-keyed item: close it explicitly
            self.store.update_item_meta(old.id, superseded_at=utcnow(), superseded_by=res.item.id)
            res.superseded.append(old.id)
        self.audit.append(user_id, "memory.correct", item=old.id, by=res.item.id)
        return res

    def confirm(self, user_id: str, item_id: str, promote: bool = True) -> MemoryItem:
        """User reviewed an item: raise trust, clear the instruction hold, optionally promote kind."""
        it = self._owned(user_id, item_id)
        fields = {"trust": Trust.USER_CONFIRMED, "flags": tuple(set(it.flags) | {CONFIRMED_FLAG})}
        proposed = it.provenance.get("proposed_kind")
        if promote and proposed:
            fields["kind"] = proposed
        self.store.update_item_meta(it.id, **fields)
        self.audit.append(user_id, "memory.confirm", item=it.id, promoted_to=fields.get("kind"))
        return self.store.get_item(it.id)

    def retract(self, user_id: str, item_id: str) -> None:
        """Reversible: the item stops being used but stays inspectable."""
        self._owned(user_id, item_id)
        self.store.update_item_meta(item_id, status="retracted")
        self.audit.append(user_id, "memory.retract", item=item_id)

    def restore(self, user_id: str, item_id: str) -> None:
        self._owned(user_id, item_id)
        self.store.update_item_meta(item_id, status="active")
        self.audit.append(user_id, "memory.restore", item=item_id)

    def pin(self, user_id: str, item_id: str, pinned: bool = True) -> None:
        self._owned(user_id, item_id)
        self.store.update_item_meta(item_id, pinned=pinned)
        self.audit.append(user_id, "memory.pin", item=item_id, pinned=pinned)

    def delete(self, user_id: str, item_id: str, cascade: bool = True) -> list[str]:
        """Irreversible. Destroys the item's key, and with cascade, everything derived from it."""
        self._owned(user_id, item_id)
        doomed, frontier = [], [item_id]
        while frontier:
            nxt = []
            for iid in frontier:
                if iid in doomed:
                    continue
                doomed.append(iid)
                if cascade:
                    nxt.extend(self.store.children_of(iid))
            frontier = nxt
        for iid in doomed:
            self.store.delete_item(iid)
        self._index.pop(user_id, None)
        self.audit.append(user_id, "memory.delete", items=doomed, cascade=cascade)
        return doomed

    # ------------------------------------------------------------------ reads
    def _owned(self, user_id: str, item_id: str) -> MemoryItem:
        it = self.store.get_item(item_id)
        if it is None or it.user_id != user_id:
            raise KeyError(item_id)
        return it

    def _current_with_key(self, user_id: str, key: str, now: dt.datetime) -> list[MemoryItem]:
        return [i for i in self.store.list_items(user_id) if i.key == key and i.is_current(now)]

    def all(self, user_id: str) -> list[MemoryItem]:
        return self.store.list_items(user_id)

    def current(self, user_id: str, now: dt.datetime | None = None) -> list[MemoryItem]:
        now = now or utcnow()
        return [i for i in self.store.list_items(user_id) if i.is_current(now)]

    def as_of(self, user_id: str, system_time: dt.datetime) -> list[MemoryItem]:
        """What CI believed, at `system_time`, to be true at `system_time`.

        Exact for writes, supersessions and corrections. One approximation: retraction
        carries no timestamp, so a retracted item is treated as never believed.
        """
        out = []
        for i in self.store.list_items(user_id):
            if i.status != "active" or i.recorded_at > system_time:
                continue
            if i.superseded_at is not None and i.superseded_at <= system_time:
                continue
            if i.valid_from > system_time or (i.valid_to is not None and i.valid_to <= system_time):
                continue
            out.append(i)
        return out

    def valid_at(self, user_id: str, valid_time: dt.datetime) -> list[MemoryItem]:
        """What CI *now* believes was true at `valid_time` ("what was my diet last June?").

        A superseded row stays valid up to its successor's valid_from — unless the
        successor was a correction, in which case the old row was never true.
        """
        items = {i.id: i for i in self.store.list_items(user_id) if i.status == "active"}
        out = []
        for i in items.values():
            end = i.valid_to
            if i.superseded_by:
                succ = items.get(i.superseded_by)
                if succ is not None:
                    if succ.provenance.get("corrects") == i.id:
                        continue
                    end = succ.valid_from if end is None else min(end, succ.valid_from)
            if i.valid_from <= valid_time and (end is None or end > valid_time):
                out.append(i)
        return out

    def embeddings(self, user_id: str) -> tuple[list[str], np.ndarray | None]:
        if user_id not in self._index:
            self._index[user_id] = self.store.item_embeddings(user_id)
        return self._index[user_id]

    def search(self, user_id: str, query: str, k: int = 10, now: dt.datetime | None = None) -> list[tuple[MemoryItem, float]]:
        ids, mat = self.embeddings(user_id)
        if mat is None:
            return []
        current = {i.id: i for i in self.current(user_id, now)}
        q = self.embedder.embed([query])[0]
        sims = mat @ q
        order = np.argsort(-sims)
        out = []
        for j in order:
            it = current.get(ids[j])
            if it is not None:
                out.append((it, float(sims[j])))
                if len(out) >= k:
                    break
        return out

    def export(self, user_id: str) -> dict:
        """Everything CI holds about the user, including superseded history. Portable JSON."""
        items = []
        for i in self.store.list_items(user_id):
            d = dataclasses.asdict(i)
            for k in ("valid_from", "valid_to", "recorded_at", "superseded_at"):
                d[k] = d[k].isoformat() if d[k] else None
            d["trust"], d["sensitivity"] = i.trust.value, i.sensitivity.value
            items.append(d)
        return {"format": "pcim-memory/1", "user_id": user_id, "exported_at": utcnow().isoformat(), "items": items}
