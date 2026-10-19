"""Projects, autonomy settings and authorizations: the durable state the workspace runs on.

A project holds what a person is trying to do and what has been learned so far:
objective, success criteria, and notes (open questions, decisions, next steps), each
with a provenance (`user`, `model`, `system`) and a timestamp. Model-written notes are
always labelled as such and rendered as data, never as instructions.

Everything a project says is sealed under the project's own key. Deleting a project
destroys that key, and with it every transcript, task argument and task result.

Autonomy:
- **Mode** (copilot | collaborative | autonomous) lives in the user's settings.
- A **workspace authorization** attaches one directory to a project, the test argv the
  person allows, and whether writes are allowed.
- An **autonomous authorization** names tools, a max action count, a max duration, and an
  expiry. It is checked on every call and is never widened by anything a task does.
- **Halt** blocks all execution until the person resumes.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import subprocess
from pathlib import Path

from .engine import PersonalCI
from .types import new_id, utcnow

MODES = ("copilot", "collaborative", "autonomous")
NOTE_KINDS = ("open_question", "decision", "next_step", "summary")
EXECUTABLE_TOOLS = ("read_file", "list_files", "search_code", "git_status", "record_note", "run_tests", "write_file")


class WorkspaceError(ValueError):
    pass


@dataclasses.dataclass
class Project:
    id: str
    name: str
    objective: str
    success_criteria: str
    notes: list[dict]
    status: str
    created_at: str
    updated_at: str

    def open_notes(self, kind: str) -> list[dict]:
        return [n for n in self.notes if n["kind"] == kind and not n.get("resolved_at")]


def _home_is_git_repo() -> bool:
    try:
        r = subprocess.run(["git", "-C", str(Path.home()), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and Path(r.stdout.strip()).resolve() == Path.home().resolve()
    except Exception:
        return False


def validate_root(root: str | Path) -> Path:
    """A workspace root must be an existing directory that is not /, $HOME, or an ancestor of $HOME."""
    p = Path(root).expanduser().resolve()
    home = Path.home().resolve()
    if not p.is_dir():
        raise WorkspaceError(f"{p} is not a directory")
    if p == Path("/") or p == home or p in home.parents:
        raise WorkspaceError(f"{p} is too broad to authorize; choose a project directory")
    return p


class Workspace:
    def __init__(self, ci: PersonalCI):
        self.ci = ci
        self.store = ci.store
        self.audit = ci.audit

    # ------------------------------------------------------------------ projects
    def _payload(self, row) -> dict:
        return self.store.open_json(row["key_id"], row["payload_ct"], row["id"])

    def _row(self, user_id: str, project_id: str):
        rows = self.store.query("SELECT * FROM projects WHERE id=? AND user_id=?", (project_id, user_id))
        if not rows:
            raise KeyError(project_id)
        return rows[0]

    def _to_project(self, row) -> Project:
        p = self._payload(row)
        return Project(id=row["id"], name=p["name"], objective=p.get("objective", ""),
                       success_criteria=p.get("success_criteria", ""), notes=p.get("notes", []),
                       status=row["status"], created_at=row["created_at"], updated_at=row["updated_at"])

    def _save(self, row, payload: dict) -> None:
        now = utcnow().isoformat()
        self.store.execute("UPDATE projects SET payload_ct=?, updated_at=? WHERE id=?",
                           (self.store.seal_json(row["key_id"], payload, row["id"]), now, row["id"]))

    def create_project(self, user_id: str, name: str, objective: str = "", success_criteria: str = "") -> Project:
        if not name.strip():
            raise WorkspaceError("a project needs a name")
        pid, key_id, now = new_id("prj"), self.store.new_project_key(user_id), utcnow().isoformat()
        payload = {"name": name.strip(), "objective": objective, "success_criteria": success_criteria, "notes": []}
        self.store.execute("INSERT INTO projects VALUES (?,?,?,?,?,?,?)",
                           (pid, user_id, key_id, now, now, "active", self.store.seal_json(key_id, payload, pid)))
        self.audit.append(user_id, "project.create", project=pid)
        return self.get_project(user_id, pid)

    def get_project(self, user_id: str, project_id: str) -> Project:
        return self._to_project(self._row(user_id, project_id))

    def list_projects(self, user_id: str) -> list[Project]:
        rows = self.store.query("SELECT * FROM projects WHERE user_id=? ORDER BY updated_at DESC", (user_id,))
        return [self._to_project(r) for r in rows]

    def update_project(self, user_id: str, project_id: str, **fields) -> Project:
        row = self._row(user_id, project_id)
        p = self._payload(row)
        for k in ("name", "objective", "success_criteria"):
            if k in fields and fields[k] is not None:
                p[k] = fields[k]
        self._save(row, p)
        self.audit.append(user_id, "project.update", project=project_id, fields=sorted(fields))
        return self.get_project(user_id, project_id)

    def add_note(self, user_id: str, project_id: str, kind: str, text: str, source: str = "user") -> dict:
        if kind not in NOTE_KINDS:
            raise WorkspaceError(f"note kind must be one of {NOTE_KINDS}")
        if source not in ("user", "model", "system"):
            raise WorkspaceError("bad note source")
        text = text.strip()[:2000]
        if not text:
            raise WorkspaceError("empty note")
        row = self._row(user_id, project_id)
        p = self._payload(row)
        note = {"id": new_id("note"), "kind": kind, "text": text, "source": source,
                "created_at": utcnow().isoformat(), "resolved_at": None}
        p.setdefault("notes", []).append(note)
        self._save(row, p)
        self.audit.append(user_id, "project.note", project=project_id, note=note["id"], kind=kind, source=source)
        return note

    def resolve_note(self, user_id: str, project_id: str, note_id: str) -> None:
        row = self._row(user_id, project_id)
        p = self._payload(row)
        for n in p.get("notes", []):
            if n["id"] == note_id:
                n["resolved_at"] = utcnow().isoformat()
                self._save(row, p)
                self.audit.append(user_id, "project.note_resolved", project=project_id, note=note_id)
                return
        raise KeyError(note_id)

    def delete_project(self, user_id: str, project_id: str) -> dict:
        row = self._row(user_id, project_id)
        counts = self.store.delete_project_rows(project_id, row["key_id"])
        self.audit.append(user_id, "project.delete", project=project_id, **{k: int(v) for k, v in counts.items()})
        return counts

    def project_key(self, user_id: str, project_id: str) -> str:
        return self._row(user_id, project_id)["key_id"]

    # ------------------------------------------------------------------ autonomy settings
    def _settings(self, user_id: str) -> dict:
        u = self.store.get_user(user_id)
        if u is None:
            raise KeyError(user_id)
        return u["settings"]

    def get_mode(self, user_id: str) -> str:
        return self._settings(user_id).get("autonomy_mode", "collaborative")

    def set_mode(self, user_id: str, mode: str) -> None:
        if mode not in MODES:
            raise WorkspaceError(f"mode must be one of {MODES}")
        s = self._settings(user_id)
        s["autonomy_mode"] = mode
        self.store.update_settings(user_id, s)
        self.audit.append(user_id, "autonomy.mode", mode=mode)

    def is_halted(self, user_id: str) -> bool:
        return bool(self._settings(user_id).get("halted"))

    def set_halted(self, user_id: str, halted: bool) -> None:
        s = self._settings(user_id)
        s["halted"] = halted
        self.store.update_settings(user_id, s)
        self.audit.append(user_id, "autonomy.halt" if halted else "autonomy.resume")

    # ------------------------------------------------------------------ authorizations
    def _insert_auth(self, user_id: str, project_id: str | None, kind: str, spec: dict,
                     expires_at: dt.datetime | None) -> str:
        aid = new_id("auth")
        self.store.execute(
            "INSERT INTO authorizations(id, user_id, project_id, kind, spec, created_at, expires_at) VALUES (?,?,?,?,?,?,?)",
            (aid, user_id, project_id, kind, json.dumps(spec), utcnow().isoformat(),
             expires_at.isoformat() if expires_at else None),
        )
        return aid

    def authorize_workspace(self, user_id: str, project_id: str, root: str, run_argv: list[str] | None = None,
                            allow_write: bool = False, run_timeout_s: int = 300) -> str:
        self._row(user_id, project_id)
        p = validate_root(root)
        if run_argv is not None:
            if not run_argv or not all(isinstance(a, str) and a for a in run_argv):
                raise WorkspaceError("run_argv must be a non-empty list of strings")
            if run_argv[0] in ("sh", "bash", "zsh", "fish", "sudo", "su") or any(c in run_argv[0] for c in ";|&`$"):
                raise WorkspaceError("run_argv must name a test runner directly, not a shell")
        for old in self.active_auths(user_id, project_id, "workspace"):
            self.revoke(user_id, old["id"], audit=False)
        spec = {"root": str(p), "run_argv": run_argv, "allow_write": bool(allow_write),
                "run_timeout_s": int(run_timeout_s), "home_is_git_repo": _home_is_git_repo()}
        aid = self._insert_auth(user_id, project_id, "workspace", spec, None)
        self.audit.append(user_id, "workspace.authorize", project=project_id, auth=aid, write=bool(allow_write),
                          run=run_argv is not None)
        return aid

    def grant_autonomy(self, user_id: str, project_id: str, tools: list[str], max_actions: int = 10,
                       max_seconds: int = 900, ttl_minutes: int = 60) -> str:
        self._row(user_id, project_id)
        bad = set(tools) - set(EXECUTABLE_TOOLS)
        if bad or not tools:
            raise WorkspaceError(f"unknown tools {sorted(bad)}" if bad else "name at least one tool")
        if not self.active_auths(user_id, project_id, "workspace"):
            raise WorkspaceError("authorize a workspace for this project first")
        if not (1 <= max_actions <= 200 and 1 <= max_seconds <= 86400 and 1 <= ttl_minutes <= 7 * 24 * 60):
            raise WorkspaceError("limits out of range")
        spec = {"tools": sorted(set(tools)), "max_actions": int(max_actions), "max_seconds": int(max_seconds)}
        aid = self._insert_auth(user_id, project_id, "autonomous", spec,
                                utcnow() + dt.timedelta(minutes=ttl_minutes))
        self.audit.append(user_id, "autonomy.grant", project=project_id, auth=aid, tools=spec["tools"],
                          max_actions=max_actions, max_seconds=max_seconds, ttl_minutes=ttl_minutes)
        return aid

    def revoke(self, user_id: str, auth_id: str, audit: bool = True) -> bool:
        cur = self.store.execute("UPDATE authorizations SET revoked_at=? WHERE id=? AND user_id=? AND revoked_at IS NULL",
                                 (utcnow().isoformat(), auth_id, user_id))
        if cur.rowcount and audit:
            self.audit.append(user_id, "authorization.revoke", auth=auth_id)
        return cur.rowcount > 0

    def active_auths(self, user_id: str, project_id: str | None, kind: str) -> list[dict]:
        now = utcnow()
        rows = self.store.query(
            "SELECT * FROM authorizations WHERE user_id=? AND kind=? AND revoked_at IS NULL"
            + (" AND project_id=?" if project_id else ""),
            (user_id, kind, project_id) if project_id else (user_id, kind))
        out = []
        for r in rows:
            if r["expires_at"] and dt.datetime.fromisoformat(r["expires_at"]) <= now:
                continue
            d = dict(r)
            d["spec"] = json.loads(d["spec"])
            out.append(d)
        return out

    def all_auths(self, user_id: str) -> list[dict]:
        rows = self.store.query("SELECT * FROM authorizations WHERE user_id=? ORDER BY created_at DESC", (user_id,))
        return [dict(r) | {"spec": json.loads(r["spec"])} for r in rows]

    def workspace_auth(self, user_id: str, project_id: str) -> dict | None:
        a = self.active_auths(user_id, project_id, "workspace")
        return a[0] if a else None

    def consume(self, auth_id: str) -> None:
        self.store.execute(
            "UPDATE authorizations SET used_actions = used_actions + 1, first_used_at = COALESCE(first_used_at, ?) "
            "WHERE id=?", (utcnow().isoformat(), auth_id))

    # ------------------------------------------------------------------ briefing
    def briefing(self, user_id: str, project_id: str) -> dict:
        """Everything a new session needs to resume, compiled from stored state only."""
        p = self.get_project(user_id, project_id)
        key = self.project_key(user_id, project_id)
        last = self.store.query(
            "SELECT * FROM sessions WHERE project_id=? AND status='closed' ORDER BY updated_at DESC LIMIT 1",
            (project_id,))
        reflection = None
        if last:
            st = self.store.open_json(key, last[0]["state_ct"], last[0]["id"])
            reflection = st.get("reflection")
        pending = self.store.query(
            "SELECT id, tool, created_at FROM tasks WHERE project_id=? AND status='awaiting_approval' ORDER BY created_at",
            (project_id,))
        ws = self.workspace_auth(user_id, project_id)
        return {
            "project": {"id": p.id, "name": p.name, "objective": p.objective, "success_criteria": p.success_criteria},
            "open_questions": p.open_notes("open_question"),
            "decisions": [n for n in p.notes if n["kind"] == "decision"][-10:],
            "next_steps": p.open_notes("next_step"),
            "last_session": reflection,
            "pending_approvals": [dict(r) for r in pending],
            "workspace": ({"root": ws["spec"]["root"], "can_run_tests": ws["spec"]["run_argv"] is not None,
                           "can_write": ws["spec"]["allow_write"]} if ws else None),
            "mode": self.get_mode(user_id),
        }


def render_briefing(b: dict) -> str:
    """Briefing as plain text for the model. Model-written notes are marked."""
    def line(n):
        tag = "" if n["source"] == "user" else f" [{n['source']}-written]"
        return f"- {n['text']}{tag} ({n['created_at'][:10]})"

    p = b["project"]
    parts = [f"Project: {p['name']}"]
    if p["objective"]:
        parts.append(f"Objective: {p['objective']}")
    if p["success_criteria"]:
        parts.append(f"Success criteria: {p['success_criteria']}")
    for title, key in (("Open questions", "open_questions"), ("Decisions so far", "decisions"),
                       ("Next steps", "next_steps")):
        if b[key]:
            parts.append(f"{title}:\n" + "\n".join(line(n) for n in b[key]))
    ls = b.get("last_session")
    if ls:
        parts.append("Last session (computed from task records):\n" + "\n".join(f"- {x}" for x in ls.get("lines", [])))
    if b.get("workspace"):
        w = b["workspace"]
        parts.append(f"Authorized workspace: {w['root']} (tests: {'allowed with approval' if w['can_run_tests'] else 'not configured'}; "
                     f"writes: {'allowed with approval' if w['can_write'] else 'not allowed'})")
    else:
        parts.append("No workspace directory is authorized for this project; file tools will be refused.")
    parts.append(f"Autonomy mode: {b['mode']}")
    return "\n\n".join(parts)
