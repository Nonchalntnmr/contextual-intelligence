"""Milestone 1 against real models: the sixteen-step demonstration, live.

    .venv/bin/python scripts/milestone1_live.py --home /tmp/ci-live --out /tmp/ci-live/report

Runs the real `pcim serve` process on 127.0.0.1, kills it after step 5 and starts a new
one on the same store, and drives everything over HTTP. Models are reached through the
local Claude Code login (`claude-cli` provider), because this machine has no API keys:
Provider A is Claude Opus 5.5 and Provider B is Claude Haiku 5.5. Both are Anthropic, so
this run shows cross-model continuity and scoped context, not cross-vendor exchange; the
cross-vendor path is covered by tests/test_milestone1_demo.py on recorded wire formats.

The project content is synthetic (a toy Sharpe-ratio module). The script acts as the
person: it approves only the write to src/sharpe.py and the test run, and denies anything
else. It records what happened, including anything that did not go to plan, in
report.json and report.md. It never edits a model's output.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CANARY = "ZEBRA-7741"
PRIVATE = "Fund terms: 2 and 20 with a 6% hurdle."
SHARPE_365 = ("import math, statistics\n\n\ndef sharpe(daily):\n"
              "    return statistics.mean(daily) / statistics.stdev(daily) * math.sqrt(365)\n")
CHECK = ("import math, statistics, sys\nsys.path.insert(0, 'src')\nfrom sharpe import sharpe\n"
         "r = [0.01, -0.005, 0.002, 0.004]\n"
         "assert abs(sharpe(r) - statistics.mean(r) / statistics.stdev(r) * math.sqrt(252)) < 1e-12, "
         "'sharpe must annualise with sqrt(252)'\nprint('1 passed')\n")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, home: Path, log: Path):
        self.home, self.log, self.port, self.proc = home, log, free_port(), None

    def start(self) -> str:
        env = dict(os.environ, PCIM_HOME=str(self.home))
        self.proc = subprocess.Popen([str(ROOT / ".venv/bin/pcim"), "serve", "--port", str(self.port)], env=env,
                                     stdout=open(self.log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
        url = f"http://127.0.0.1:{self.port}"
        for _ in range(100):
            try:
                if httpx.get(url + "/v1/me", timeout=1).status_code == 401:
                    return url
            except httpx.HTTPError:
                time.sleep(0.1)
        raise SystemExit("server did not start; see " + str(self.log))

    def stop(self) -> None:
        os.killpg(self.proc.pid, signal.SIGTERM)
        self.proc.wait(timeout=20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="empty directory for the store (synthetic data only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-a", default="opus")
    ap.add_argument("--model-b", default="haiku")
    a = ap.parse_args()
    home, out = Path(a.home), Path(a.out)
    if home.exists() and any(home.iterdir()):
        raise SystemExit(f"{home} is not empty; use a fresh directory")
    home.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    repo = home / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "sharpe.py").write_text(SHARPE_365)
    (repo / "check.py").write_text(CHECK)

    steps: list[dict] = []
    t_start = time.time()

    def record(n, title, ok, **detail):
        steps.append({"step": n, "title": title, "ok": bool(ok), **detail})
        print(f"[{n:>2}] {'ok ' if ok else 'NO '} {title}" + (f" — {detail.get('note')}" if detail.get("note") else ""),
              flush=True)

    srv = Server(home, out / "server.log")
    url = srv.start()
    c = httpx.Client(base_url=url, timeout=900)

    # 1. sign in
    tok = c.post("/v1/onboard", json={"user_id": "demo", "providers": ["claude-cli"]}).json()["api_token"]
    c.headers["Authorization"] = f"Bearer {tok}"
    record(1, "sign in", c.get("/v1/me").status_code == 200, note="bearer token issued at onboarding")

    # 2-3. project, goal, constraints, workspace, private memory
    p = c.post("/v1/projects", json={"name": "Sharpe audit", "objective": "Make the backtest's Sharpe ratio correct",
                                     "constraints": ["daily returns", "no new dependencies"],
                                     "success_criteria": "check.py passes"}).json()
    pid = p["id"]
    record(2, "create a project", bool(pid), project=pid, providers=c.get(f"/v1/projects/{pid}/providers").json())
    c.post(f"/v1/projects/{pid}/workspace", json={"root": str(repo), "run_argv": [sys.executable, "check.py"],
                                                  "allow_write": True})
    c.post("/v1/consent", json={"source": "ingest:fund", "purpose": "store"})  # stored, never sent anywhere
    fund_id = c.post("/v1/ingest", json={"text": PRIVATE, "source": "ingest:fund"}).json()["items"][0]["id"]
    record(3, "goal and constraints recorded", True, constraints=p["constraints"], private_item=fund_id)

    # 4-5. a real request to Provider A
    t0 = time.time()
    turn = c.post(f"/v1/projects/{pid}/chat", json={"provider": "claude-cli", "model": a.model_a, "message": (
        "Read src/sharpe.py and tell me which annualisation factor sharpe() uses. Put that finding on the claim "
        "ledger with record_claim, citing the read as evidence, and record a decision note that we report "
        f"annualised Sharpe from daily returns. (My canary phrase is {CANARY}; it is not project information.)")})
    turn_a = turn.json()
    tools_a = [(t["tool"], t["status"]) for t in turn_a.get("tasks", [])]
    record(4, f"real request to Provider A (claude-cli:{a.model_a})", turn.status_code == 200
           and turn_a.get("status") in ("done", "awaiting_approval") and ("read_file", "succeeded") in tools_a,
           status=turn_a.get("status"), seconds=round(time.time() - t0, 1), reply=turn_a.get("text"), tools=tools_a,
           cost_usd=turn_a.get("cost_usd"))
    claims = c.get(f"/v1/projects/{pid}/claims").json()
    briefing = c.get(f"/v1/projects/{pid}").json()["briefing"]
    record(5, "authorized context saved with provenance", bool(claims) and bool(briefing["decisions"]),
           claims=[{k: x[k] for k in ("id", "statement", "status", "author", "evidence_refs")} for x in claims],
           decisions=briefing["decisions"])

    # 6. close the application. Anything Provider A left awaiting approval, the person declines:
    # changes happen next session. (A closed session is not resumed by the denial.)
    c.post(f"/v1/projects/{pid}/session/close")
    for tid in turn_a.get("pending", []):
        c.post(f"/v1/tasks/{tid}/deny", json={"reason": "not in this session; we will make changes next time"})
    srv.stop()
    record(6, "application closed", srv.proc.poll() is not None, note=f"server pid {srv.proc.pid} exited")

    # 7. restart and recover
    url = srv.start()
    c = httpx.Client(base_url=url, timeout=900, headers={"Authorization": f"Bearer {tok}"})
    got = c.get(f"/v1/projects/{pid}").json()
    record(7, "project state recovered after restart", got["briefing"]["decisions"] == briefing["decisions"]
           and len(got["briefing"]["claims"]) == len(claims), last_session=got["briefing"]["last_session"])

    # 8-9. resume with Provider B
    t0 = time.time()
    tb = c.post(f"/v1/projects/{pid}/chat", json={"provider": "claude-cli", "model": a.model_b, "message":
                                                  "Pick up where we left off: what have we established so far?"}).json()
    record(8, f"resumed with Provider B (claude-cli:{a.model_b}) without re-explaining",
           tb.get("status") in ("done", "awaiting_approval") and bool(tb.get("text")), status=tb.get("status"),
           seconds=round(time.time() - t0, 1), reply=tb.get("text"), cost_usd=tb.get("cost_usd"),
           tools=[(t["tool"], t["status"]) for t in tb.get("tasks", [])])
    disc = c.get(f"/v1/projects/{pid}/disclosures").json()
    to_b = [d for d in disc if d["model"] == a.model_b]
    btext = to_b[0]["briefing_text"] if to_b else ""
    withheld_b = {i: why for i, why in (to_b[0]["memory_withheld"] if to_b else [])}
    record(9, "Provider B received only authorized context", bool(to_b) and CANARY not in btext and PRIVATE not in btext
           and fund_id in withheld_b, canary_in_briefing=CANARY in btext, private_item_withheld=withheld_b.get(fund_id),
           briefing_sha256=to_b[0]["briefing_sha256"] if to_b else None, briefing_text=btext)

    # 10-11. narrow verification, routed deterministically
    if claims:
        cid = claims[0]["id"]
        route = c.get(f"/v1/projects/{pid}/route", params={"role": "verify", "claim_id": cid}).json()
        t0 = time.time()
        v = c.post(f"/v1/projects/{pid}/claims/{cid}/verify", json={})
        vj = v.json()
        record(10, "verification delegated through the exchange gateway", v.status_code == 200, route=route,
               seconds=round(time.time() - t0, 1), http=v.status_code,
               assessment=vj.get("assessment") if v.status_code == 200 else vj)
        msgs = c.get(f"/v1/projects/{pid}/exchange").json()
        record(11, "evidence and any disagreement displayed", v.status_code == 200, claim_status=vj.get("claim", {}).get("status"),
               messages=[m["message"] for m in msgs], withheld=[m.get("withheld") for m in msgs])
    else:
        record(10, "verification delegated", False, note="Provider A recorded no claim, so there was nothing to verify")
        record(11, "evidence and disagreement displayed", False, note="no claim")

    # 12-14. a reversible action, approved, executed, verified by a test
    t0 = time.time()
    if tb.get("status") == "awaiting_approval":  # B already proposed it while resuming
        tw, when = tb, "proposed by Provider B during step 8, unprompted"
    else:
        tw = c.post(f"/v1/projects/{pid}/chat", json={"provider": "claude-cli", "model": a.model_b, "message": (
            "Change sharpe() in src/sharpe.py to annualise with sqrt(252) trading days, then run the tests.")}).json()
        when = "proposed after the person asked for the change"
    proposed = [(t["tool"], t["args"].get("path"), t["status"]) for t in tw.get("tasks", [])
                if t["status"] == "awaiting_approval"]
    record(12, "CI proposed a reversible action", bool(proposed), note=when, proposed=proposed,
           file_unchanged=(repo / "src" / "sharpe.py").read_text() == SHARPE_365)
    decisions, test_task, write_task = [], None, None

    def drive(tw):
        nonlocal test_task, write_task
        rounds = 0
        while tw.get("status") == "awaiting_approval" and rounds < 6:
            rounds += 1
            for tid in tw["pending"]:
                t = c.get(f"/v1/tasks/{tid}").json()
                ok = t["tool"] == "run_tests" or (t["tool"] == "write_file" and t["args"].get("path") == "src/sharpe.py")
                decisions.append({"task": tid, "tool": t["tool"], "approved": ok})
                if ok:
                    tw = c.post(f"/v1/tasks/{tid}/approve").json()
                    test_task = tid if t["tool"] == "run_tests" else test_task
                    write_task = tid if t["tool"] == "write_file" else write_task
                else:
                    tw = c.post(f"/v1/tasks/{tid}/deny", json={"reason": "only src/sharpe.py may change"}).json()
        return tw
    tw = drive(tw)
    if write_task and not test_task and tw.get("status") == "done":  # the person asks for the check once
        tw = drive(c.post(f"/v1/projects/{pid}/chat", json={"provider": "claude-cli", "model": a.model_b,
                                                            "message": "Now run the tests."}).json())
    record(13, "the person approved", any(d["approved"] for d in decisions), decisions=decisions)
    wt = c.get(f"/v1/tasks/{write_task}").json() if write_task else {}
    tt = c.get(f"/v1/tasks/{test_task}").json() if test_task else {}
    verified = None
    if tt.get("status") == "succeeded":
        mine = c.post(f"/v1/projects/{pid}/claims", json={"statement": "check.py passes after the change",
                                                          "evidence_refs": [test_task]}).json()
        verified = c.post(f"/v1/projects/{pid}/claims/{mine['id']}/resolve", json={"test_task_id": test_task}).json()
    record(14, "executed, real result recorded", wt.get("status") == "succeeded",
           seconds=round(time.time() - t0, 1), write=wt.get("result"), test=tt.get("result"), test_status=tt.get("status"),
           claim_settled_by_test=verified and {k: verified[k] for k in ("statement", "status", "resolution")},
           file_now=(repo / "src" / "sharpe.py").read_text(), final_reply=tw.get("text"))

    # 15-16. revoke and prove it holds
    rv = c.delete(f"/v1/projects/{pid}/providers/anthropic").json()
    record(15, "provider access revoked", rv["vendors"] == [], vendors_now=rv["vendors"])
    before = sum(1 for e in c.get("/v1/audit").json()["entries"] if e["event"] == "session.disclosure")
    chat = c.post(f"/v1/projects/{pid}/chat", json={"provider": "claude-cli", "model": a.model_b, "message": "hello?"})
    ver = c.post(f"/v1/projects/{pid}/claims/{claims[0]['id']}/verify", json={"provider": "claude-cli",
                                                                             "model": "sonnet"}) if claims else None
    audit = c.get("/v1/audit").json()
    after = sum(1 for e in audit["entries"] if e["event"] == "session.disclosure")
    blocked = [e for e in audit["entries"] if e["event"] == "egress.blocked"]
    record(16, "subsequent access blocked", chat.status_code == 403 and (ver is None or ver.status_code == 403)
           and after == before, chat_http=chat.status_code, chat_detail=chat.json().get("detail"),
           verify_http=ver.status_code if ver is not None else None, new_disclosures=after - before,
           egress_blocked_events=len(blocked), audit_chain_ok=audit["chain_ok"])
    srv.stop()

    report = {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t_start, 1),
              "provider_a": f"claude-cli:{a.model_a}", "provider_b": f"claude-cli:{a.model_b}",
              "passed": sum(s["ok"] for s in steps), "of": len(steps), "steps": steps, "disclosures": disc}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    lines = [f"# Milestone 1, live run ({report['ran_at']})", "",
             f"Provider A: `{report['provider_a']}` · Provider B: `{report['provider_b']}` · "
             f"{report['passed']}/{report['of']} steps passed · {report['seconds']} s", ""]
    for s in steps:
        lines.append(f"- **{s['step']}. {s['title']}**: {'passed' if s['ok'] else 'did not pass'}"
                     + (f" ({s['note']})" if s.get("note") else ""))
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n{report['passed']}/{report['of']} steps passed in {report['seconds']} s; report in {out}")
    return 0 if report["passed"] == report["of"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
