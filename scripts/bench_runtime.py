"""Runtime benchmark: where the time goes in one CI request, measured rather than guessed.

    .venv/bin/python scripts/bench_runtime.py --out bench.json [--sizes 100 1000 3000] [--reps 20]

Offline and deterministic in setup. The model is the mock agent (zero latency), so every
number is CI's own overhead: context selection, briefing, encryption, SQLite, audit, the
agent loop. Model latency is measured separately, in live runs (docs/CI_BENCHMARKS.md).

Per memory size N it reports, as median and p95 milliseconds over --reps repetitions:
- select_context, split into its stages (load current memory, embeddings, egress, candidate
  generation, policy selection);
- a first agent turn (new session, briefing compiled) and a follow-up turn;
- one read_file tool call through the governor and executor;
plus SQL statements and record decryptions per follow-up turn, peak Python memory during
select_context, prompt size of the first turn, and throughput with concurrent sessions.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np

import pcim.store as store_mod
from pcim.adapters.mock_agent import MockAgentAdapter
from pcim.agent import Agent
from pcim.engine import PersonalCI
from pcim.exchange import ClaimLedger
from pcim.governor import Governor
from pcim.policy.candidates import build_candidates
from pcim.memory import effective_confidence
from pcim.intent import classify_task
from pcim.tasks import TaskQueue
from pcim.types import KINDS, TASK_TYPES, approx_tokens, utcnow
from pcim.workspace import Workspace

WORDS = ("budget backtest crowding signal sharpe volatility portfolio equity bond momentum value carry "
         "travel dinner meeting deadline report draft paper lemma proof dataset model training python "
         "rust latency cache index query schema migration review hiring offer visa rent gym diet coffee").split()


class Instant(MockAgentAdapter):
    """Answers every step at once with no tool calls."""

    def agent_step(self, system, messages, tools, max_tokens=None):
        self.requests.append(len(json.dumps(messages, default=str)))
        return self._step("ok")


def ms(xs):
    xs = sorted(xs)
    return {"median_ms": round(statistics.median(xs) * 1000, 3), "p95_ms": round(xs[int(0.95 * (len(xs) - 1))] * 1000, 3)}


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def build(root: Path, n_items: int, seed: int = 0):
    rng = random.Random(seed)
    ci = PersonalCI(root / "data", seed=seed)
    ci.onboard("u", providers=("mock",))
    for i in range(n_items):
        kind = KINDS[i % len(KINDS)]
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 18))) + f" #{i}"
        ci.remember("u", text, kind)
    ws = Workspace(ci)
    ledger = ClaimLedger(ws)
    q = TaskQueue(ws, Governor(ws), claims=ledger)
    agent = Agent(ci, ws, q, ledger)
    agent.set_adapter("mock", None, Instant())
    repo = root / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("alpha\n" * 50)
    return ci, ws, q, agent, repo


def stage_breakdown(ci, task, reps):
    stages = {k: [] for k in ("memory_current", "embeddings", "egress_filter", "candidates", "policy_select")}
    for _ in range(reps):
        now = utcnow()
        t = time.perf_counter(); current = ci.memory.current("u", now); stages["memory_current"].append(time.perf_counter() - t)
        t = time.perf_counter(); ids, mat = ci.memory.embeddings("u"); stages["embeddings"].append(time.perf_counter() - t)
        emb = {i: mat[j] for j, i in enumerate(ids)} if mat is not None else {}
        t = time.perf_counter(); d = ci.egress.filter("u", current, "mock", "bench", ref="bench"); stages["egress_filter"].append(time.perf_counter() - t)
        task_emb = ci.embedder.embed([task])[0]
        t = time.perf_counter(); cs = build_candidates(d.allowed, emb, task_emb, effective_confidence, now); stages["candidates"].append(time.perf_counter() - t)
        probs = classify_task(task)
        pol = ci.learner.policy_for("u")
        t = time.perf_counter(); pol.select(cs, np.array([probs[k] for k in TASK_TYPES]), 1200, ci.rng, propensity_samples=ci.propensity_samples)
        stages["policy_select"].append(time.perf_counter() - t)
    return {k: ms(v) for k, v in stages.items()}


def count_turn(ci, agent, pid):
    """SQL statements and record decryptions in one follow-up turn."""
    sql = [0]
    ci.store._db.set_trace_callback(lambda s: sql.__setitem__(0, sql[0] + 1))
    calls = [0]
    real = store_mod.unseal

    def counting(*a, **k):
        calls[0] += 1
        return real(*a, **k)
    store_mod.unseal = counting
    try:
        agent.send("u", pid, "what is the next step for the backtest budget?")
    finally:
        store_mod.unseal = real
        ci.store._db.set_trace_callback(None)
    return {"sql_statements": sql[0], "record_decryptions": calls[0]}


def concurrency(root: Path, threads: int, turns: int):
    ci, ws, q, agent, _ = build(root, 200, seed=1)
    pids = [ws.create_project("u", f"P{i}").id for i in range(threads)]
    errors = []

    def worker(pid):
        try:
            for k in range(turns):
                agent.send("u", pid, f"turn {k}")
        except Exception as e:  # recorded, not swallowed: the benchmark reports it
            errors.append(repr(e))
    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(p,)) for p in pids]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = time.perf_counter() - t0
    return {"threads": threads, "turns_each": turns, "seconds": round(dt, 3),
            "turns_per_second": round(threads * turns / dt, 1), "errors": errors[:5], "n_errors": len(errors)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 3000])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = "plan the backtest budget and the meeting deadline for the crowding paper"
    results = {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reps": a.reps, "sizes": {}}
    for n in a.sizes:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            t0 = time.perf_counter()
            ci, ws, q, agent, repo = build(root, n)
            setup = time.perf_counter() - t0
            p = ws.create_project("u", "Bench", objective="measure CI's overhead", constraints=["offline"])
            ws.authorize_workspace("u", p.id, str(repo))
            for i in range(30):
                ws.add_note("u", p.id, ("open_question", "decision", "next_step")[i % 3], f"note {i} about the budget")
            r = {"setup_seconds": round(setup, 2)}
            r["select_context"] = ms(timed(lambda: ci.select_context("u", task, "mock", "bench", 1200, ref="b"), a.reps))
            r["select_context_stages"] = stage_breakdown(ci, task, a.reps)
            tracemalloc.start()
            ci.select_context("u", task, "mock", "bench", 1200, ref="b")
            r["select_context_peak_kib"] = round(tracemalloc.get_traced_memory()[1] / 1024, 1)
            tracemalloc.stop()
            first = []
            for _ in range(max(3, a.reps // 4)):
                t0 = time.perf_counter()
                agent.send("u", p.id, "start", new_session=True)
                first.append(time.perf_counter() - t0)
            r["agent_first_turn"] = ms(first)
            r["first_turn_prompt_tokens"] = approx_tokens("x" * agent.adapter("mock", None).requests[-1])
            r["agent_followup_turn"] = ms(timed(lambda: agent.send("u", p.id, "next?"), a.reps))
            r["per_followup_turn"] = count_turn(ci, agent, p.id)
            r["tool_read_file"] = ms(timed(lambda: q.propose("u", p.id, "read_file", {"path": "a.txt"}), a.reps))
            r["briefing_build_and_render"] = ms(timed(lambda: agent.briefing("u", p.id, vendor="mock"), a.reps))
            results["sizes"][str(n)] = r
            ci.store.close()
            print(f"N={n}: select {r['select_context']['median_ms']} ms, first turn {r['agent_first_turn']['median_ms']} ms, "
                  f"follow-up {r['agent_followup_turn']['median_ms']} ms, {r['per_followup_turn']}", flush=True)
    with tempfile.TemporaryDirectory() as d:
        results["concurrency"] = concurrency(Path(d), threads=8, turns=10)
        print("concurrency:", results["concurrency"], flush=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
