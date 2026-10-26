"""E1: does CI's context engine change answers on a fixed task set, against no context and
against naive conversation history? A capability test, not a general benchmark.

    python -m pcim.observatory.e1 --prompts-only --out experiments/e1/results/prompts       # no model calls
    python -m pcim.observatory.e1 --model haiku --reps 2 --out experiments/e1/results/haiku # live, spends usage

Arms (experiments/e1/PREREGISTRATION.md):
    A              the task alone
    B              naive history: every statement the person ever made, oldest first, as prior conversation
    C              CI: scope → temporal → egress → learned selection under a 1200-token budget → render
    C_all          every statement through CI's renderer, no filtering (CI's presentation without its selection)
    C_no_temporal  C with superseded statements kept current
    C_no_egress    C with private items granted and the document's instructions confirmed
    C_no_scope     C with project scoping removed
    C_unlimited    C with a 100k-token budget

Each ablation removes one CI component by changing the data so it has nothing to act on,
so no experiment-only switch exists in production code. Grading is deterministic (regex
rules fixed in dataset.json before any run); no model grades another.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime as dt
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
import time
from pathlib import Path

from ..adapters import ContextPacket, PacketItem, RenderedPrompt, render
from ..engine import PersonalCI
from ..types import Sensitivity, Trust
from ..workspace import Workspace

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "e1"
ARMS = ("A", "B", "C", "C_all", "C_no_temporal", "C_no_egress", "C_no_scope", "C_unlimited")
VARIANT = {"C": "full", "C_unlimited": "full", "C_no_temporal": "no_temporal", "C_no_egress": "no_egress",
           "C_no_scope": "no_scope"}
BASE_SYSTEM = ("You are assisting one specific person. Answer their request directly and concisely. "
               "Never claim to remember anything that is not in the conversation.")
STALE = {"m_city_old", "m_meet_old", "m_budget_old"}
PROVIDER = "claude-cli"


def load_dataset(path: Path = EXP / "dataset.json") -> dict:
    return json.loads(path.read_text())


def _utc(day: str) -> dt.datetime:
    return dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)


def build_ci(ds: dict, root: Path, variant: str) -> tuple[PersonalCI, dict[str, str], dict[str, str]]:
    """A CI store holding the dataset's memories, written through the normal APIs."""
    ci = PersonalCI(root / variant, seed=0)
    ci.onboard("u", providers=(PROVIDER,))
    ws = Workspace(ci)
    pids = {name: ws.create_project("u", name).id for name in ds["projects"]}
    to_ds: dict[str, str] = {}
    for m in ds["memories"]:
        tags = (f"project:{pids[m['project']]}",) if m.get("project") and variant != "no_scope" else ()
        if m.get("private"):
            src = m["private"]
            ci.consent("u", src, "store", max_sensitivity=Sensitivity.SENSITIVE)
            if variant == "no_egress":
                ci.consent("u", src, "egress", "anthropic", Sensitivity.SENSITIVE)
            r = ci.memory.remember("u", m["content"], m["kind"], source=src, trust=Trust.IMPORTED,
                                   sensitivity=Sensitivity(m.get("sensitivity", "personal")),
                                   valid_from=_utc(m["date"]), tags=tags)
        elif m.get("document"):
            src = m["document"]
            ci.consent("u", src, "store")
            ci.consent("u", src, "egress", "anthropic")
            r = ci.ingest_text("u", m["content"], source=src, title="vendor brochure", tags=tags)[0]
            if variant == "no_egress":
                ci.memory.confirm("u", r.item.id, promote=False)  # lifts the instruction hold
        else:
            key = m.get("key")
            if variant == "no_temporal" and key:
                key = f"{key}#{m['id']}"  # every version stays current
            r = ci.memory.remember("u", m["content"], m["kind"], key=key, valid_from=_utc(m["date"]), tags=tags)
        to_ds[r.item.id] = m["id"]
    return ci, pids, to_ds


def _nonce(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _trust(m: dict) -> str:
    return "untrusted" if m.get("document") else "imported" if m.get("private") else "user_stated"


def build_prompts(ds: dict, workdir: Path, model_label: str) -> dict[tuple[str, str], dict]:
    """(arm, task id) -> {"prompt": RenderedPrompt, "included": [dataset memory ids]}."""
    out: dict[tuple[str, str], dict] = {}
    history = "\n".join(f"- {'I shared this document: ' if m.get('document') else ''}{m['content']}"
                        for m in ds["memories"])
    every = [PacketItem(m["id"], m["kind"], m["content"], m.get("private") or m.get("document") or "user", _trust(m),
                        m["date"], 1.0) for m in ds["memories"]]
    stores = {v: build_ci(ds, workdir, v) for v in sorted(set(VARIANT.values()))}
    for t in ds["tasks"]:
        out[("A", t["id"])] = {"prompt": RenderedPrompt(BASE_SYSTEM, t["text"]), "included": []}
        out[("B", t["id"])] = {"prompt": RenderedPrompt(
            BASE_SYSTEM, f"Earlier in our conversations I told you:\n{history}\n\nRequest:\n{t['text']}"),
            "included": [m["id"] for m in ds["memories"]]}
        out[("C_all", t["id"])] = {"prompt": render(ContextPacket(
            f"e1-{t['id']}-all", t["text"], [], [], every, _nonce("C_all", t["id"]))),
            "included": [m["id"] for m in ds["memories"]]}
        for arm, variant in VARIANT.items():
            ci, pids, to_ds = stores[variant]
            budget = 100_000 if arm == "C_unlimited" else 1200
            cx = ci.select_context("u", t["text"], PROVIDER, model_label, budget, ref=f"e1-{arm}-{t['id']}",
                                   project_id=pids[t["project"]])
            packet = ContextPacket(f"e1-{arm}-{t['id']}", t["text"], [], cx.profile, cx.memories, _nonce(arm, t["id"]))
            out[(arm, t["id"])] = {"prompt": render(packet), "included": [to_ds[i.id] for i in cx.chosen]}
    for ci, _, _ in stores.values():
        ci.store.close()
    return out


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def prompt_metrics(ds: dict, task: dict, entry: dict) -> dict:
    p = entry["prompt"]
    text = p.system + "\n" + p.user
    inc, rel = set(entry["included"]), set(task["relevant"])
    return {"prompt_tokens_est": (len(text) + 3) // 4, "included": sorted(inc),
            "prompt_private": _any(ds["private_patterns"], text),
            "prompt_injection": _any(ds["injection_patterns"], text),
            "stale_in_prompt": bool(inc & STALE),
            "precision": (len(inc & rel) / len(inc)) if inc else None,
            "recall": (len(inc & rel) / len(rel)) if rel else None,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}


def grade(ds: dict, task: dict, text: str) -> dict:
    req = [bool(re.search(p, text, re.IGNORECASE)) for p in task["required"]]
    bad = [p for p in task["forbidden"] if re.search(p, text, re.IGNORECASE)]
    return {"success": all(req) and not bad, "required_hit": req, "forbidden_hit": bad,
            "output_private": _any(ds["private_patterns"], text),
            "output_injection": _any(ds["injection_patterns"], text)}


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(ds: dict, rows: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    summary = {}
    for arm in ARMS:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        live = [r for r in rs if "success" in r]
        k, n = sum(r["success"] for r in live), len(live)
        prec = [r["precision"] for r in rs if r["precision"] is not None]
        rec = [r["recall"] for r in rs if r["recall"] is not None]
        s = {"n_prompts": len(rs),
             "prompt_private_rate": sum(r["prompt_private"] for r in rs) / len(rs),
             "prompt_injection_rate": sum(r["prompt_injection"] for r in rs) / len(rs),
             "stale_in_prompt_rate": sum(r["stale_in_prompt"] for r in rs) / len(rs),
             "context_precision_mean": statistics.mean(prec) if prec else None,
             "context_recall_mean": statistics.mean(rec) if rec else None,
             "prompt_tokens_est_median": statistics.median(r["prompt_tokens_est"] for r in rs)}
        if n:
            cost = [r["cost_usd"] for r in live if r.get("cost_usd") is not None]
            s.update({"n": n, "successes": k, "success_rate": k / n, "success_ci95": wilson(k, n),
                      "errors": sum(1 for r in rs if r.get("error")),
                      "output_private_rate": sum(r["output_private"] for r in live) / n,
                      "injection_followed_rate": (sum(r["output_injection"] for r in live if r["category"] == "injection")
                                                  / max(1, sum(1 for r in live if r["category"] == "injection"))),
                      "by_category": {cat: f"{sum(r['success'] for r in live if r['category'] == cat)}/"
                                           f"{sum(1 for r in live if r['category'] == cat)}"
                                      for cat in sorted({r["category"] for r in live})},
                      "tokens_in_median": statistics.median(r["tokens_in"] for r in live),
                      "latency_s_median": round(statistics.median(r["latency_s"] for r in live), 2),
                      "cost_usd_total": round(sum(cost), 4) if cost else None,
                      "cost_per_success_usd": round(sum(cost) / k, 5) if cost and k else None})
        summary[arm] = s
    pairs = {}
    index = {(r["arm"], r["task"], r["rep"]): r for r in rows if "success" in r}
    for x, y in [("C", "A"), ("C", "B"), ("C", "C_all")] + [(a, "C") for a in ARMS if a.startswith("C_") and a != "C_all"]:
        keys = [(t, rep) for (arm, t, rep) in index if arm == x and (y, t, rep) in index]
        b = sum(1 for t, rep in keys if index[(x, t, rep)]["success"] and not index[(y, t, rep)]["success"])
        c = sum(1 for t, rep in keys if not index[(x, t, rep)]["success"] and index[(y, t, rep)]["success"])
        if keys:
            pairs[f"{x} vs {y}"] = {"pairs": len(keys), f"{x} only": b, f"{y} only": c, "p_exact": round(mcnemar_exact(b, c), 4)}
    return {"arms": summary, "paired": pairs}


# ---------------------------------------------------------------- running
def file_hashes() -> dict[str, str]:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() if (ROOT / p).exists() else None
            for p in ("experiments/e1/dataset.json", "experiments/e1/PREREGISTRATION.md", "src/pcim/observatory/e1.py")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-only", action="store_true")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args(argv)
    ds = load_dataset()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hashes = file_hashes()
    registered = (EXP / "PREREG_HASHES.json")
    confirmatory = registered.exists() and json.loads(registered.read_text()) == hashes
    with tempfile.TemporaryDirectory() as d:
        prompts = build_prompts(ds, Path(d), a.model)
    tasks = {t["id"]: t for t in ds["tasks"]}
    jobs = [(arm, tid, rep) for arm in a.arms for tid in tasks for rep in range(1 if a.prompts_only else a.reps)]
    random.Random(a.seed).shuffle(jobs)  # interleave arms so drift in the provider hits all of them alike
    rows = []
    for arm, tid, rep in jobs:
        t = tasks[tid]
        rows.append({"arm": arm, "task": tid, "rep": rep, "category": t["category"],
                     **prompt_metrics(ds, t, prompts[(arm, tid)])})
    if not a.prompts_only:
        from ..adapters.base import ProviderError
        from ..adapters.claude_cli import ClaudeCLIAdapter
        ad = ClaudeCLIAdapter(a.model, workdir=Path(tempfile.mkdtemp(prefix="e1-cli-")), max_budget_usd=0.5)

        def call(row):
            p = prompts[(row["arm"], row["task"])]["prompt"]
            for attempt in range(2):  # one retry on a provider error, as registered
                try:
                    t0 = time.perf_counter()
                    r = ad.generate(p)
                    u = r.usage
                    return {"response": r.text, "served_model": r.model, "cost_usd": r.cost_usd,
                            "tokens_in": u.input_tokens + u.cache_read_tokens + u.cache_write_tokens,
                            "tokens_out": u.output_tokens, "latency_s": time.perf_counter() - t0,
                            **grade(ds, tasks[row["task"]], r.text)}
                except ProviderError as e:
                    err = str(e)[:300]
            return {"error": err, "response": "", "cost_usd": None, "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
                    **grade(ds, tasks[row["task"]], "")}  # a failed call counts as a failed task
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for row, res in zip(rows, pool.map(call, rows)):
                row.update(res)
                print(f"{row['arm']:>14} {row['task']} rep{row['rep']}: "
                      f"{'ok ' if row['success'] else 'NO '}{'(error) ' if row.get('error') else ''}", flush=True)
    result = {"experiment": "e1", "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": None if a.prompts_only else a.model,
              "reps": 1 if a.prompts_only else a.reps, "seed": a.seed, "arms": a.arms, "file_hashes": hashes,
              "confirmatory": confirmatory and not a.prompts_only, "summary": summarize(ds, rows), "rows": rows}
    (out / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["summary"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
