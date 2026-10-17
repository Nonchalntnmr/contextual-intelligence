"""G0 runner: experiments/g0/PREREGISTRATION.md, executed.

    python -m pcim.observatory.g0 --dev          # small configuration on dev seed 0
    python -m pcim.observatory.g0 --confirm      # the pre-registered run, once

A confirmatory run refuses to start unless PREREGISTRATION.md and config.json hash to
the values recorded in PREREG_HASHES.txt at freeze time, or the deviation log explains
the difference (`--accept-deviations`, which records the new hashes in the result).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import os
import platform
import sys
import time

from pathlib import Path

import numpy as np

from ..policy.features import N_FEATURES, featurize, regression_vector
from ..policy.heuristic import HeuristicParams, HeuristicPolicy, ProfilePlusSimilarityPolicy, SimilarityPolicy
from ..policy.learned import BayesLinearPolicy, PopulationPrior, fit_population_prior
from . import metrics as M
from .sim import N_KIND, N_TASK, World, feedback, oracle_select, random_select

ROOT = Path(__file__).resolve().parents[3]
G0_DIR = ROOT / "experiments" / "g0"  # default experiment; --exp selects another
SYSTEMS = ["A", "B1", "B2", "C", "D", "E1", "E2", "O"]


def _rng(ss: np.random.SeedSequence) -> np.random.Generator:
    return np.random.default_rng(ss)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------- stages
def tune_b2(world: World, cfg: dict, model: dict, rng: np.random.Generator) -> tuple[HeuristicParams, float]:
    """Random search over B2's parameters against noiseless quality (the generous baseline)."""
    t = cfg["b2_tuning"]
    budget = cfg["budget_tokens"]
    cases = []
    for _ in range(t["n_users"]):
        user = world.new_user(rng)
        for _ in range(t["n_tasks"]):
            task = user.new_task(rng)
            idx, cs = user.candidates(task, cfg["candidates"])
            cases.append((user, task, idx, cs, user.utilities(task, model)))
    configs = [HeuristicParams()] + [HeuristicParams.random(rng, allow_density=t.get("allow_density", False))
                                     for _ in range(t["n_configs"])]
    for p in configs:  # same near-duplicate threshold as the learned policies (D2)
        p.dedup_cos = cfg["policy"]["dedup_cos"]
    best, best_q = None, -np.inf
    for p in configs:
        pol = HeuristicPolicy(p)
        q = np.mean([u.quality(tk, idx[pol.select(cs, tk.probs, budget).idx], ut, model)
                     for u, tk, idx, cs, ut in cases])
        if q > best_q:
            best, best_q = p, q
    return best, float(best_q)


def collect_logs(world: World, cfg: dict, model: dict, b2: HeuristicPolicy, fb_kind: str,
                 rng: np.random.Generator, n_users: int, n_inter: int) -> list[tuple[np.ndarray, np.ndarray]]:
    lp, budget = cfg["logging_policy"], cfg["budget_tokens"]
    logs = []
    for _ in range(n_users):
        user = world.new_user(rng)
        X, R = [], []
        for _ in range(n_inter):
            task = user.new_task(rng)
            idx, cs = user.candidates(task, cfg["candidates"])
            util = user.utilities(task, model)
            if rng.random() < lp["random_prob"]:
                sel = random_select(cs, budget, lp["random_include_prob"], rng)
            else:
                sel = b2.select(cs, task.probs, budget).idx
            q = user.quality(task, idx[sel], util, model)
            r = feedback(q, fb_kind, cfg["feedback"], rng.random(), rng.random(), rng.standard_normal())
            if r is not None:
                X.append(regression_vector(featurize(cs, task.probs), sel, task.probs))
                R.append(r)
        if R:
            logs.append((np.array(X), np.array(R)))
    return logs


def run_condition(cond: dict, cfg: dict, seed: int, scale: dict | None = None) -> dict:
    """One pre-registered condition end to end. `scale` shrinks counts for --dev runs."""
    t0 = time.time()
    proto = dict(cfg["protocol"])
    if scale:
        proto.update({k: v for k, v in scale.items() if not k.startswith("b2_")})
        cfg = copy.deepcopy(cfg)
        for k in ("n_configs", "n_users", "n_tasks"):
            if f"b2_{k}" in scale:
                cfg["b2_tuning"][k] = scale[f"b2_{k}"]
    # D2: one seed sequence for every condition (common random numbers across conditions),
    # so conditions differ only in the factors they manipulate — the same A_raw, P, users,
    # tasks and noise draws, with λ scaling them. Per-condition seeds confounded, e.g.,
    # dense-vs-binary comparisons with a different random world.
    ss = np.random.SeedSequence(seed)
    s_world, s_tune, s_train, s_train2, s_test, s_pol = ss.spawn(6)
    world = World.create(cfg["world"], cond["lambda_shared"], cond["lambda_user"], _rng(s_world),
                         similarity_world=cond.get("similarity_world", False))
    m_train = cfg["models"][cond.get("train_model", "M1")]
    m_test = cfg["models"][cond.get("test_model", "M1")]
    pc, budget, T = cfg["policy"], cfg["budget_tokens"], proto["n_interactions"]

    b2_params, b2_tune_q = tune_b2(world, cfg, m_train, _rng(s_tune))
    b2 = HeuristicPolicy(b2_params)
    logs = collect_logs(world, cfg, m_train, b2, cond["feedback"], _rng(s_train),
                        proto["n_train_users"], T)
    prior = fit_population_prior(logs, rank=pc["prior_rank"], iters=pc["em_iters"])
    weak = PopulationPrior.weak(sd=pc["weak_prior_sd"], noise_var=prior.noise_var)

    # G0 registered value ranking and Thompson-sampling E2 (secondary: posterior-mean E2g);
    # G0b registers density ranking and posterior-mean E2 (secondary: Thompson E2ts).
    rank = pc.get("ranking", "value")
    e2_greedy = pc.get("e2_mode", "thompson") == "greedy"
    secondary = pc.get("secondary", "E2g")

    def bl(p, name, greedy=e2_greedy, frozen=False, forgetting=pc["forgetting"]):
        return lambda: BayesLinearPolicy(p, name=name, frozen=frozen, greedy=greedy, rank=rank,
                                         forgetting=forgetting, ts_scale=pc["ts_scale"], dedup_cos=pc["dedup_cos"])

    learners = {
        "D": bl(prior, "D", frozen=True),
        # E1 is Thompson sampling in both experiments, as registered: a posterior-mean learner
        # on a zero prior never includes anything and so never learns, which would make G0-4
        # pass trivially rather than measure what the population prior is worth.
        "E1": bl(weak, "E1", greedy=False),
        "E2": bl(prior, "E2"),
        # The other exploration mode, reported but never gated.
        secondary: bl(prior, secondary, greedy=not e2_greedy),
    }
    systems = list(SYSTEMS) + [secondary]
    if cond.get("drift"):
        learners["E2f"] = bl(prior, "E2f", forgetting=pc["forgetting_drift"])
        systems.append("E2f")
    if cond.get("test_model") and cond["test_model"] != cond.get("train_model", "M1"):
        logs_in = collect_logs(world, cfg, m_test, b2, cond["feedback"], _rng(s_train2),
                               proto["n_train_users"], T)
        prior_in = fit_population_prior(logs_in, rank=pc["prior_rank"], iters=pc["em_iters"])
        learners["E2in"] = bl(prior_in, "E2in")
        systems.append("E2in")
    fixed = {"B1": SimilarityPolicy(), "B2": b2, "C": ProfilePlusSimilarityPolicy()}

    n_users = proto["n_test_users"]
    Q = {s: np.zeros((n_users, T)) for s in systems}
    TOK = {s: np.zeros((n_users, T)) for s in systems}
    est_dev = np.zeros((n_users, N_KIND * N_TASK))
    true_dev = np.zeros((n_users, N_KIND * N_TASK))
    rng = _rng(s_test)
    pol_rngs = {s: _rng(x) for s, x in zip(systems, s_pol.spawn(len(systems)))}
    for ui in range(n_users):
        user = world.new_user(rng)
        live = {name: make() for name, make in learners.items()}
        for t in range(T):
            if cond.get("drift") and t == proto["drift_at"]:
                user.redraw_z(rng)
            task = user.new_task(rng)
            idx, cs = user.candidates(task, cfg["candidates"])
            util = user.utilities(task, m_test)
            F = featurize(cs, task.probs)
            u1, u2, nz = rng.random(), rng.random(), rng.standard_normal()
            for s in systems:
                if s == "A":
                    sel_idx, pol = np.zeros(0, dtype=int), None
                elif s == "O":
                    sel_idx, pol = oracle_select(user, task, idx, util, m_test, budget), None
                elif s in fixed:
                    sel_idx, pol = fixed[s].select(cs, task.probs, budget).idx, None
                else:
                    pol = live[s]
                    sel = pol.select(cs, task.probs, budget, pol_rngs[s], F=F)
                    sel_idx = sel.idx
                q = user.quality(task, idx[sel_idx], util, m_test)
                Q[s][ui, t] = q
                TOK[s][ui, t] = cs.tokens[sel_idx].sum()
                if pol is not None and not pol.frozen:
                    r = feedback(q, cond["feedback"], cfg["feedback"], u1, u2, nz)
                    if r is not None:
                        pol.update(sel.x, r)
        # value[t|k] coordinates are kind-major: index k*N_TASK + t.
        est_dev[ui] = live["E2"].deviation()[: N_KIND * N_TASK]
        true_dev[ui] = user.deviation.T.reshape(-1)

    brng = np.random.default_rng(seed + 1)
    nb = proto["bootstrap"]
    windows = proto["windows"]
    contrasts = [("E2", "B2"), ("E2", "D"), ("E2", "E1"), ("E2", "C"), ("B2", "B1"), ("D", "B2"), ("O", "B2"),
                 ("B2", "A"), (secondary, "D"), (secondary, "B2"), (secondary, "E2")]
    if "E2f" in systems:
        contrasts.append(("E2f", "E2"))
    if "E2in" in systems:
        contrasts += [("E2in", "E2"), ("E2", "B2")]
    out = {
        "condition": cond,
        "seconds": None,
        "b2_params": b2_params.__dict__,
        "b2_tuning_quality": b2_tune_q,
        "prior": {"noise_var": prior.noise_var, "basis_var": prior.basis_var.tolist(), "floor": prior.floor,
                  "n_users": prior.n_users, "n_logged": int(sum(len(r) for _, r in logs))},
        "mean_quality": {s: {f"{lo}-{hi}": float(M.window_mean(Q[s], lo, hi).mean()) for lo, hi in windows}
                         | {"1-200": float(Q[s].mean())} for s in systems},
        "tokens_per_interaction": {s: float(TOK[s].mean()) for s in systems},
        "learning_curve": {s: M.learning_curve(Q[s]) for s in systems},
        "contrasts": {},
        "capture_fraction": {f"{lo}-{hi}": M.capture_fraction(Q["E2"], Q["B2"], Q["O"], lo, hi, nb, brng)
                             for lo, hi in windows},
        "onset_vs_D": M.onset(Q["E2"], Q["D"], 25, nb, brng),
        "onset_vs_B2": M.onset(Q["E2"], Q["B2"], 25, nb, brng),
        "secondary": secondary,
        "onset_secondary_vs_D": M.onset(Q[secondary], Q["D"], 25, nb, brng),
        "identifiability": M.identifiability(est_dev, true_dev),
    }
    for a, b in dict.fromkeys(contrasts):
        out["contrasts"][f"{a}-{b}"] = {f"{lo}-{hi}": M.paired_diff(Q[a], Q[b], lo, hi, nb, brng)
                                        for lo, hi in windows + [[1, T]]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------- gates
def evaluate_gates(res: dict[str, dict]) -> list[dict]:
    def c(cond, contrast, window):
        return res[cond]["contrasts"][contrast][window]

    gates = []
    g = c("main-u1-binary", "E2-B2", "101-200")
    cf_ = res["main-u1-binary"]["capture_fraction"]["101-200"]
    gates.append({"gate": "G0-1 positive control", "pass": g["lo"] > 0 and cf_["mean"] >= 0.25,
                  "evidence": f"E2−B2 101–200 = {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]; "
                              f"CF = {cf_['mean']:.2f} [{cf_['lo']:.2f}, {cf_['hi']:.2f}]"})
    g = c("negctrl", "E2-B2", "1-200")
    gates.append({"gate": "G0-2 negative control", "pass": g["lo"] >= -0.01,
                  "evidence": f"E2−B2 1–200 = {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]"})
    n_star = res["main-u1-sparse"]["onset_vs_D"]
    gates.append({"gate": "G0-3 sample efficiency", "pass": n_star is not None and n_star <= 100,
                  "evidence": f"onset n* (E2 vs D, sparse) = {n_star}"})
    g50 = res["main-u1-binary"]["contrasts"]["E2-E1"].get("1-50")  # run_all adds the 1–50 window
    gates.append({"gate": "G0-4 population prior", "pass": g50 is not None and g50["lo"] > 0,
                  "evidence": f"E2−E1 1–50 = {g50['mean']:+.4f} [{g50['lo']:+.4f}, {g50['hi']:+.4f}]" if g50 else "missing"})
    g = c("main-u1-binary", "E2-D", "101-200")
    gates.append({"gate": "G0-5a personalization exists", "pass": g["lo"] > 0,
                  "evidence": f"E2−D 101–200 (λ_user=1) = {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]"})
    g = c("main-u0-binary", "E2-D", "101-200")
    gates.append({"gate": "G0-5b personalization is free when absent", "pass": g["lo"] >= -0.01,
                  "evidence": f"E2−D 101–200 (λ_user=0) = {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]"})
    idf = res["main-u1-dense"]["identifiability"]
    gates.append({"gate": "G0-6 identifiability", "pass": idf["median"] >= 0.5,
                  "evidence": f"median corr = {idf['median']:.2f} (IQR {idf['q25']:.2f}–{idf['q75']:.2f}, n={idf['n']})"})
    return gates


def _cond_worker(args):
    cond, cfg, seed, scale = args
    res = run_condition(cond, cfg, seed, scale)
    return cond["name"], res


def run_all(cfg: dict, seed: int, scale: dict | None, workers: int, only: list[str] | None = None) -> dict:
    conds = [c for c in cfg["conditions"] if not only or c["name"] in only]
    # G0-4 needs the 1–50 window; add it to the windows used for contrasts.
    cfg = copy.deepcopy(cfg)
    cfg["protocol"]["windows"] = cfg["protocol"]["windows"] + [[1, 50]]
    results = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for name, res in ex.map(_cond_worker, [(c, cfg, seed, scale) for c in conds]):
            results[name] = res
            print(f"  {name:<18} done in {res['seconds']:>6.1f}s", flush=True)
    return results


def write_report(results: dict, gates: list[dict], meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "gates": gates, "results": results}, indent=1))
    L = [f"# G0 result — {meta['mode']}", "",
         f"Run {meta['started_utc']} · seed {meta['seed']} · {meta['python']} · numpy {meta['numpy']} · "
         f"{meta['wall_seconds']:.0f}s wall", "",
         f"Pre-registration sha256 `{meta['prereg_sha256'][:16]}…` · config sha256 `{meta['config_sha256'][:16]}…` · "
         f"hashes match freeze: **{meta['hashes_match_freeze']}**", ""]
    verdict = all(g["pass"] for g in gates) if len(gates) == 7 else None
    L += ["## Gates", "", "| gate | result | evidence |", "|---|---|---|"]
    for g in gates:
        L.append(f"| {g['gate']} | {'PASS' if g['pass'] else '**FAIL**'} | {g['evidence']} |")
    L += ["", f"**G0 overall: {'PASS' if verdict else ('FAIL' if verdict is False else 'incomplete')}**", ""]
    L += ["## Mean true quality by system (interactions 1–200)", "",
          "| condition | " + " | ".join(SYSTEMS) + " | tokens E2 / B2 |", "|---|" + "---|" * (len(SYSTEMS) + 1)]
    for name, r in results.items():
        mq = r["mean_quality"]
        row = " | ".join(f"{mq[s]['1-200']:.3f}" for s in SYSTEMS)
        L.append(f"| {name} | {row} | {r['tokens_per_interaction']['E2']:.0f} / {r['tokens_per_interaction']['B2']:.0f} |")
    L += ["", "## Key contrasts, interactions 101–200 (mean [95% CI], paired within user)", "",
          "| condition | E2−B2 | E2−D | E2−E1 | D−B2 | B2−B1 | headroom O−B2 | CF | onset vs D |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        cc = r["contrasts"]

        def f(k):
            x = cc[k]["101-200"]
            return f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
        cfx = r["capture_fraction"]["101-200"]
        L.append(f"| {name} | {f('E2-B2')} | {f('E2-D')} | {f('E2-E1')} | {f('D-B2')} | {f('B2-B1')} | "
                 f"{cfx['headroom']:.3f} | {cfx['mean']:.2f} | {r['onset_vs_D']} |")
    sec = next(iter(results.values())).get("secondary", "E2g")
    what = {"E2g": "E2's prior and updates, acting on the posterior mean instead of a Thompson draw. Conceived "
                   "after a dev run (G0 D2); any claim about it needs its own pre-registered confirmation.",
            "E2ts": "E2's prior and updates, acting on a Thompson draw — G0's registered mode, demoted to a "
                    "secondary system in G0b."}.get(sec, sec)
    L += ["", f"## Secondary system {sec} — reported, never gated", "", f"{sec} = {what}", "",
          f"| condition | {sec}−D 101–200 | {sec}−B2 101–200 | {sec}−E2 1–50 | {sec}−E2 101–200 | first 25-window {sec}>D |",
          "|---|---|---|---|---|---|"]
    for name, r in results.items():
        cc = r["contrasts"]

        def f(k, w="101-200"):
            x = cc[k][w]
            return f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
        L.append(f"| {name} | {f(sec + '-D')} | {f(sec + '-B2')} | {f(sec + '-E2', '1-50')} | {f(sec + '-E2')} | "
                 f"{r.get('onset_secondary_vs_D', r.get('onset_E2g_vs_D'))} |")
    for special, key in (("drift", "E2f-E2"), ("transfer", "E2in-E2")):
        if special in results and key in results[special]["contrasts"]:
            L += ["", f"## {special}: {key} by window", ""]
            for w, x in results[special]["contrasts"][key].items():
                L.append(f"- {w}: {x['mean']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]")
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    return out_dir / "REPORT.md"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true", help="small configuration on a dev seed")
    mode.add_argument("--confirm", action="store_true", help="the pre-registered confirmatory run")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--only", nargs="*", help="condition names (dev only)")
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--accept-deviations", action="store_true")
    ap.add_argument("--exp", default="g0", help="experiment directory under experiments/ (g0, g0b)")
    args = ap.parse_args(argv)
    exp_dir = ROOT / "experiments" / args.exp

    prereg, cfg_path = exp_dir / "PREREGISTRATION.md", exp_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    frozen = dict(line.split()[::-1] for line in (exp_dir / "PREREG_HASHES.txt").read_text().splitlines()[1:])
    hp, hc = sha256(prereg), sha256(cfg_path)
    match = frozen.get("PREREGISTRATION.md") == hp and frozen.get("config.json") == hc
    if args.confirm:
        if not match and not args.accept_deviations:
            print("refusing: pre-registration or config changed since freeze; document the change under "
                  "Deviations and pass --accept-deviations", file=sys.stderr)
            return 2
        seed, scale, only, out = cfg["seeds"]["confirmatory"], None, None, exp_dir / "results"
        if (out / "results.json").exists():
            print(f"refusing: {out/'results.json'} exists — the confirmatory run happens once. "
                  "Move it aside deliberately if a documented bug forces a rerun.", file=sys.stderr)
            return 2
    else:
        seed = args.seed if args.seed is not None else cfg["seeds"]["dev"][0]
        if seed not in cfg["seeds"]["dev"]:
            print("dev runs must use a dev seed", file=sys.stderr)
            return 2
        scale = {"n_train_users": 60, "n_test_users": 40, "n_interactions": 200, "bootstrap": 400,
                 "b2_n_configs": 40, "b2_n_users": 15, "b2_n_tasks": 20}
        only, out = args.only, exp_dir / "runs" / f"dev-seed{seed}"
    t0 = time.time()
    meta = {"experiment": args.exp, "mode": "confirmatory" if args.confirm else f"dev (seed {seed}, reduced scale)",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": seed,
            "python": platform.python_version(), "numpy": np.__version__, "prereg_sha256": hp,
            "config_sha256": hc, "hashes_match_freeze": match, "scale": scale}
    print(f"G0 {meta['mode']}: {len(only or cfg['conditions'])} conditions, {args.workers} workers", flush=True)
    results = run_all(cfg, seed, scale, args.workers, only)
    gates = evaluate_gates(results) if set(results) >= {"main-u1-binary", "negctrl", "main-u1-sparse",
                                                        "main-u0-binary", "main-u1-dense"} else []
    meta["wall_seconds"] = time.time() - t0
    path = write_report(results, gates, meta, out)
    print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
