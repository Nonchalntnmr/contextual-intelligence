"""Post-hoc diagnostic of the confirmatory G0-2 failure. Exploratory; does not touch results/.

Replays the confirmatory negctrl condition (same seed, first 60 test users) and asks what
D and B2 actually select: how many items, how long, how relevant. Then checks whether
ranking by value *density* (score per token, the knapsack shadow-price fix) closes the gap.
"""
import json
import numpy as np
from pcim.observatory.g0 import G0_DIR, collect_logs, tune_b2, _rng
from pcim.observatory.sim import World, oracle_select
from pcim.policy.features import featurize
from pcim.policy.heuristic import HeuristicPolicy
from pcim.policy.learned import BayesLinearPolicy, fit_population_prior
from pcim.policy.select import greedy_select

cfg = json.load(open(G0_DIR / "config.json"))
cond = next(c for c in cfg["conditions"] if c["name"] == "negctrl")
seed = cfg["seeds"]["confirmatory"]
ss = np.random.SeedSequence(seed)
s_world, s_tune, s_train, s_train2, s_test, s_pol = ss.spawn(6)
world = World.create(cfg["world"], cond["lambda_shared"], cond["lambda_user"], _rng(s_world), similarity_world=True)
model, budget, pc = cfg["models"]["M1"], cfg["budget_tokens"], cfg["policy"]
b2p, _ = tune_b2(world, cfg, model, _rng(s_tune))
b2 = HeuristicPolicy(b2p)
logs = collect_logs(world, cfg, model, b2, cond["feedback"], _rng(s_train), cfg["protocol"]["n_train_users"], 200)
prior = fit_population_prior(logs, rank=pc["prior_rank"], iters=pc["em_iters"])
D = BayesLinearPolicy(prior, frozen=True, dedup_cos=pc["dedup_cos"])
print("B2 tuned:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b2p.__dict__.items() if k != "kind_bonus"})

def density_select(pol, cs, F):
    s = pol._item_scores(pol.mean(), F, cs)
    return greedy_select(s / cs.tokens, cs.tokens, budget, emb=cs.emb, threshold=0.0, dedup_cos=pol.dedup_cos,
                         eligible=~cs.superseded)

rng = _rng(s_test)
stats = {k: {"q": [], "n": [], "tok_per_item": [], "relevant_frac": []} for k in ("B2", "D", "D-density", "O")}
for ui in range(60):
    u = world.new_user(rng)
    for t in range(200):
        task = u.new_task(rng)
        idx, cs = u.candidates(task, cfg["candidates"])
        util = u.utilities(task, model)
        F = featurize(cs, task.probs)
        rng.random(); rng.random(); rng.standard_normal()  # keep the stream aligned with g0
        g = u.relevance(task)
        sels = {"B2": b2.select(cs, task.probs, budget).idx, "D": D.select(cs, task.probs, budget, F=F).idx,
                "D-density": density_select(D, cs, F), "O": oracle_select(u, task, idx, util, model, budget)}
        for k, sel in sels.items():
            stats[k]["q"].append(u.quality(task, idx[sel], util, model))
            stats[k]["n"].append(len(sel))
            if len(sel):
                stats[k]["tok_per_item"].append(cs.tokens[sel].mean())
                stats[k]["relevant_frac"].append((g[idx[sel]] > 0.5).mean())
print(f"{'policy':<10} {'mean Q':>7} {'items':>6} {'tokens/item':>12} {'relevant':>9}")
for k, s in stats.items():
    print(f"{k:<10} {np.mean(s['q']):7.3f} {np.mean(s['n']):6.2f} {np.mean(s['tok_per_item']):12.1f} {np.mean(s['relevant_frac']):9.2f}")
