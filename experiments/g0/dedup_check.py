"""D2 calibration: cosine between distinct same-topic items vs true near-duplicates.
World statistics only (dev seed 0). No policy is run."""
import json
import numpy as np
from pcim.observatory.sim import World

cfg = json.load(open("experiments/g0/config.json"))
rng = np.random.default_rng(0)
world = World.create(cfg["world"], 1.0, 1.0, rng)
distinct, dups, stale = [], [], []
for _ in range(60):
    u = world.new_user(rng)
    E = u.emb
    C = E @ E.T
    n = len(E)
    for i in range(n):
        for j in range(i + 1, n):
            if u.group[i] == u.group[j]:
                dups.append(C[i, j])
            elif u.superseded[i] or u.superseded[j]:
                stale.append(C[i, j])
            else:
                distinct.append(C[i, j])
distinct, dups = np.array(distinct), np.array(dups)
same_topic = distinct[distinct > 0.6]
print(f"distinct same-topic pairs: n={len(same_topic)} mean {same_topic.mean():.3f} "
      f"p99 {np.quantile(same_topic, .99):.4f} p99.9 {np.quantile(same_topic, .999):.4f} max {same_topic.max():.4f}")
print(f"true duplicate pairs:      n={len(dups)} min {dups.min():.4f} p1 {np.quantile(dups, .01):.4f}")
print(f"P(distinct same-topic pair > 0.92) = {(same_topic > 0.92).mean():.3f}")
