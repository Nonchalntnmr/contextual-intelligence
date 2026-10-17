"""Pre-run world check for D1: similarity/relevance statistics only. No policy is run."""
import json, numpy as np
from pcim.observatory.sim import World, _unit, _sigmoid
cfg = json.load(open("experiments/g0/config.json"))
w = cfg["world"]
rng = np.random.default_rng(0)  # dev seed
def stats(noise_fn, label):
    world = World.create(w, 1.0, 1.0, rng)
    same, cross, gate_same, gate_cross = [], [], [], []
    for _ in range(40):
        topics = _unit(rng.normal(size=(w["n_topics"], w["dim"])))
        for _ in range(30):
            a, b = rng.integers(0, w["n_topics"], 2)
            ta = noise_fn(topics[a]); ib = noise_fn(topics[a]); ic = noise_fn(topics[b]) if b != a else None
            same.append(ta @ ib)
            if ic is not None: cross.append(ta @ ic)
    same, cross = np.array(same), np.array(cross)
    g = lambda s: _sigmoid((s - w["relevance_center"]) / w["relevance_width"])
    print(f"{label:<44} same-topic cos {same.mean():.2f}±{same.std():.2f}  gate {g(same).mean():.2f}  | "
          f"cross-topic cos {cross.mean():+.2f}±{cross.std():.2f}  gate {g(cross).mean():.2f}")
d, tn = w["dim"], w["topic_noise"]
stats(lambda c: _unit(c + rng.normal(0, tn / np.sqrt(d), d)), "D1 chosen: RMS noise norm = topic_noise")
stats(lambda c: _unit(c + rng.normal(0, tn, d)), "rejected: per-coordinate SD = topic_noise")
