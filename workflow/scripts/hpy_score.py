"""Score every support form under a hierarchical Pitman-Yor model (hsr.hpy)
trained on each fold's training equations, using fnprior's tree events
(first children: parent and grandparent; second children: parent and left
sibling). Runs in the fnprior environment. Same input/output format as
fnprior_score.py.

    python hpy_score.py folds.json out.json [n_sweeps] [n_keep]
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr.fnprior_tools import to_fn, train  # noqa: E402
from hsr.hpy import HPY  # noqa: E402


def events(kp, keys):
    out = []
    for k in keys:
        try:
            t = kp.coder.process_all_equations(kp.n + 1, [to_fn(k)], kp.maxvar)
            first, second = kp.split_events(t)
            out.append(([(tuple(e[-3:-1]), e[-1]) for e in first], [(tuple(e[-3:-1]), e[-1]) for e in second]))
        except Exception:
            out.append(None)
    return out


def main(folds_path, out_path, n_sweeps=20, n_keep=10):
    with open(folds_path) as fh:
        F = json.load(fh)
    support = F["support"]
    res = {"settings": {"model": "hierarchical Pitman-Yor (Teh 2006) on fnprior tree events, context 2",
                        "n_sweeps": n_sweeps, "n_keep": n_keep}, "logprior": {"hpy": []}, "hyper": []}
    with tempfile.TemporaryDirectory() as tmp:
        for f, fold in enumerate(F["folds"]):
            kp = train(fold["train"], False, 2, tmp, f"hpy{f}")     # used only for its tree coder
            tr = [e for e in events(kp, fold["train"]) if e is not None]
            sup = events(kp, support)
            vl = {e[1] for ev in tr + [s for s in sup if s] for e in ev[0]}
            vr = {e[1] for ev in tr + [s for s in sup if s] for e in ev[1]}
            left = HPY(vl, seed=f).fit([e for ev in tr for e in ev[0]], n_sweeps, n_keep)
            right = HPY(vr, seed=100 + f).fit([e for ev in tr for e in ev[1]], n_sweeps, n_keep)
            cache = {}

            def p(model, key, u, w):
                if (key, u, w) not in cache:
                    cache[(key, u, w)] = model.predictive(u, w)
                return cache[(key, u, w)]

            lp = []
            for ev in sup:
                if ev is None:
                    lp.append(None)
                    continue
                v = sum(np.log(p(left, "l", u, w)) for u, w in ev[0]) + sum(np.log(p(right, "r", u, w)) for u, w in ev[1])
                lp.append(float(v) if np.isfinite(v) else None)
            res["logprior"]["hpy"].append(lp)
            res["hyper"].append({"left": {"d": left.d.tolist(), "theta": left.theta.tolist()},
                                 "right": {"d": right.d.tolist(), "theta": right.theta.tolist()}})
            print("hpy fold", f, res["hyper"][-1], flush=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *[int(a) for a in sys.argv[3:]])
