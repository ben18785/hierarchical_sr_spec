"""Code every support form of every comparison run as an expression tree in
fnprior's representation (hsr.treecode), and check that the tree's events
are exactly fnprior's own (SymbolCoder + split_events) at n = 2 and n = 3.
Runs in the fnprior environment.

    python hpyv_trees.py out.json folds.json [folds.json ...]

Output: {"trees": {key: prefix symbol list or null}, "check": {...}}.
"""
import json
import os
import sys
import tempfile
from collections import Counter

import sympy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr import treecode as tc  # noqa: E402
from hsr.fnprior_tools import to_fn, train  # noqa: E402


def main(out, *folds):
    keys = set()
    for p in folds:
        with open(p) as fh:
            F = json.load(fh)
        keys |= set(F["support"])
        for f in F["folds"]:
            keys |= set(f["train"]) | set(f["test"])
    keys = sorted(keys)
    with tempfile.TemporaryDirectory() as tmp:
        kps = {n: train(["c0*x1/(c1 + x1)"], False, n, tmp, f"coder{n}") for n in (2, 3)}
    inv = {n: {v: k for k, v in kp.coder.code.items()} for n, kp in kps.items()}
    nsym = 40
    x = sympy.symbols([f"x{i}" for i in range(nsym)], real=True)
    a = sympy.symbols([f"a{i}" for i in range(nsym)], real=True)
    locs = {**{f"x{i}": x[i] for i in range(nsym)}, **{f"a{i}": a[i] for i in range(nsym)}}
    trees, fail, mismatch = {}, Counter(), []
    for k in keys:
        try:
            t = tc.from_labels(kps[2].coder.equation2labels(to_fn(k), locs))
        except Exception as e:
            trees[k] = None
            fail[type(e).__name__] += 1
            continue
        trees[k] = tc.to_prefix(t)
        for n, kp in kps.items():
            tup = kp.coder.process_all_equations(n + 1, [to_fn(k)], kp.maxvar)
            left, right = kp.split_events(tup)
            left = [tuple(inv[n][c] for c in e) for e in left]
            right = [tuple(inv[n][c] for c in e[-(n + 1):]) for e in right]
            mine = tc.fnprior_events(t, n)
            if Counter(left) != Counter(mine[0]) or Counter(right) != Counter(mine[1]):
                mismatch.append((k, n))
    by_tree = Counter(json.dumps(v) for v in trees.values() if v is not None)
    check = {"n_keys": len(keys), "n_coded": sum(v is not None for v in trees.values()), "uncoded": dict(fail),
             "event_mismatches_vs_fnprior": len(mismatch), "mismatch_examples": mismatch[:20],
             "n_distinct_trees": len(by_tree),
             "n_keys_sharing_a_tree": sum(c for c in by_tree.values() if c > 1),
             "largest_tree_classes": [[json.loads(t), c] for t, c in by_tree.most_common(5)]}
    print(json.dumps(check, indent=1))
    with open(out, "w") as fh:
        json.dump({"trees": trees, "check": check}, fh)


if __name__ == "__main__":
    main(*sys.argv[1:])
