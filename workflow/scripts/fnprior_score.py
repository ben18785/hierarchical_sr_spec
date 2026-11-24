"""Score every support form under fnprior (Bartlett & Desmond's n-gram prior),
trained separately on each fold's training equations. Runs in the fnprior
environment (workflow/envs/fnprior.yaml), so it does not import hsr.

Input  folds.json: {"support": [keys], "folds": [{"train": [keys], "test": [keys]}, ...]}
Output JSON: {"settings": ..., "logprior": {variant: [[logp per support key] per fold]}}

Keys use x1 for the variable and c0, c1, ... for constants; fnprior wants
x0 and a0, a1, ... (with code_constants=True, constants are coded "a", so a
power law x**a and an exponential a**x get different priors).
"""
import csv
import json
import os
import re
import sys
import tempfile

import numpy as np
from fnprior.prior import FunctionPrior

BASIS = [["a", "x"], ["sqrt", "exp", "log", "sin", "tanh", "inv"], ["+", "-", "*", "/", "pow"]]


def to_fn(key):
    s = re.sub(r"\bx1\b", "x0", key)
    return re.sub(r"\bc(\d+)\b", r"a\1", s)


def train(keys, tune, n, tmpdir, tag):
    path = os.path.join(tmpdir, f"train_{tag}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Filename", "Number", "Old Formula", "New Formula"])
        for i, k in enumerate(keys):
            w.writerow([f"f{i}", i, k, to_fn(k)])
    kp = FunctionPrior(n, BASIS, None, path, code_constants=True)
    if tune:
        kp.tune_discounts()
    return kp


def score(kp, keys):
    out = []
    for k in keys:
        try:
            v = float(kp.logprior(to_fn(k)))
        except Exception:
            v = float("-inf")
        out.append(v if np.isfinite(v) else None)
    return out


def main(folds_path, out_path, n=2):
    with open(folds_path) as fh:
        F = json.load(fh)
    support = F["support"]
    res = {"settings": {"n": n, "basis": BASIS, "code_constants": True, "smoothing": "kn (default)"},
           "logprior": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for tune in (False, True):
            name = "fnprior_kn_tuned" if tune else "fnprior_kn"
            res["logprior"][name] = []
            for f, fold in enumerate(F["folds"]):
                kp = train(fold["train"], tune, n, tmp, f"{f}_{tune}")
                res["logprior"][name].append(score(kp, support))
                print(name, "fold", f, flush=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
