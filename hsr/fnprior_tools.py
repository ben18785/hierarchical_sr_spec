"""Thin wrapper around fnprior (Bartlett & Desmond's n-gram prior on
functions): train on a list of skeleton keys, score keys. Imports only
fnprior, numpy and the standard library, so it also runs in the workflow's
separate fnprior environment."""
import csv
import os
import re

import numpy as np

BASIS = [["a", "x"], ["sqrt", "exp", "log", "sin", "tanh", "inv"], ["+", "-", "*", "/", "pow"]]


def to_fn(key):
    """hsr key (x1, c0, c1, ...) -> fnprior string (x0, a0, a1, ...)."""
    s = re.sub(r"\bx1\b", "x0", key)
    return re.sub(r"\bc(\d+)\b", r"a\1", s)


def train(keys, tune, n, tmpdir, tag):
    """FunctionPrior (Kneser-Ney, root context, constants coded 'a') trained on keys."""
    from fnprior.prior import FunctionPrior
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
    """log prior of each key (None where fnprior cannot score it)."""
    out = []
    for k in keys:
        try:
            v = float(kp.logprior(to_fn(k)))
        except Exception:
            v = float("-inf")
        out.append(v if np.isfinite(v) else None)
    return out
