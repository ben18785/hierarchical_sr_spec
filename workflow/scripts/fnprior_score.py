"""Score every support form under fnprior (Bartlett & Desmond's n-gram prior),
trained separately on each fold's training equations. Runs in the fnprior
environment (workflow/envs/fnprior.yaml); it imports only hsr.fnprior_tools (fnprior, numpy).

Input  folds.json: {"support": [keys], "folds": [{"train": [keys], "test": [keys]}, ...]}
Output JSON: {"settings": ..., "logprior": {variant: [[logp per support key] per fold]}}

Keys use x1 for the variable and c0, c1, ... for constants; fnprior wants
x0 and a0, a1, ... (with code_constants=True, constants are coded "a", so a
power law x**a and an exponential a**x get different priors).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr.fnprior_tools import BASIS, score, train  # noqa: E402  (imports only fnprior/numpy)


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
