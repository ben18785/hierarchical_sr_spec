"""Verify the transcribed formulas of hsr/data/sources/curve_libraries.csv
against the R evaluations (curvelib_eval.R) and build a corpus file in the
format of hsr/data/corpus.json from the verified ones.

    python curvelib_build.py <csv> <r_eval.csv> <out.json>
"""
import csv
import json
import sys

import numpy as np
import sympy as sp

from hsr.canonical import canonicalise
from hsr.grammar_compare import generalise


def _pow_of_x(e, x):
    """exp(A*log(x) + R) -> x**A * exp(R) (x > 0), for A free of x."""
    arg = sp.expand(sp.expand_log(e.args[0], force=True))
    A = arg.coeff(sp.log(x))
    if A == 0 or A.has(x):
        return e
    return x**A * sp.exp(sp.expand(arg - A*sp.log(x)))


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="|"))


def main(src, r_eval, out):
    rows = load(src)
    with open(r_eval) as fh:
        ev = list(csv.DictReader(fh))
    with open(r_eval + ".versions") as fh:
        versions = fh.read().strip()
    x = sp.Symbol("x", positive=True)
    per_model, verification, freq = {}, {}, {}
    for r in rows:
        pn = r["params"].split(",")
        syms = [sp.Symbol(p, real=True) for p in pn]
        expr = sp.sympify(r["formula"], locals={**{p: s for p, s in zip(pn, syms)}, "x": x})
        f = sp.lambdify([x] + syms, expr, modules="numpy")
        errs, n_ok = [], 0
        for e in [e for e in ev if e["func"] == r["function"]]:
            yv = float(e["y"]) if e["y"] not in ("NA", "") else np.nan
            if not np.isfinite(yv) or yv == 0.0:
                continue        # R implementations set values outside the domain to 0 or NA
            with np.errstate(all="ignore"):
                yc = complex(f(float(e["x"]), *[float(v) for v in e["params"].split(",")]))
            yp = yc.real if abs(yc.imag) < 1e-12 else np.nan
            if np.isfinite(yp):
                n_ok += 1
                errs.append(abs(yp - yv) / max(1.0, abs(yv)))
        ok = n_ok >= 5 and max(errs) < 1e-6
        # x > 0 in these libraries: exp(b*log(x)) -> x**b, so that log-logistic and
        # Weibull forms get the same keys as the corresponding power forms
        expr = expr.replace(lambda e: isinstance(e, sp.exp) and e.has(sp.log(x)), lambda e: _pow_of_x(e, x))
        key, note = None, ""
        if ok:
            reps = {s: sp.Symbol(f"c{i}", real=True) for i, s in enumerate(syms)}
            e2 = expr.xreplace({x: sp.Symbol("x1", real=True), **reps})
            try:
                sk = canonicalise(str(e2), 1, max_consts=None)
                key = None if sk is None else generalise(sk.key)
            except Exception as exc:
                note = f"not expressible in the grammar: {type(exc).__name__}"
            if key is None and not note:
                note = "not expressible in the grammar"
        verification[f"{r['package']}::{r['function']}"] = {
            "verified": bool(ok), "n_points": n_ok, "max_rel_err": float(max(errs)) if errs else None,
            "key": key, "note": note, "formula": r["formula"]}
        if ok and key:
            per_model[f"{r['package']}::{r['function']}"] = [key]
            freq[key] = freq.get(key, 0) + 1
    res = {"source": "R curve-fitting libraries (stats self-start models, nlraa, drc); formulas transcribed "
                     "and verified numerically against the packages",
           "versions": versions, "per_model": per_model, "freq": freq, "verification": verification,
           "stats": {"functions": len(rows), "verified": sum(v["verified"] for v in verification.values()),
                     "in_corpus": len(per_model)}}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(res["stats"])
    for k, v in verification.items():
        if not (v["verified"] and v["key"]):
            print("  excluded", k, v["n_points"], v["max_rel_err"], v["note"])


if __name__ == "__main__":
    main(*sys.argv[1:])
