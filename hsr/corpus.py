"""A corpus of published model equations, used to inform the grammar prior.

Sources
  * BioModels (manually curated SBML models): every reaction's kinetic law.
  * A short hand-written list of standard ecological, pharmacological and
    growth/decay forms (CURATED below).

Each law is reduced to one-variable skeletons, one per species it involves:
that species becomes x1 and every other symbol (parameters, compartments,
other species) becomes a constant. The result is canonicalised with
``hsr.canonical.canonicalise``, so a corpus entry is a canonical key, exactly
like a candidate. Within a model each distinct key is counted once, so a
model with twenty Michaelis-Menten reactions contributes Michaelis-Menten
once.

The corpus is stored as key frequencies (``hsr/data/corpus.json``); rule
counts for any grammar are derived from the keys, so a context-dependent
grammar can be fitted to the same file later.

Build (needs network and python-libsbml) on GitHub Actions:
    gh workflow run benchmark.yml --ref <branch> -f task=corpus -f first=1 -f last=1100
which runs ``python -m hsr.corpus build`` and uploads corpus.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter

import numpy as np
import sympy as sp

from .canonical import UnsupportedExpression, canonicalise, counts_array, make_vars
from .grammar import Grammar

BIOMODELS_URL = "https://www.ebi.ac.uk/biomodels/model/download/{id}?filename={id}_url.xml"

# Standard forms outside BioModels' reaction kinetics (x = the variable,
# letters other than x are constants). Includes the benchmark families'
# textbook forms; the held-out corpus drops them (``held_out``).
CURATED = [
    # growth
    "r*x*(1 - x/K)", "K/(1 + a*exp(-r*x))", "K*exp(-b*exp(-c*x))", "a*(1 - exp(-k*x))",
    "Linf*(1 - exp(-k*(x - t0)))", "a*x**b", "a*x**b*exp(-c*x)", "a*exp(r*x)",
    # functional responses and stock-recruitment
    "a*x", "a*x/(1 + a*h*x)", "a*x**2/(1 + a*h*x**2)", "a*x/(1 + b*x)", "a*x*exp(-b*x)",
    "a*x/(b + x)", "a*x**n/(b**n + x**n)",
    # pharmacology
    "E0 + Emax*x/(EC50 + x)", "E0 + Emax*x**n/(EC50**n + x**n)", "a*exp(-k*x)",
    "a*exp(-k1*x) + b*exp(-k2*x)", "1/(1 + (x/c)**n)",
    # survival and dose-response
    "exp(-(x/l)**k)", "exp(-l*x)", "1/(1 + exp(-(a + b*x)))", "x**k/(x**k + c**k)",
    "a*exp(-(x/l)**k)",
    # temperature and allometry
    "a*exp(-E/x)", "a*q**(x/10)", "a*x**(3/4)",
    # oscillations and peaks
    "a*sin(w*x + p)", "a*exp(-d*x)*sin(w*x + p)", "a*exp(-(x - m)**2/s)", "a*log(x) + b",
]

_FUNCS = {"exp": "exp", "ln": "log", "log": "log", "sqrt": "sqrt", "sin": "sin", "cos": "cos",
          "tanh": "tanh", "pow": "Pow", "abs": None, "piecewise": None, "delay": None,
          "floor": None, "ceil": None, "factorial": None, "log10": "_log10"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ---------------------------------------------------------------- reduction
def _to_sympy(formula: str):
    """Parse an SBML L3 infix formula; every identifier becomes a symbol
    named s<k>. Returns (expr, {original name: symbol}) or None."""
    names = {}
    out = []
    pos = 0
    f = formula.replace("^", "**")
    for m in _IDENT.finditer(f):
        name = m.group(0)
        out.append(f[pos:m.start()])
        rest = f[m.end():].lstrip()
        if rest.startswith("("):
            fn = _FUNCS.get(name, None) if name in _FUNCS else None
            if fn is None:
                return None
            out.append(fn)
        elif name in ("true", "false", "time", "avogadro", "pi", "exponentiale", "INF", "NaN"):
            if name == "pi":
                out.append("pi")
            elif name == "exponentiale":
                out.append("E")
            else:
                return None
        else:
            sym = names.setdefault(name, f"s{len(names)}")
            out.append(sym)
        pos = m.end()
    out.append(f[pos:])
    text = "".join(out)
    local = {v: sp.Symbol(v, positive=True) for v in names.values()}
    local["_log10"] = lambda a: sp.log(a) / sp.log(10)
    local["Pow"] = sp.Pow
    try:
        expr = sp.sympify(text, locals=local)
    except Exception:
        return None
    return expr, {k: local[v] for k, v in names.items()}


def collapse_constants(expr, x, prefix="k"):
    """Replace every maximal x-free part of ``expr`` that is not a plain number
    by one fresh constant: within a sum or product, all x-free terms merge into
    a single constant (c1*c2*x -> k0*x, c0/c3 + x/c4 -> k0 + k1*x)."""
    memo = {}

    def new(part):
        # identical constant parts share one constant (a Hill exponent n used twice stays shared)
        if part not in memo:
            memo[part] = sp.Symbol(f"{prefix}{len(memo)}", real=True)
        return memo[part]

    def rec(e):
        if not e.has(x):
            return e if e.is_number else new(e)
        if e.is_Add or e.is_Mul:
            free = [a for a in e.args if not a.has(x)]
            dep = [rec(a) for a in e.args if a.has(x)]
            if free and not all(a.is_number for a in free):
                dep.append(new(e.func(*free)))
            else:
                dep += free
            return e.func(*dep)
        if e.is_Pow and e.base.is_Mul and not e.exp.has(x):
            # (k*f)**p -> k'*f**p
            free = [a for a in e.base.args if not a.has(x)]
            if free and not all(a.is_number for a in free):
                dep = sp.Mul(*[a for a in e.base.args if a.has(x)])
                return new(sp.Pow(sp.Mul(*free), e.exp)) * rec(sp.Pow(dep, e.exp))
        if not e.args:
            return e
        return e.func(*[rec(a) for a in e.args])

    return rec(expr)


def project(expr, variables, sym_map):
    """One-variable canonical keys of ``expr``: for each name in ``variables``
    that occurs in it, that symbol -> x1 and every other symbol -> a constant."""
    x1 = make_vars(1)[0]
    present = [n for n in variables if n in sym_map and expr.has(sym_map[n])]
    keys = set()
    for name in present:
        sub = {}
        k = 0
        for other, s in sym_map.items():
            if other == name:
                sub[s] = x1
            elif expr.has(s):
                sub[s] = sp.Symbol(f"c{k}", real=True)
                k += 1
        try:
            e = collapse_constants(expr.xreplace(sub), x1)
            e = e.xreplace({s: sp.Symbol(f"c{i}", real=True)
                            for i, s in enumerate(sorted(e.free_symbols - {x1}, key=str))})
            sk = canonicalise(e, 1, max_consts=None)
        except Exception:
            sk = None
        if sk is not None:
            keys.add(sk.key)
    return keys


def curated_keys():
    out = {}
    for s in CURATED:
        parsed = _to_sympy(s)
        if parsed is None:
            continue
        expr, syms = parsed
        ks = project(expr, ["x"], syms)
        out[s] = sorted(ks)
    return out


def sbml_laws(path):
    """(species ids, [kinetic law formula strings]) of one SBML file."""
    import libsbml
    doc = libsbml.readSBMLFromFile(path)
    if doc.getModel() is None:
        return [], []
    props = libsbml.ConversionProperties()
    props.addOption("expandFunctionDefinitions", True)
    doc.convert(props)
    m = doc.getModel()
    species = [s.getId() for s in m.getListOfSpecies()]
    laws = []
    for r in m.getListOfReactions():
        kl = r.getKineticLaw()
        if kl is not None and kl.getMath() is not None:
            laws.append(libsbml.formulaToL3String(kl.getMath()))
    return species, laws


def model_keys(path):
    species, laws = sbml_laws(path)
    keys = set()
    n_ok = 0
    for f in laws:
        parsed = _to_sympy(f)
        if parsed is None:
            continue
        expr, syms = parsed
        ks = project(expr, species, syms)
        if ks:
            n_ok += 1
        keys |= ks
    return keys, len(laws), n_ok


# ------------------------------------------------------------------- corpus
def _fetch(url, tries=3):
    """GET url -> (bytes, None) or (None, error string). A 404 is not retried."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "hsr-corpus/0.1 (python urllib)",
                                               "Accept": "application/xml,*/*"})
    err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read(), None
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.reason}"
            if e.code == 404:
                break
        except Exception as e:  # network, TLS, timeout
            err = repr(e)
        time.sleep(2 * (k + 1))
    return None, err


def build(out, first, last, cache_dir, sleep=0.2):
    os.makedirs(cache_dir, exist_ok=True)
    freq = Counter()
    per_model = {}
    stats = Counter()
    for i in range(first, last + 1):
        mid = f"BIOMD{i:010d}"
        path = os.path.join(cache_dir, mid + ".xml")
        if not os.path.exists(path):
            data, err = _fetch(BIOMODELS_URL.format(id=mid))
            if data is None:
                stats["missing"] += 1
                if stats["missing"] <= 5:
                    print(f"{mid}: download failed: {err}", flush=True)
                if stats["missing"] >= 25 and stats["models"] == 0:
                    raise RuntimeError(f"first {stats['missing']} downloads all failed; last error: {err}")
                continue
            with open(path, "wb") as fh:
                fh.write(data)
            time.sleep(sleep)
        try:
            keys, n_laws, n_ok = model_keys(path)
        except Exception:
            stats["unreadable"] += 1
            continue
        stats["models"] += 1
        stats["laws"] += n_laws
        stats["laws_reduced"] += n_ok
        per_model[mid] = sorted(keys)
        freq.update(keys)
        if i % 50 == 0:
            print(f"{mid}: {stats['models']} models, {len(freq)} distinct keys", flush=True)
    cur = curated_keys()
    for ks in cur.values():
        freq.update(ks)
    data = {"source": "BioModels manually curated (kinetic laws) + hsr.corpus.CURATED",
            "range": [first, last], "stats": dict(stats), "freq": dict(freq.most_common()),
            "curated": cur, "per_model": per_model}
    with open(out, "w") as fh:
        json.dump(data, fh, indent=0)
    print(f"{stats['models']} models, {stats['laws']} laws ({stats['laws_reduced']} reduced), "
          f"{len(freq)} distinct keys -> {out}")


def load(path, exclude=()):
    """{key: frequency}, minus any key in ``exclude`` (held-out forms)."""
    with open(path) as fh:
        freq = json.load(fh)["freq"]
    ex = set(exclude)
    return {k: v for k, v in freq.items() if k not in ex}


def rule_totals(freq, grammar: Grammar):
    """Frequency-weighted rule counts of the corpus keys under ``grammar``
    (keys the grammar cannot express are skipped). Returns (counts, n_used)."""
    from .candidates import skeleton_from_key
    tot = np.zeros(grammar.n_rules)
    used = 0
    for k, w in freq.items():
        try:
            tot += w * counts_array(skeleton_from_key(k, grammar.n_vars).expr, grammar)
            used += w
        except (UnsupportedExpression, Exception):
            continue
    return tot, used


def corpus_alpha(freq, grammar: Grammar, kappa=50.0, floor=0.1):
    """Dirichlet parameters alpha_A = kappa * p_A + floor, with p_A the corpus
    rule frequencies within nonterminal A (kappa = prior sample size)."""
    tot, _ = rule_totals(freq, grammar)
    alpha = {}
    for nt in grammar.nonterminals:
        c = tot[grammar.slices[nt]]
        p = c / c.sum() if c.sum() > 0 else np.full(len(c), 1.0 / len(c))
        alpha[nt] = kappa * p + floor
    return alpha


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="hsr/data/corpus.json")
    b.add_argument("--first", type=int, default=1)
    b.add_argument("--last", type=int, default=1100)
    b.add_argument("--cache", default="corpus_cache")
    s = sub.add_parser("summary")
    s.add_argument("path")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args.out, args.first, args.last, args.cache)
    else:
        freq = load(args.path)
        g = Grammar()
        tot, used = rule_totals(freq, g)
        print(f"{len(freq)} keys, weight {sum(freq.values())}, used {used}")
        for nt in g.nonterminals:
            c = tot[g.slices[nt]]
            print(nt, dict(zip(g.options[nt], np.round(c / max(c.sum(), 1), 3))))
        print("top keys:", list(freq.items())[:25])


if __name__ == "__main__":
    main()
