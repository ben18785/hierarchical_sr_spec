"""Compare grammars as priors over equation forms, using the corpus of
published equations (hsr/data/corpus.json) before any target data is seen.

Grammars ("schemes") differ in what each rule choice may depend on:

  cf         context-free: the current PCFG (hsr.grammar)
  parent     every choice depends on the nearest enclosing function
             (root, exp, log, sin, sqrt, tanh, or the base of a power)
  position   as parent, and a factor's choice (T rules) also depends on
             whether it sits in a numerator or a denominator
  term       as position, and also whether the product is one term of a
             sum of several terms (e.g. the x in c + x)
  term_root  as term, and also whether the choice is made at the top level
             of the expression (not inside brackets or a function): a
             start-of-tree context, as in fnprior's root context

Each scheme is a top-down generative process in which every choice is made
from a distribution indexed by a context that depends only on the
ancestors, so it is a proper prior over derivations. With independent
Dirichlet priors per (context, nonterminal), the probability of a set of
derivations is a product of Dirichlet-multinomial terms (closed form).

Two scores:
  evidence  log p(corpus | scheme), Dirichlet(alpha) on every (context,
            nonterminal), alpha the default Grammar prior
  cv        K-fold cross-validation by BioModels model: posterior-mean rule
            probabilities from the training models, mean log probability
            per held-out form; optional back-off of each context's prior
            to the context-free fit: alpha_ctx = kappa * p_cf + alpha0

The unit is each model's distinct forms (a model counts each form once).
Forms are scored by their canonical derivation (hsr.canonical.rule_counts
conventions), the same for every scheme.

    python -m hsr.grammar_compare            # print the comparison table
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import sympy as sp
from scipy.special import gammaln

from .canonical import _SqrtItem, _is_negative_term, _split_factors, is_const
from .grammar import Grammar

CORPUS = os.path.join(os.path.dirname(__file__), "data", "corpus.json")
SCHEMES = ("cf", "parent", "position", "term", "term_root")
FUNCS = {sp.exp: "exp", sp.log: "log", sp.sin: "sin", sp.tanh: "tanh"}


# --------------------------------------------------------------- derivations
class _Counter:
    """Walks the canonical derivation of rule_counts, recording each choice
    with its context: (fn, pos), fn = nearest enclosing function, pos =
    'num' / 'den' for factors (T choices) and '-' otherwise."""

    def __init__(self):
        self.c = Counter()

    def add(self, fn, pos, nt, opt, n=1):
        if n:
            self.c[(fn, pos + ("+" if self.insum and pos in ("num", "den") else ""), self.depth == 1, nt, opt)] += n

    insum = False
    depth = 0

    def E(self, expr, fn):
        self.depth += 1
        try:
            self._E(expr, fn)
        finally:
            self.depth -= 1

    def _E(self, expr, fn):
        if isinstance(expr, sp.Add):
            outer, self.insum = self.insum, True
            self._sum(expr, fn)
            self.insum = outer
            return
        self.add(fn, "-", "E", "F")
        outer, self.insum = self.insum, False
        self.F(expr, fn)
        self.insum = outer

    def _sum(self, expr, fn):
        terms = list(expr.as_ordered_terms())
        pos_t = [t for t in terms if not _is_negative_term(t)]
        neg = [t for t in terms if _is_negative_term(t)]
        self.add(fn, "-", "E", "F")
        if pos_t:
            self.F(pos_t[0], fn)
            rest = pos_t[1:]
        else:
            self.F(neg[0], fn)
            neg, rest = neg[1:], []
        for t in rest:
            self.add(fn, "-", "E", "E+F")
            self.F(t, fn)
        for t in neg:
            self.add(fn, "-", "E", "E-F")
            self.F(-t, fn)

    def F(self, expr, fn):
        num, den = _split_factors(expr)
        if not num:
            num = [sp.Symbol("c_one", real=True)]
        self.add(fn, "-", "F", "T")
        self.add(fn, "-", "F", "F*T", len(num) - 1)
        self.add(fn, "-", "F", "F/T", len(den))
        for item in num:
            self.T(item, fn, "num")
        for item in den:
            self.T(item, fn, "den")

    def R(self, name, fn, pos, arg, child_fn):
        self.add(fn, pos, "T", "R(E)")
        self.add(fn, pos, "R", name)
        self.E(arg, child_fn)

    def T(self, item, fn, pos):
        if isinstance(item, _SqrtItem):
            self.R("sqrt", fn, pos, item.base, "sqrt")
            return
        if isinstance(item, sp.Symbol):
            if is_const(item) or item.name.startswith("c_"):
                self.add(fn, pos, "T", "C")
            else:
                self.add(fn, pos, "T", "V")
                self.add(fn, pos, "V", item.name)
            return
        if item.is_Number:
            self.add(fn, pos, "T", "C")
            return
        for cls, name in FUNCS.items():
            if isinstance(item, cls):
                self.R(name, fn, pos, item.args[0], name)
                return
        if isinstance(item, sp.Pow):
            if item.exp.is_Rational and item.exp.q in (1, 2):
                self.add(fn, pos, "T", "(E)")
                self.E(item, fn)
                return
            if is_const(item.exp):
                self.add(fn, pos, "T", "(E)^C")
                self.E(item.base, "pow")
                return
            # b**p, p not a constant or half-integer: exp(p*log(b))
            self.add(fn, pos, "T", "R(E)")
            self.add(fn, pos, "R", "exp")
            self.add("exp", "-", "E", "F")
            self.add("exp", "-", "F", "T")
            self.add("exp", "-", "F", "F*T")
            ex = item.exp
            if is_const(ex) or ex.is_Number:
                self.add("exp", "num", "T", "C")
            else:
                self.add("exp", "num", "T", "(E)")
                self.E(ex, "exp")
            self.R("log", "exp", "num", item.base, "log")
            return
        if isinstance(item, (sp.Add, sp.Mul)):
            self.add(fn, pos, "T", "(E)")
            self.E(item, fn)
            return
        raise ValueError(f"cannot count {item!r}")


def context_counts(expr) -> Counter:
    """Counter over (fn, pos, nonterminal, option) for one sympy skeleton."""
    w = _Counter()
    w.E(expr, "root")
    return w.c


def project(counts: Counter, scheme: str) -> Counter:
    """Collapse full contexts to the scheme's context."""
    out = Counter()
    for (fn, pos, top, nt, opt), n in counts.items():
        if scheme == "cf":
            ctx = ()
        elif scheme == "parent":
            ctx = (fn,)
        elif scheme == "position":
            ctx = (fn, pos.rstrip("+") if nt in ("T", "R", "V") else "-")
        elif scheme == "term":
            ctx = (fn, pos if nt in ("T", "R", "V") else "-")
        elif scheme == "term_root":
            ctx = (fn, pos if nt in ("T", "R", "V") else "-", top)
        else:
            raise ValueError(scheme)
        out[(ctx, nt, opt)] += n
    return out


# ------------------------------------------------------------------ scoring
def _tables(items, scheme, grammar):
    """{(ctx, nt): count vector over grammar.options[nt]} summed over items."""
    tabs = {}
    for c in items:
        for (ctx, nt, opt), n in project(c, scheme).items():
            v = tabs.setdefault((ctx, nt), np.zeros(len(grammar.options[nt])))
            v[grammar.options[nt].index(opt)] += n
    return tabs


def log_evidence(items, scheme, grammar=None) -> float:
    """log p(derivations | scheme) with Dirichlet(alpha_nt) per (context, nt)."""
    g = grammar or Grammar()
    tot = 0.0
    for (ctx, nt), n in _tables(items, scheme, g).items():
        a = g.alpha[nt]
        tot += gammaln(a.sum()) - gammaln(a.sum() + n.sum()) + np.sum(gammaln(a + n) - gammaln(a))
    return float(tot)


def _log_prob(c, scheme, probs, fallback):
    lp = 0.0
    for (ctx, nt, opt), n in project(c, scheme).items():
        p = probs.get((ctx, nt), fallback[nt])
        lp += n * np.log(p[Grammar().options[nt].index(opt)])
    return lp


def cv_score(models, scheme, k=5, kappa=None, seed=0, grammar=None):
    """Mean held-out log probability per form, K-fold by model. models:
    list of lists of Counters (each model's distinct forms). kappa: back-off
    strength (None = plain Dirichlet(alpha) per context)."""
    g = grammar or Grammar()
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(models)) % k
    scores = []
    for f in range(k):
        train = [c for m, ff in zip(models, fold) if ff != f for c in m]
        test = [c for m, ff in zip(models, fold) if ff == f for c in m]
        cf = _tables(train, "cf", g)
        p_cf = {nt: (g.alpha[nt] + cf.get(((), nt), 0)) / (g.alpha[nt] + cf.get(((), nt), 0)).sum()
                for nt in g.nonterminals}
        probs = {}
        for (ctx, nt), n in _tables(train, scheme, g).items():
            a = g.alpha[nt] if kappa is None else kappa * p_cf[nt] + 0.1
            probs[(ctx, nt)] = (a + n) / (a + n).sum()
        # unseen contexts: the prior mean (or the context-free fit when backing off)
        fb = {nt: (g.alpha[nt] / g.alpha[nt].sum() if kappa is None else p_cf[nt]) for nt in g.nonterminals}
        scores += [_log_prob(c, scheme, probs, fb) for c in test]
    return float(np.mean(scores)), len(scores)


def prior_only_score(models, grammar=None):
    """Mean log probability per form under the fixed default prior mean (no learning)."""
    g = grammar or Grammar()
    fb = {nt: g.alpha[nt] / g.alpha[nt].sum() for nt in g.nonterminals}
    sc = [_log_prob(c, "cf", {}, fb) for m in models for c in m]
    return float(np.mean(sc))


def load_models(path=CORPUS):
    from .candidates import skeleton_from_key
    with open(path) as fh:
        d = json.load(fh)
    cache = {}
    models = []
    for mid, keys in d["per_model"].items():
        forms = []
        for key in keys:
            if key not in cache:
                try:
                    cache[key] = context_counts(skeleton_from_key(key).expr)
                except Exception:
                    cache[key] = None
            if cache[key] is not None:
                forms.append(cache[key])
        if forms:
            models.append(forms)
    return models, d


def compare(path=CORPUS, k=5, kappas=(1, 5, 20, 50, 200), seed=0):
    models, _ = load_models(path)
    items = [c for m in models for c in m]
    rows = [{"scheme": "prior mean (no learning)", "cv": prior_only_score(models)}]
    for s in SCHEMES:
        rows.append({"scheme": s, "evidence": log_evidence(items, s), "cv": cv_score(models, s, k, seed=seed)[0],
                     "n_contexts": len({ctx for (ctx, _) in _tables(items, s, Grammar())})})
        if s != "cf":
            for kap in kappas:
                rows.append({"scheme": f"{s}+backoff(kappa={kap})", "cv": cv_score(models, s, k, kappa=kap, seed=seed)[0]})
    return rows, len(models), len(items)


if __name__ == "__main__":
    rows, nm, ni = compare()
    print(f"{nm} models, {ni} model-form pairs")
    for r in rows:
        print(f"{r['scheme']:<32} evidence={r.get('evidence', float('nan')):10.1f}  "
              f"cv logp/form={r['cv']:8.3f}  contexts={r.get('n_contexts', '')}")


# ------------------------------------------------------ common-support scoring
# Different priors put mass on different spaces of trees, so raw log
# probabilities are not comparable. For a fair comparison every prior is
# renormalised over one common support: the candidate set plus every corpus
# form. Held-out forms are scored by log P(form) / sum_{s in support} P(s).

CANDIDATES = os.path.join(os.path.dirname(__file__), "data", "candidates.json")


def generalise(key):
    """Canonical key with every numeric literal replaced by a free constant,
    except the structural ones: integer and half-integer exponents and signs.
    Neither the grammar nor fnprior models the value of a literal (both code
    it as a constant), so without this step every prior would spread its mass
    over arbitrary numeric variants (2*x1, 0.693*x1, ...) of the same form,
    which make up about half of the candidate set."""
    from .candidates import skeleton_from_key
    from .canonical import canonicalise
    expr = skeleton_from_key(key).expr
    n = [0]
    used = {int(str(c)[1:]) for c in expr.free_symbols if str(c).startswith("c")}
    nxt = [max(used) + 1 if used else 0]

    def fresh():
        sym = sp.Symbol(f"c{nxt[0]}", real=True)
        nxt[0] += 1
        n[0] += 1
        return sym

    def walk(e, in_sum=False):
        if not e.free_symbols:
            # a symbol-free subexpression (2, 0.693, sin(tanh(1)), ...) is a
            # constant; a sign coefficient (-1, 1) is structure, but an
            # additive 1 (as in x1 + 1) is a constant term
            return e if (e in (1, -1) and not in_sum) else fresh()
        if isinstance(e, sp.Add):
            return sp.Add(*[walk(a, in_sum=True) for a in e.args], evaluate=False)
        if isinstance(e, sp.Pow):
            base = walk(e.base)
            ex = e.exp
            if ex.is_Rational and ex.q in (1, 2):
                return sp.Pow(base, ex, evaluate=False)
            return sp.Pow(base, walk(ex), evaluate=False)
        if isinstance(e, sp.Mul):
            args = [a for a in e.args if not (a.is_Number and a in (1, -1))]
            sign = -1 if any(a.is_Number and a == -1 for a in e.args) else 1
            out = sp.Mul(*[walk(a) for a in args], evaluate=False)
            return -out if sign < 0 else out
        if e.args:
            return e.func(*[walk(a) for a in e.args])
        return e

    new = walk(expr)
    if n[0] == 0:
        return key          # no literal to generalise: skip the (slow) re-canonicalisation
    sk = canonicalise(str(new), 1, max_consts=None)
    return key if sk is None else sk.key


def generalised_map(corpus=CORPUS, candidates=CANDIDATES, generalise_literals=True) -> dict:
    """{key: generalised key or None (unparseable)} for every corpus and candidate key."""
    from .candidates import skeleton_from_key
    with open(corpus) as fh:
        keys = {k for ks in json.load(fh)["per_model"].values() for k in ks}
    with open(candidates) as fh:
        keys |= set(json.load(fh)["keys"])
    out = {}
    for key in sorted(keys):
        try:
            kk = generalise(key) if generalise_literals else key
            context_counts(skeleton_from_key(kk).expr)
            out[key] = kk
        except Exception:
            out[key] = None
    return out


def make_folds(corpus=CORPUS, candidates=CANDIDATES, k=5, seed=0, generalise_literals=True, gen_map=None) -> dict:
    """Folds by BioModels model. Each model contributes its distinct forms
    once; forms that cannot be parsed are dropped (as in load_models). With
    generalise_literals, every form (corpus and candidates) is mapped to its
    literal-free generalisation (see generalise) before deduplication."""
    from .candidates import skeleton_from_key
    with open(corpus) as fh:
        d = json.load(fh)
    gen = {}

    if gen_map is not None:
        gen.update(gen_map)

    def g(key):
        if key not in gen:
            try:
                kk = generalise(key) if generalise_literals else key
                context_counts(skeleton_from_key(kk).expr)
                gen[key] = kk
            except Exception:
                gen[key] = None
        return gen[key]

    models = []
    for mid, keys in d["per_model"].items():
        good = sorted({g(key) for key in keys} - {None})
        if good:
            models.append((mid, good))
    with open(candidates) as fh:
        cand = json.load(fh)["keys"]
    support = sorted({g(c) for c in cand} - {None} | {key for _, ks in models for key in ks})
    fold = np.random.default_rng(seed).permutation(len(models)) % k
    folds = [{"train": [key for (_, ks), ff in zip(models, fold) if ff != f for key in ks],
              "test": [key for (_, ks), ff in zip(models, fold) if ff == f for key in ks],
              "test_models": [mid for (mid, _), ff in zip(models, fold) if ff == f]} for f in range(k)]
    return {"support": support, "folds": folds, "k": k, "seed": seed, "generalise_literals": generalise_literals}


def support_counts(support):
    from .candidates import skeleton_from_key
    out = []
    for key in support:
        try:
            out.append(context_counts(skeleton_from_key(key).expr))
        except Exception:
            out.append(None)
    return out


def scheme_logprob(train_counts, scheme, items, kappa=None, grammar=None):
    """Unnormalised log P(item) for each item (Counter or None) under the
    scheme's posterior-mean rule probabilities given the training counts."""
    g = grammar or Grammar()
    cf = _tables(train_counts, "cf", g)
    p_cf = {nt: (g.alpha[nt] + cf.get(((), nt), 0)) / (g.alpha[nt] + cf.get(((), nt), 0)).sum()
            for nt in g.nonterminals}
    probs = {}
    for (ctx, nt), n in _tables(train_counts, scheme, g).items():
        a = g.alpha[nt] if kappa is None else kappa * p_cf[nt] + 0.1
        probs[(ctx, nt)] = (a + n) / (a + n).sum()
    fb = {nt: (g.alpha[nt] / g.alpha[nt].sum() if kappa is None else p_cf[nt]) for nt in g.nonterminals}
    return [None if c is None else float(_log_prob(c, scheme, probs, fb)) for c in items]


def grammar_logpriors(folds: dict, schemes=SCHEMES, kappas=(20,)) -> dict:
    """{variant: [[log P per support key] per fold]} for our grammars."""
    sup = folds["support"]
    sc = dict(zip(sup, support_counts(sup)))
    items = [sc[s] for s in sup]
    out = {"default_prior": [], **{s: [] for s in schemes}}
    for kap in kappas:
        for s in schemes:
            if s != "cf":
                out[f"{s}_backoff{kap}"] = []
    for fold in folds["folds"]:
        train = [sc[key] for key in fold["train"]]
        out["default_prior"].append(scheme_logprob([], "cf", items))
        for s in schemes:
            out[s].append(scheme_logprob(train, s, items))
            for kap in kappas:
                if s != "cf":
                    out[f"{s}_backoff{kap}"].append(scheme_logprob(train, s, items, kappa=kap))
    return {"logprior": out}


def heldout_table(folds: dict, logpriors: dict) -> list:
    """Mean normalised held-out log probability per form (over all test forms
    of all folds), the number of test forms with zero probability, and the
    per-fold means."""
    sup = folds["support"]
    idx = {s: i for i, s in enumerate(sup)}
    rows = [{"prior": "uniform over support", "heldout_logp": -float(np.log(len(sup))), "n_zero": 0,
             "fold_means": [-float(np.log(len(sup)))] * len(folds["folds"])}]
    for name, per_fold in logpriors.items():
        vals, zeros, fmeans = [], 0, []
        for fold, lp in zip(folds["folds"], per_fold):
            lp = np.array([-np.inf if v is None else v for v in lp], float)
            lz = lp - np.logaddexp.reduce(lp[np.isfinite(lp)])
            fv = []
            for key in fold["test"]:
                v = lz[idx[key]]
                if not np.isfinite(v):
                    zeros += 1
                    continue
                fv.append(v)
            vals += fv
            fmeans.append(float(np.mean(fv)) if fv else float("nan"))
        rows.append({"prior": name, "heldout_logp": float(np.mean(vals)), "n_zero": zeros, "fold_means": fmeans})
    return sorted(rows, key=lambda r: -r["heldout_logp"])


def make_transfer(gen_map: dict, test_corpus, corpus=CORPUS, candidates=CANDIDATES) -> dict:
    """One 'fold': train on every BioModels model, test on the forms of another
    corpus (e.g. the R curve-fitting libraries). Same format as make_folds, so
    the same scoring code applies; the support also contains the test forms.
    A test model with several distinct forms contributes each once."""
    base = make_folds(corpus, candidates, k=2, seed=0, gen_map=gen_map)
    train = [key for f in base["folds"] for key in f["test"]]
    with open(test_corpus) as fh:
        T = json.load(fh)
    # every distinct form of every test model (model-form pairs, as in make_folds)
    test = [key for ks in T["per_model"].values() for key in sorted(set(ks))]
    support = sorted(set(base["support"]) | set(test))
    return {"support": support, "folds": [{"train": train, "test": test, "test_models": list(T["per_model"])}],
            "k": 1, "seed": 0, "generalise_literals": True, "test_corpus": os.path.basename(test_corpus)}


SOURCES = {"biomodels": CORPUS,
           "sabiork": os.path.join(os.path.dirname(__file__), "data", "corpus_sabiork.json"),
           "cellml": os.path.join(os.path.dirname(__file__), "data", "corpus_cellml.json"),
           "curvelib": os.path.join(os.path.dirname(__file__), "data", "corpus_curvelib.json"),
           "biocat": os.path.join(os.path.dirname(__file__), "data", "corpus_biocat.json"),
           "biomodels_nc": os.path.join(os.path.dirname(__file__), "data", "corpus_biomodels_nc.json"),
           "thermal": os.path.join(os.path.dirname(__file__), "data", "corpus_thermal.json"),
           "ecocurves": os.path.join(os.path.dirname(__file__), "data", "corpus_ecocurves.json"),
           "modeldb": os.path.join(os.path.dirname(__file__), "data", "corpus_modeldb.json"),
           "nlmixr2lib": os.path.join(os.path.dirname(__file__), "data", "corpus_nlmixr2lib.json"),
           "allodb": os.path.join(os.path.dirname(__file__), "data", "corpus_allodb.json"),
           "jws": os.path.join(os.path.dirname(__file__), "data", "corpus_jws.json")}


def source_items(path, gen_map):
    """Model-form pairs of a corpus (each model's distinct forms once), keys
    mapped to their literal-free generalisation where known."""
    with open(path) as fh:
        pm = json.load(fh)["per_model"]
    out = []
    for ks in pm.values():
        forms = sorted({gen_map.get(k, k) for k in ks} - {None})
        out += forms
    return out


def make_loso(gen_map: dict, test: str, weighting="pooled", sources=None, candidates=CANDIDATES) -> dict:
    """Leave one source out: train on every other source, test on ``test``.
    weighting: 'pooled' (every model-form pair counts once) or 'equal' (each
    training source is replicated to the size of the largest, so every source
    carries the same total weight)."""
    sources = sources or SOURCES
    items = {name: source_items(p, gen_map) for name, p in sources.items()}
    train_parts = {n: v for n, v in items.items() if n != test}
    if weighting == "equal":
        big = max(len(v) for v in train_parts.values())
        train = [k for v in train_parts.values() for k in v * max(1, round(big / len(v)))]
    elif weighting == "pooled":
        train = [k for v in train_parts.values() for k in v]
    else:
        raise ValueError(weighting)
    with open(candidates) as fh:
        cand = json.load(fh)["keys"]
    support = sorted({gen_map.get(c, c) for c in cand} - {None} | {k for v in items.values() for k in v})
    return {"support": support, "folds": [{"train": train, "test": items[test], "test_models": []}],
            "k": 1, "seed": 0, "generalise_literals": True, "test_corpus": f"{test} (left out; {weighting})",
            "train_sizes": {n: len(v) for n, v in train_parts.items()}}
