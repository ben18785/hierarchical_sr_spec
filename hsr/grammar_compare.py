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
SCHEMES = ("cf", "parent", "position", "term")
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
            self.c[(fn, pos + ("+" if self.insum and pos in ("num", "den") else ""), nt, opt)] += n

    insum = False

    def E(self, expr, fn):
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
    for (fn, pos, nt, opt), n in counts.items():
        if scheme == "cf":
            ctx = ()
        elif scheme == "parent":
            ctx = (fn,)
        elif scheme == "position":
            ctx = (fn, pos.rstrip("+") if nt in ("T", "R", "V") else "-")
        elif scheme == "term":
            ctx = (fn, pos if nt in ("T", "R", "V") else "-")
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
