"""A fixed, corpus-derived prior over skeletons as the DP base measure.

With ``--prior fnprior`` the base measure of the Dirichlet process is the
n-gram prior on functions of Bartlett & Desmond (fnprior, Kneser-Ney
smoothing, start-of-tree context, constants coded separately from the
variable), trained by default on all the biology corpora in
hsr.grammar_compare.SOURCES (BioModels, SABIO-RK, CellML, R curve libraries,
biology catalogue) with equal weight per source: within a source every model
contributes its distinct forms once, after numeric literals are generalised to
constants (hsr.grammar_compare.generalise), and each source is replicated to
the size of the largest. This prior (with this weighting) won the held-out and
leave-one-source-out comparisons of priors (workflow/Snakefile, rule
``grammar``). ``corpus="biomodels"`` gives the earlier BioModels-only prior.

The prior is fixed: the grammar's rule probabilities are no longer updated
during sampling; the DP, pooled constants and themes are unchanged.

Mechanism. The samplers and baselines take a ``grammar`` and per-candidate
``counts`` and only ever call grammar.log_prob(counts, theta),
grammar.prior_mean(), grammar.sample_theta(rng, counts) and
grammar.nonterminals. ``FixedPrior`` implements these with log P(e) carried
as an extra last column of ``counts`` (``with_log_prior``), so every code
path that subsets candidates keeps the right prior for each row.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .grammar import Grammar

CORPUS = os.path.join(os.path.dirname(__file__), "data", "corpus.json")
_MODELS = {}


class FixedPrior:
    """Grammar stand-in whose log_prob is the last column of ``counts``."""

    def __init__(self, base: Grammar, label: str = "fixed"):
        self.base = base
        self.label = label
        self.nonterminals = base.nonterminals
        self.options = base.options
        self.alpha = base.alpha

    def prior_mean(self):
        return self.base.prior_mean()

    def sample_theta(self, rng, counts=None):
        return self.base.prior_mean()

    def log_prob(self, counts, theta=None):
        return np.asarray(counts, float)[..., -1]


def with_log_prior(counts, log_prior):
    """Append log P(e) as the last column of a (C, R) counts matrix."""
    return np.hstack([np.asarray(counts, float), np.asarray(log_prior, float)[:, None]])


GENERALISED = os.path.join(os.path.dirname(__file__), "data", "corpus_generalised.json")


def build_generalised(corpus=CORPUS, out=GENERALISED):
    """Precompute {corpus key: literal-free key} (slow: sympy on large rate
    laws). Built by the workflow rule ``corpus_generalised``."""
    from .grammar_compare import generalise
    with open(corpus) as fh:
        keys = sorted({k for ks in json.load(fh)["per_model"].values() for k in ks})
    m = {}
    for k in keys:
        try:
            m[k] = generalise(k)
        except Exception:
            m[k] = None
    with open(out, "w") as fh:
        json.dump({"corpus": os.path.basename(corpus), "map": m}, fh, indent=0, sort_keys=True)


def _training_keys(corpus, exclude):
    from .grammar_compare import generalise
    with open(corpus) as fh:
        per_model = json.load(fh)["per_model"]
    gen, out = {}, []
    if os.path.abspath(corpus) == os.path.abspath(CORPUS) and os.path.exists(GENERALISED):
        with open(GENERALISED) as fh:
            gen.update(json.load(fh)["map"])
    ex = set()
    for k in exclude:
        try:
            ex.add(generalise(k))
        except Exception:
            ex.add(k)
    for keys in per_model.values():
        forms = set()
        for k in keys:
            if k not in gen:
                try:
                    gen[k] = generalise(k)
                except Exception:
                    gen[k] = None
            if gen[k] is not None and gen[k] not in ex:
                forms.add(gen[k])
        out += sorted(forms)
    return out


def _combined_training_keys(exclude, weighting="equal"):
    """Training keys from every source in hsr.grammar_compare.SOURCES, with the
    given source weighting (keys matching ``exclude`` after generalisation are dropped)."""
    from .grammar_compare import SOURCES, generalise, weight_sources
    parts = {}
    for name, path in SOURCES.items():
        if os.path.abspath(path) == os.path.abspath(CORPUS):
            parts[name] = _training_keys(path, exclude)
            continue
        ex = set()
        for k in exclude:
            try:
                ex.add(generalise(k))
            except Exception:
                ex.add(k)
        with open(path) as fh:
            pm = json.load(fh)["per_model"]
        # these corpora are stored with literals already generalised
        parts[name] = [k for ks in pm.values() for k in sorted(set(ks)) if k not in ex]
    return weight_sources(parts, weighting)


def fnprior_model(corpus="combined", exclude=(), weighting="equal"):
    """Trained fnprior FunctionPrior (memoised per corpus, weighting and
    exclusion set). corpus: "combined" (all sources), "biomodels", or a path."""
    if corpus == "biomodels":
        corpus = CORPUS
    key = (corpus if corpus == "combined" else os.path.abspath(corpus), weighting, tuple(sorted(exclude)))
    if key not in _MODELS:
        import tempfile
        from . import fnprior_tools as mod
        keys = (_combined_training_keys(exclude, weighting) if corpus == "combined"
                else _training_keys(corpus, exclude))
        with tempfile.TemporaryDirectory() as tmp:
            kp = mod.train(keys, False, 2, tmp, "corpus")
        _MODELS[key] = (kp, mod, len(keys))
    return _MODELS[key]


def fnprior_log_prior(skeleton_keys, corpus="combined", exclude=(), weighting="equal"):
    """log P(e) under fnprior for each key (floor for unscorable keys: the
    smallest finite value minus 10 nats)."""
    kp, mod, _ = fnprior_model(corpus, exclude, weighting)
    v = mod.score(kp, list(skeleton_keys))
    arr = np.array([np.nan if x is None else x for x in v], float)
    if np.any(~np.isfinite(arr)):
        floor = np.nanmin(arr[np.isfinite(arr)]) - 10.0 if np.any(np.isfinite(arr)) else -50.0
        arr = np.where(np.isfinite(arr), arr, floor)
    return arr
