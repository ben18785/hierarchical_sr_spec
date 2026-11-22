"""Stage A step 1: build a finite candidate set of canonical skeletons.

Two grammar-only sources (no data):
  * exhaustive bottom-up enumeration up to ``enum_size`` rule applications;
  * random derivations from the PCFG (prior-mean theta, optionally tempered
    towards uniform so that larger expressions appear) up to ``max_size``.

One optional data-driven source (``expand_candidates``): one-step edits of the
skeletons that currently fit some dataset best. This is a cheap stand-in for
the PySR route in the spec and for the later "feed the posterior back into the
search" stage.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from .canonical import (Skeleton, canonicalise, consts_of, counts_array, is_const,
                        make_vars, rule_counts, structural_size)
from .grammar import Grammar


@dataclass
class CandidateSet:
    """Canonical skeletons with their rule-count matrix."""

    grammar: Grammar
    skeletons: list
    counts: np.ndarray                      # (n_candidates, n_rules)
    stats: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.skeletons)

    @property
    def keys(self):
        return [s.key for s in self.skeletons]

    def index(self, key: str) -> int:
        if not hasattr(self, "_index"):
            self._index = {s.key: i for i, s in enumerate(self.skeletons)}
        return self._index[key]

    def subset(self, idx) -> "CandidateSet":
        idx = list(idx)
        return CandidateSet(self.grammar, [self.skeletons[i] for i in idx], self.counts[idx], dict(self.stats))

    def add(self, skeletons) -> list:
        """Add new skeletons (deduplicated by key); return indices of the added ones."""
        have = set(self.keys)
        new = [s for s in skeletons if s.key not in have]
        new = list({s.key: s for s in new}.values())
        if not new:
            return []
        start = len(self.skeletons)
        self.skeletons.extend(new)
        self.counts = np.vstack([self.counts, np.array([counts_array(s.expr, self.grammar) for s in new])])
        if hasattr(self, "_index"):
            del self._index
        return list(range(start, len(self.skeletons)))

    # ------------------------------------------------------------- persistence
    def to_json(self, path: str):
        data = {
            "n_vars": self.grammar.n_vars,
            "keys": [s.key for s in self.skeletons],
            "stats": self.stats,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1)

    @classmethod
    def from_json(cls, path: str, grammar: Grammar | None = None) -> "CandidateSet":
        with open(path) as fh:
            data = json.load(fh)
        grammar = grammar or Grammar(n_vars=data["n_vars"])
        sks = [skeleton_from_key(k, grammar.n_vars) for k in data["keys"]]
        return from_skeletons(sks, grammar, data.get("stats", {}))


def skeleton_from_key(key: str, n_vars: int = 1) -> Skeleton:
    """Rebuild a canonical skeleton from its key without re-canonicalising."""
    xs = make_vars(n_vars)
    local = {x.name: x for x in xs}
    local.update({f"c{i}": sp.Symbol(f"c{i}", real=True) for i in range(20)})
    expr = sp.sympify(key, locals=local)
    return Skeleton(key=str(expr), expr=expr, n_consts=len(consts_of(expr)))


def from_skeletons(skeletons, grammar: Grammar, stats=None) -> CandidateSet:
    uniq = list({s.key: s for s in skeletons}.values())
    counts = np.array([counts_array(s.expr, grammar) for s in uniq]).reshape(len(uniq), grammar.n_rules)
    return CandidateSet(grammar, uniq, counts, stats or {})


def from_strings(strings, grammar: Grammar, simplify=True) -> CandidateSet:
    sks = [canonicalise(s, grammar.n_vars, simplify=simplify, max_consts=grammar.max_consts) for s in strings]
    return from_skeletons([s for s in sks if s is not None], grammar)


# ------------------------------------------------------------------ building
def _canon_worker(args):
    s, n_vars, max_consts = args
    try:
        sk = canonicalise(s, n_vars, max_consts=max_consts)
    except Exception:
        return s, None
    return s, (None if sk is None else sk.key)


def canonicalise_many(strings, n_vars=1, max_consts=4, n_jobs=1):
    """Map raw derivation strings to canonical keys (None when dropped)."""
    work = [(s, n_vars, max_consts) for s in strings]
    if n_jobs == 1:
        return dict(map(_canon_worker, work))
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return dict(ex.map(_canon_worker, work, chunksize=64))


def tempered_theta(grammar: Grammar, temperature: float) -> dict:
    """Prior mean raised to 1/temperature and renormalised (T>1 flattens)."""
    out = {}
    for nt, p in grammar.prior_mean().items():
        q = p ** (1.0 / temperature)
        out[nt] = q / q.sum()
    return out


def build_candidates(grammar: Grammar, enum_size: int = 12, n_samples: int = 20000,
                     sample_temperature: float = 2.0, max_size: int | None = None,
                     seed: int = 0, n_jobs: int = 1, verbose: bool = False) -> CandidateSet:
    """Grammar-only candidate set: exhaustive enumeration plus random samples."""
    rng = np.random.default_rng(seed)
    max_size = grammar.max_size if max_size is None else max_size
    raw = list(grammar.enumerate(enum_size)) if enum_size > 0 else []
    n_enum = len(raw)
    theta = tempered_theta(grammar, sample_temperature)
    sampled = []
    for _ in range(n_samples):
        out = grammar.sample(rng, theta, max_size)
        if out is None:
            continue
        s, size, nc = out
        if nc <= grammar.max_consts and size > enum_size:
            sampled.append(s)
    raw_unique = list(dict.fromkeys(raw + sampled))
    if verbose:
        print(f"enumerated {n_enum}, sampled {len(sampled)} beyond size {enum_size}; "
              f"{len(raw_unique)} unique strings to canonicalise")
    mapping = canonicalise_many(raw_unique, grammar.n_vars, grammar.max_consts, n_jobs)
    keys = [k for k in mapping.values() if k is not None]
    key_counts = Counter(keys)
    skeletons = [skeleton_from_key(k, grammar.n_vars) for k in key_counts]
    stats = {
        "n_enumerated": n_enum,
        "n_sampled_beyond_enum": len(sampled),
        "n_unique_strings": len(raw_unique),
        "n_dropped": sum(k is None for k in mapping.values()),
        "n_canonical": len(skeletons),
        "n_merged": len(keys) - len(skeletons),
        "largest_merge": max(key_counts.values()) if key_counts else 0,
    }
    if verbose:
        print(stats)
    return from_skeletons(skeletons, grammar, stats)


# ------------------------------------------------------------ probe merging
def probe_datasets(n_probes=6, n=30, noise=0.05, seed=0, x_low=0.1, x_high=5.0):
    """Generic noisy curves (random combinations of several basis functions)
    used only to detect skeletons that are the same family. They do not
    depend on any benchmark data."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(x_low, x_high, (n_probes, n)), axis=1)
    basis = [lambda t: np.exp(-0.7 * t), lambda t: np.sin(1.3 * t), lambda t: np.sqrt(t),
             lambda t: t / (1 + t), lambda t: np.log(t + 0.5), lambda t: 0.1 * t**2]
    y = np.zeros_like(x)
    for p in range(n_probes):
        w = rng.normal(0, 1, len(basis))
        y[p] = sum(wi * b(x[p]) for wi, b in zip(w, basis))
        y[p] += rng.normal(0, noise * (np.ptp(y[p]) + 1e-9), n)
    return x, y


def merge_by_probes(cands: CandidateSet, n_probes=6, n=30, n_starts=10, decimals=4, seed=0,
                    verbose=False) -> CandidateSet:
    """Merge candidates with the same number of constants whose maximised
    log-likelihoods agree (to ``decimals``) on every probe dataset: these are
    almost surely reparameterisations of one family that the symbolic rules
    missed. A failed optimisation can only split a family, never merge two.
    The representative is the skeleton with the smallest ``skeleton_score``.
    """
    from .canonical import skeleton_score
    from .likelihood import CompiledSkeleton, fit_skeleton, scale_population
    x, y = probe_datasets(n_probes, n, seed=seed)
    X, Y, _, _ = scale_population(list(x), list(y), "pooled")
    X, Y = np.array(X), np.array(Y)
    groups = {}
    for i, sk in enumerate(cands.skeletons):
        try:
            res = fit_skeleton(CompiledSkeleton(sk, cands.grammar.n_vars), X, Y, n_starts=n_starts, seed=i)
            ll = res.max_loglik
        except Exception:
            ll = np.full(n_probes, -np.inf)
        if not np.all(np.isfinite(ll)):
            groups[("unique", i)] = [i]   # cannot judge: keep as is
            continue
        key = (sk.n_consts, tuple(np.round(ll, decimals)))
        groups.setdefault(key, []).append(i)
    keep, merged = [], []
    for members in groups.values():
        best = min(members, key=lambda i: skeleton_score(cands.skeletons[i].expr))
        keep.append(best)
        if len(members) > 1:
            merged.append([cands.skeletons[i].key for i in members])
    keep.sort()
    out = cands.subset(keep)
    out.stats["n_probe_merged"] = len(cands) - len(keep)
    out.stats["probe_merge_groups"] = merged[:200]
    if verbose:
        print(f"probe merge: {len(cands)} -> {len(keep)} candidates ({len(merged)} groups merged)")
    return out


# --------------------------------------------------------- data-driven expansion
def default_edit_library(n_vars=1):
    """Small subexpressions used to replace a variable occurrence."""
    lib = []
    for v in [f"x{i + 1}" for i in range(n_vars)]:
        lib += [f"{v}*{v}", f"C*{v}", f"{v}+C", f"exp(C*{v})", f"log({v})", f"sqrt({v})",
                f"sin(C*{v})", f"{v}*{v}*{v}", f"1/{v}", f"{v}/(C+{v})", f"C*{v}+C*{v}*{v}",
                f"({v})**C", f"tanh(C*{v})"]
    return lib


def _var_positions(expr, path=()):
    if isinstance(expr, sp.Symbol) and not is_const(expr):
        yield path
        return
    for i, a in enumerate(expr.args):
        yield from _var_positions(a, path + (i,))


def _replace_at(expr, path, new):
    if not path:
        return new
    args = list(expr.args)
    args[path[0]] = _replace_at(args[path[0]], path[1:], new)
    return expr.func(*args)


def one_step_edits(skeleton: Skeleton, library_strings, n_vars=1):
    """All expressions obtained by replacing one variable occurrence with a
    library expression, plus the skeleton with one extra additive or
    multiplicative constant."""
    from .canonical import parse_skeleton
    out = []
    n0 = len(consts_of(skeleton.expr))
    for lib in library_strings:
        sub = parse_skeleton(lib, n_vars)
        # shift constants in the library expression so they are fresh
        shift = {c: sp.Symbol(f"c{int(c.name[1:]) + n0 + 50}", real=True) for c in consts_of(sub)}
        sub = sub.xreplace(shift)
        for path in _var_positions(skeleton.expr):
            out.append(_replace_at(skeleton.expr, path, sub))
        if len(list(_var_positions(skeleton.expr))) > 1:
            # the same substitution at every occurrence, sharing constants
            # (x -> x**C everywhere turns Michaelis-Menten into Hill)
            xs = {s for s in skeleton.expr.free_symbols if not is_const(s)}
            out.append(skeleton.expr.xreplace({x: sub for x in xs}))
    c_new = sp.Symbol("c99", real=True)
    c_new2 = sp.Symbol("c98", real=True)
    out.append(skeleton.expr + c_new)
    out.append(skeleton.expr * c_new)
    for x in {s for s in skeleton.expr.free_symbols if not is_const(s)}:
        out.append(skeleton.expr * sp.exp(c_new2 * x))     # damping / cutoff
    return out


def expand_candidates(cands: CandidateSet, top_idx, library=None, n_jobs=1):
    """Add canonicalised one-step edits of the skeletons ``top_idx`` to ``cands``.
    Returns the indices of new candidates."""
    library = library or default_edit_library(cands.grammar.n_vars)
    exprs = []
    for i in top_idx:
        exprs.extend(one_step_edits(cands.skeletons[i], library, cands.grammar.n_vars))
    strings = list(dict.fromkeys(str(e) for e in exprs))
    # strings here already use c-symbols; canonicalise parses them via sympify
    mapping = canonicalise_many_exprs(strings, cands.grammar, n_jobs)
    new = [skeleton_from_key(k, cands.grammar.n_vars) for k in set(mapping.values()) if k is not None]
    return cands.add(new)


def _canon_expr_worker(args):
    s, n_vars, max_consts = args
    xs = make_vars(n_vars)
    local = {x.name: x for x in xs}
    local.update({f"c{i}": sp.Symbol(f"c{i}", real=True) for i in range(200)})
    try:
        expr = sp.sympify(s, locals=local)
        sk = canonicalise(expr, n_vars, max_consts=max_consts)
    except Exception:
        return s, None
    return s, (None if sk is None else sk.key)


def canonicalise_many_exprs(strings, grammar, n_jobs=1):
    work = [(s, grammar.n_vars, grammar.max_consts) for s in strings]
    if n_jobs == 1:
        return dict(map(_canon_expr_worker, work))
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return dict(ex.map(_canon_expr_worker, work, chunksize=32))


def default_n_jobs():
    return max(1, (os.cpu_count() or 2) - 1)


# ------------------------------------------------------------ edit graph
def specialisations(skeleton: Skeleton):
    """Expressions obtained by fixing one constant at a null value (0 or 1).

    These are the candidate *parents* of a skeleton: e.g. fixing the exponent
    of a Hill function at 1 gives Michaelis-Menten."""
    out = []
    for c in sorted(consts_of(skeleton.expr), key=lambda s: s.name):
        for val in (0, 1):
            out.append(skeleton.expr.xreplace({c: sp.Integer(val)}))
    return out


def edit_graph(cands: CandidateSet, sources, library=None, n_jobs=1):
    """Grow ``cands`` around ``sources`` and return the edit edges.

    For each source skeleton s: its one-step edits become children of s, and
    its null-value specialisations (with fewer constants) become parents of
    s. New skeletons are added to ``cands``. Returns (new_idx, edges) with
    edges a set of (parent_idx, child_idx), parent != child."""
    library = library or default_edit_library(cands.grammar.n_vars)
    n_vars = cands.grammar.n_vars
    child_strings, parent_strings = {}, {}
    for i in sources:
        sk = cands.skeletons[i]
        child_strings[i] = [str(e) for e in one_step_edits(sk, library, n_vars)]
        parent_strings[i] = [str(e) for e in specialisations(sk)]
    allstr = list(dict.fromkeys(s for d in (child_strings, parent_strings) for v in d.values() for s in v))
    mapping = canonicalise_many_exprs(allstr, cands.grammar, n_jobs)
    mapping = {s: (k if k is not None and "x" in k else None) for s, k in mapping.items()}
    new_keys = {k for k in mapping.values() if k is not None}
    new_idx = cands.add([skeleton_from_key(k, n_vars) for k in sorted(new_keys)])
    key_idx = {k: i for i, k in enumerate(cands.keys)}
    edges = set()
    for i in sources:
        k_i = cands.skeletons[i].n_consts
        for s in child_strings[i]:
            k = mapping.get(s)
            if k is not None and key_idx[k] != i and cands.skeletons[key_idx[k]].n_consts >= k_i:
                edges.add((i, key_idx[k]))
        for s in parent_strings[i]:
            k = mapping.get(s)
            if k is not None and key_idx[k] != i and cands.skeletons[key_idx[k]].n_consts < k_i:
                edges.add((key_idx[k], i))
    return new_idx, edges
