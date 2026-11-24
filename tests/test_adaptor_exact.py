"""Exactness of the adaptor sampler against brute-force enumeration of every
seating path on tiny data (hyperparameters fixed)."""
from collections import defaultdict

import numpy as np
import pytest

from hsr import treecode as tc
from hsr.adaptor import TreeAdaptor
from hsr.pyp import ContextSpec

E3 = tc.from_prefix(["exp", "exp", "exp", "x"])     # visits restaurant (exp, exp) twice
E1 = tc.from_prefix(["exp", "x"])
NV = len(tc.V)


def enumerate_joint(trees, spec, mode, a, b, d, th, frozen=None):
    """{signature: P(trees, adaptor seating with that signature)} summed over
    every seating path. frozen: fixed symbol probabilities key -> vector
    (then the symbol level is not a restaurant hierarchy)."""
    out = defaultdict(float)

    def adapted(pos):
        return mode == "all" or pos.depth == 0

    def hpy_add(H, key, w):
        """[(prob, new H)] over the seating choices of a symbol event."""
        r = H.get(key, {})
        tabs = r.get(w, ())
        c = sum(sum(t) for t in r.values())
        t = sum(len(x) for x in r.values())
        res = []
        for k, s in enumerate(tabs):
            H2 = dict(H)
            r2 = dict(r)
            r2[w] = tabs[:k] + (s + 1,) + tabs[k + 1:]
            H2[key] = r2
            res.append(((s - d) / (th + c), H2))
        pnew = (th + d * t) / (th + c)
        H2 = dict(H)
        r2 = dict(r)
        r2[w] = tabs + (1,)
        H2[key] = r2
        ps = spec.parents(key)
        if not ps:
            res.append((pnew / NV, H2))
        else:
            for p, H3 in hpy_add(H2, ps[0], w):
                res.append((pnew * p, H3))
        return res

    def run(A, H, tasks, prob):
        if not tasks:
            sig = frozenset((k, T, tuple(sorted(s))) for k, ds in A.items() for T, s in ds.items() if s)
            out[sig] += prob
            return
        task, rest = tasks[0], tasks[1:]
        if task[0] == "sym":
            _, key, w = task
            if frozen is not None:
                run(A, H, rest, prob * frozen[key][w])
                return
            for p, H2 in hpy_add(H, key, w):
                run(A, H2, rest, prob * p)
            return
        _, pos, T = task
        key = spec.key(pos, None)
        kids = [("cust", cp, Tc) for cp, Tc in zip(TreeAdaptor.kids_pos(pos, T), T[1:])]
        sym = [("sym", key, tc.V.index(T[0]))]
        if not adapted(pos):
            run(A, H, sym + kids + rest, prob)
            return
        ds = A.get(key, {})
        n = sum(sum(s) for s in ds.values())
        K = sum(len(s) for s in ds.values())
        tabs = ds.get(T, ())
        for k, s in enumerate(tabs):
            A2 = dict(A)
            A2[key] = {**ds, T: tabs[:k] + (s + 1,) + tabs[k + 1:]}
            run(A2, H, rest, prob * (s - a) / (b + n))
        A2 = dict(A)
        A2[key] = {**ds, T: tabs + (1,)}
        run(A2, H, sym + kids + rest, prob * (b + a * K) / (b + n))

    run({}, {}, [("cust", tc.ROOT, T) for T in trees], 1.0)
    return out


def _fixed(m, a, b, d, th):
    """Fix every hyperparameter of a TreeAdaptor (no updates)."""
    m.a0, m.b0 = a, b
    m.hpy.d0, m.hpy.theta0 = d, th
    m.update_ahyper = lambda *k, **kw: None
    m.hpy.update_hyper = lambda *k, **kw: None


def _signature(m):
    return frozenset((k, T, tuple(sorted(t.size for t in ts))) for k, r in m.A.items()
                     for T, ts in r.dishes.items() if ts)


@pytest.mark.slow
def test_adaptor_posterior_matches_enumeration_with_fixed_base():
    spec = ContextSpec(n=2)
    trees = [E3, E3, E1]
    a, b = 0.3, 5.0
    rng = np.random.default_rng(0)
    # most base mass on exp and x, so that fresh tables (and the inexact part
    # of the proposal: E3 revisits restaurant (exp, exp)) are common
    conc = 0.3 + 30 * np.isin(np.arange(NV), [tc.V.index("x"), tc.V.index("exp")])
    frozen = defaultdict(lambda: rng.dirichlet(conc))
    exact = enumerate_joint(trees, spec, "all", a, b, 0.5, 1.0, frozen=frozen)
    Z = sum(exact.values())
    m = TreeAdaptor(spec, mode="all", seed=1)
    _fixed(m, a, b, 0.5, 1.0)
    m.hpy.prob = lambda key, w: frozen[key][w]
    m.hpy.add = lambda key, w, obs=True: frozen[key][w]
    m.hpy.remove = lambda key, w, obs=True: None
    m.hpy.gibbs_sweep = lambda *k: None
    m.fit(trees, n_sweeps=1, n_keep=1)
    freq = defaultdict(int)
    N = 20000
    for _ in range(N):
        for i in range(len(trees)):
            m.update_tree(i)
        freq[_signature(m)] += 1
    assert set(freq) <= set(exact)
    for sig, p in exact.items():
        p /= Z
        assert abs(freq[sig] / N - p) < 4 * np.sqrt(p * (1 - p) / N) + 0.005, (sig, freq[sig] / N, p)
    assert m.n_acc < m.n_prop        # the proposal is not exact here, so some moves are rejected


@pytest.mark.slow
def test_adaptor_predictive_matches_enumeration_with_hpy_base():
    # live symbol-level HPY (random customer removal + MH): the posterior
    # predictive of a single-leaf tree equals P(data + leaf) / P(data)
    spec = ContextSpec(n=2)
    trees = [E3, E1, E3]
    a, b, d, th = 0.4, 0.7, 0.6, 0.8
    ex0 = sum(enumerate_joint(trees, spec, "all", a, b, d, th).values())
    exact = {leaf: sum(enumerate_joint(trees + [(leaf,)], spec, "all", a, b, d, th).values()) / ex0
             for leaf in ("x", "a")}
    m = TreeAdaptor(spec, mode="all", seed=2)
    _fixed(m, a, b, d, th)
    m.fit(trees, n_sweeps=1, n_keep=1)
    est = defaultdict(list)
    for _ in range(15000):
        for i in range(len(trees)):
            m.update_tree(i)
        m.hpy.gibbs_sweep()
        for leaf in exact:
            est[leaf].append(m.p_tree(tc.ROOT, None, (leaf,)))
    for leaf, p in exact.items():
        v = np.array(est[leaf])
        se = v.std() / np.sqrt(len(v) / 50)          # generous for autocorrelation
        assert abs(v.mean() - p) < 4 * se + 1e-4, (leaf, v.mean(), p)
