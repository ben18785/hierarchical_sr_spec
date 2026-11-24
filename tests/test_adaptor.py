import numpy as np
import pytest

from hsr import treecheck
from hsr import treecode as tc
from hsr.adaptor import TreeAdaptor
from hsr.pyp import ContextSpec

MM = tc.from_prefix(["*", "a", "/", "x", "+", "a", "x"])
EXP = tc.from_prefix(["*", "a", "exp", "*", "a", "x"])
POW = tc.from_prefix(["pow", "x", "a"])
TWIN = tc.from_prefix(["+", "exp", "x", "exp", "x"])     # the same subtree twice in one tree


def corpus():
    return [MM] * 15 + [EXP] * 8 + [POW] * 4 + [TWIN] * 3 + [tc.from_prefix(["exp", "x"])] * 3


@pytest.mark.parametrize("mode", ["root", "all"])
@pytest.mark.parametrize("graph", [(), ("a2", "a1", "l")])
def test_invariants_normalisation_and_sampling(mode, graph):
    spec = ContextSpec(n=2, graph=graph)
    m = TreeAdaptor(spec, mode=mode, seed=3).fit(corpus(), n_sweeps=8, n_keep=2)
    m.check()
    assert 0 < np.mean(m.acc_rate) <= 1
    snap = m.samples[-1]
    nd = lambda p: snap.node_dist(p)                          # noqa: E731
    res = treecheck.termination(nd, lambda p: (spec.key(p, None), m.adapted(p)))
    assert abs(res["total_min"] - 1) < 1e-9 and abs(res["total_max"] - 1) < 1e-9
    assert 0.9 < res["p_finite"] <= 1 + 1e-9
    rng = np.random.default_rng(1)
    draws = [treecheck.sample_tree(nd, rng, cap=400) for _ in range(3000)]
    for T in (MM, POW):
        p = snap.p_tree(tc.ROOT, None, T)
        f = np.mean([d == T for d in draws])
        assert abs(f - p) < 4 * np.sqrt(p * (1 - p) / 3000) + 1e-3, (T, f, p)
    # reuse: a frequent form is far more probable than under a fresh draw
    assert m.logp(MM) > np.log(0.2)


def test_remove_then_restore_is_identity_on_counts():
    spec = ContextSpec(n=2)
    m = TreeAdaptor(spec, mode="all", seed=5).fit(corpus(), n_sweeps=2, n_keep=1)
    before = {k: (r.n, r.K) for k, r in m.A.items()}
    for i in range(len(m.trees)):
        T, rec = m.trees[i], m.recs[i]
        m._remove(tc.ROOT, None, T, rec)
        m._restore(tc.ROOT, None, T, rec)
        m.check()
    assert {k: (r.n, r.K) for k, r in m.A.items()} == before


def test_logw_of_removal_matches_proposal():
    # W accumulated when proposing an analysis equals W recomputed when removing it
    spec = ContextSpec(n=2)
    m = TreeAdaptor(spec, mode="all", seed=6).fit(corpus(), n_sweeps=2, n_keep=1)
    # freeze the symbol-level seating (its removals pick a random customer, so
    # with it live the two would differ by the seating of the symbol events)
    m.hpy.add = lambda key, w, obs=True: m.hpy.prob(key, w)
    m.hpy.remove = lambda key, w, obs=True: None
    for T in (MM, TWIN, EXP, TWIN):
        rec, lw = m._propose(tc.ROOT, None, T)
        lw2 = m._remove(tc.ROOT, None, T, rec)
        assert abs(lw - lw2) < 1e-9


@pytest.mark.slow
def test_recovers_root_adaptor_parameters_with_the_true_base():
    # forms from PY(a=0.5, b=20) over a fixed symbol-level base; the symbol
    # model is frozen at that base, so the model is correctly specified
    rng = np.random.default_rng(11)
    probs = {"a": 0.25, "x": 0.25, "exp": 0.1, "+": 0.1, "*": 0.1, "-": 0.1, "/": 0.1}
    syms = list(probs)
    pv = np.array([probs[s] for s in syms])

    def base():
        s = syms[rng.choice(len(syms), p=pv)]
        return (s,) + tuple(base() for _ in range(tc.ARITY[s]))

    a, b = 0.5, 20.0
    sizes, dishes, trees = [], [], []
    for _ in range(2000):
        w = np.array([s - a for s in sizes] + [b + a * len(sizes)])
        k = int(rng.choice(len(w), p=w / w.sum()))
        if k == len(sizes):
            sizes.append(1)
            dishes.append(base())
        else:
            sizes[k] += 1
        trees.append(dishes[k])
    m = TreeAdaptor(ContextSpec(n=2), mode="root", seed=2)
    vec = np.array([probs.get(s, 0.0) for s in tc.V])
    m.hpy.prob = lambda key, w: vec[w]
    m.hpy.add = lambda key, w, obs=True: vec[w]
    m.hpy.remove = lambda key, w, obs=True: None
    m.hpy.gibbs_sweep = lambda *k: None
    m.hpy.update_hyper = lambda *k: None
    g = ("A", "root")
    est = []
    m.fit(trees, n_sweeps=1, n_keep=1)
    for _ in range(30):
        for i in m.rng.permutation(len(trees)):
            m.update_tree(int(i))
        m.update_ahyper()
        est.append((m.a[g], m.b[g], m.A[m.spec.key(tc.ROOT, None)].K))
    am, bm, Km = np.mean(est[10:], 0)
    # the discount and the number of tables are recovered; the strength is
    # weakly identified (it trades off against the discount) and its
    # Gamma(1, 1) prior pulls it down, so it is not checked
    assert abs(am - a) < 0.1 and abs(Km / len(sizes) - 1) < 0.1, (am, bm, Km, len(sizes))
