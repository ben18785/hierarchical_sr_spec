import numpy as np
import pytest

from hsr import treecheck
from hsr import treecode as tc
from hsr.pyp import ContextSpec, GraphPY, TreeHPY

MM = tc.from_prefix(["*", "a", "/", "x", "+", "a", "x"])           # c0*x/(c1 + x)
EXP = tc.from_prefix(["*", "a", "exp", "*", "a", "x"])             # c0*exp(c1*x)
POW = tc.from_prefix(["pow", "x", "a"])
LIN = tc.from_prefix(["+", "a", "*", "a", "x"])


def corpus():
    return [MM] * 20 + [EXP] * 10 + [POW] * 5 + [LIN] * 5 + [tc.from_prefix(["exp", "x"])] * 3


SPECS = {
    "hpy2": ContextSpec(n=2),
    "hpy2_parent": ContextSpec(n=2, drop_R=["l", "a1"]),
    "hpy3": ContextSpec(n=3),
    "graph": ContextSpec(n=2, graph=("a1", "a2", "l")),
    "depth": ContextSpec(n=2, depth_cap=4),
    "source": ContextSpec(n=2, source=True),
    "dhpy": ContextSpec(n=2, source=True, graph=("s",)),
}


def test_treecode_roundtrip_and_events():
    assert tc.to_prefix(MM) == ["*", "a", "/", "x", "+", "a", "x"]
    assert tc.size(MM) == 7 and tc.depth(MM) == 4
    left, right = tc.fnprior_events(MM)
    # fnprior's coding of c0*x/(c1 + x) (codes mapped back to symbols)
    assert left == [("None", "None", "*"), ("None", "*", "a"), ("*", "/", "x"), ("/", "+", "a")]
    assert right == [("*", "a", "/"), ("/", "x", "+"), ("+", "a", "x")]
    with pytest.raises(ValueError):
        tc.from_prefix(["+", "x"])


@pytest.mark.parametrize("name", list(SPECS))
def test_predictive_normalised_and_seating_consistent(name):
    spec = SPECS[name]
    trees = corpus()
    src = ["A" if i % 2 else "B" for i in range(len(trees))] if spec.source else None
    m = TreeHPY(spec, seed=1).fit(trees, src, n_sweeps=6, n_keep=2)
    m.model.check()
    for snap in m.samples:
        for s in ["A", "B", "unseen"] if spec.source else [None]:
            res = treecheck.termination(m.cond_dist(snap, s), lambda p: spec.key(p, s))
            assert abs(res["sym_sum_min"] - 1) < 1e-9 and abs(res["sym_sum_max"] - 1) < 1e-9
            assert 0.9 < res["p_finite"] <= 1 + 1e-9
    # live predictive agrees with the snapshot taken at the end
    key = spec.key(tc.ROOT, "A" if spec.source else None)
    live = np.array([m.model.prob(key, i) for i in range(len(tc.V))])
    assert np.allclose(live, m.samples[-1].vec(key))
    assert m.logp(MM, "A" if spec.source else None) > m.logp(tc.from_prefix(["sin", "x"]), "A" if spec.source else None)


def test_seating_after_add_remove_cycles():
    spec = ContextSpec(n=2, graph=("a2", "a1"))
    g = GraphPY(tc.V, spec, seed=0)
    data = [(spec.key(p, None), tc.V.index(s)) for t in corpus() for p, s in tc.events(t)]
    for k, w in data:
        g.add(k, w)
    g.check()
    rng = np.random.default_rng(0)
    for i in rng.permutation(len(data))[:50]:
        g.remove(*data[i])
        g.add(*data[i])
    g.check()
    for k, w in data:
        g.remove(k, w)
    assert all(r.c == 0 and r.t == 0 for r in g.R.values())


def test_uniform_symbols_lose_mass_to_infinite_trees():
    # 2 leaves, 6 unary, 5 binary out of 13: q = 2/13 + 6/13 q + 5/13 q^2 -> q = 0.4
    res = treecheck.termination(lambda p: ([], 1.0, np.full(13, 1 / 13)), lambda p: 0)
    assert abs(res["p_finite"] - 0.4) < 1e-6


def test_termination_matches_forward_sampling():
    m = TreeHPY(ContextSpec(n=2), seed=2).fit(corpus(), n_sweeps=5, n_keep=1)
    nd = m.cond_dist(m.samples[0])
    pf = treecheck.termination(nd, lambda p: m.spec.key(p, None))["p_finite"]
    rng = np.random.default_rng(0)
    draws = [treecheck.sample_tree(nd, rng, cap=400) for _ in range(3000)]
    frac = np.mean([d is not None for d in draws])
    assert abs(frac - pf) < 0.03
    # the sampled trees' empirical frequency of MM matches its predictive under this sample
    p_mm = np.exp(m.logp_sample(m.samples[0], MM))
    assert abs(np.mean([d == MM for d in draws]) - p_mm) < 4 * np.sqrt(p_mm * (1 - p_mm) / 3000) + 1e-3


@pytest.mark.slow
def test_recovers_discount_and_strength():
    # one restaurant (n=0 left context = the root) fed by a PY(d=0.6, theta=5) CRP over a large uniform base
    rng = np.random.default_rng(3)
    d, th = 0.6, 5.0

    class Flat:
        def parents(self, key):
            return ()

        def group(self, key):
            return key

        def mixgroup(self, key):
            return key

    # simulate a CRP with a base uniform over 1000 dishes
    V = list(range(1000))
    sizes, dishes, data = [], [], []
    for n in range(3000):
        w = np.array([s - d for s in sizes] + [th + d * len(sizes)])
        k = int(rng.choice(len(w), p=w / w.sum()))
        if k == len(sizes):
            sizes.append(1)
            dishes.append(int(rng.integers(1000)))
        else:
            sizes[k] += 1
        data.append(("r", dishes[k]))
    est = []
    g = GraphPY(V, Flat(), seed=4)

    def cb(m, sweep):
        if sweep >= 10:
            est.append((m.d["r"], m.theta["r"]))

    g.fit(data, n_sweeps=40, n_keep=1, callback=cb)
    dm, tm = np.mean(est, 0)
    assert abs(dm - d) < 0.1 and 2 < tm < 12
