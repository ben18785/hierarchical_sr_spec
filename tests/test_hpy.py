import numpy as np

from hsr.hpy import HPY


def test_predictive_is_a_distribution_and_learns_contexts():
    rng = np.random.default_rng(0)
    V = ["a", "b", "c", "d"]
    ev = [(("x", "y"), "a")] * 30 + [(("x", "z"), "b")] * 30 + [((rng.choice(V), "q"), rng.choice(V)) for _ in range(40)]
    m = HPY(V, seed=1).fit([(tuple(u), str(w)) for u, w in ev], n_sweeps=15, n_keep=5)
    for u in [("x", "y"), ("x", "z"), ("p", "q"), ("new", "ctx")]:
        assert abs(sum(m.predictive(u, w) for w in V) - 1) < 1e-9
    assert m.predictive(("x", "y"), "a") > 0.8 and m.predictive(("x", "z"), "b") > 0.8
    assert all(0 < d < 1 for d in m.d) and all(t > 0 for t in m.theta)


def test_seating_counts_stay_consistent():
    m = HPY(["a", "b"], seed=3).fit([(("u", "v"), "a"), (("u", "v"), "b"), (("w", "v"), "a")] * 5, n_sweeps=5, n_keep=1)
    for u, ws in m.tables.items():
        assert m.c[u] == sum(sum(s) for s in ws.values())
        assert m.t[u] == sum(len(s) for s in ws.values())
    # customers of the empty context = tables of the length-1 contexts
    assert m.c[()] == sum(m.t[u] for u in m.tables if len(u) == 1)
