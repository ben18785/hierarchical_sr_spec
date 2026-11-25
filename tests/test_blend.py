"""Blended priors (hsr.blend) and the shared gap diagnostics (hsr.gap_diagnostics)."""
import importlib.util
import itertools
import os

import numpy as np
import pytest

from hsr import blend
from hsr import gap_diagnostics as gd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_mix2_matches_direct_sum_and_endpoints():
    lA = np.log(np.array([0.5, 0.3, 0.2]))
    lB = np.log(np.array([0.1, 0.1, 0.8]))
    for w in (0.0, 0.25, 0.7, 1.0):
        assert np.allclose(blend.mix2(lA, lB, w), np.log(w * np.exp(lA) + (1 - w) * np.exp(lB)))
    assert np.array_equal(blend.mix2(lA, lB, 1.0), lA)
    assert np.array_equal(blend.mix2(lA, lB, 0.0), lB)


def test_mixture_of_proper_priors_is_proper():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(50), size=3)
    for w in ([0.2, 0.3, 0.5], [1.0, 0.0, 0.0], [0.01, 0.0, 0.99]):
        assert np.isclose(np.exp(blend.mix_logp(list(np.log(P)), w)).sum(), 1.0)


def test_zero_probability_component():
    lA = np.array([-np.inf, np.log(0.5)])
    lB = np.log(np.array([0.5, 0.5]))
    m = blend.mix2(lA, lB, 0.4)
    assert np.allclose(m, np.log([0.6 * 0.5, 0.4 * 0.5 + 0.6 * 0.5]))
    assert blend.mix2(lA, lB, 1.0)[0] == -np.inf


def test_bad_weights_rejected():
    with pytest.raises(ValueError):
        blend.mix_logp([np.zeros(2), np.zeros(2)], [0.6, 0.6])
    with pytest.raises(ValueError):
        blend.mix_logp([np.zeros(2), np.zeros(2)], [1.2, -0.2])
    with pytest.raises(ValueError):
        blend.mix_logp([np.zeros(2)], [0.5, 0.5])


def test_w_grid_covers_ends_finely():
    g = blend.w_grid()
    assert g[0] == 0.0 and g[-1] == 1.0 and np.all(np.diff(g) > 0)
    assert g[1] < 1e-5 and 1 - g[-2] < 1e-5
    assert np.max(np.diff(g)) <= 0.005 + 1e-12


def _bernoulli_groups():
    # two forms; A puts 0.9 on form 0, B puts 0.1 on it; data: three form-0 pairs, one form-1 pair.
    # P_w(form 0) = 0.1 + 0.8 w, maximised at P_w(form 0) = 3/4, i.e. w = 0.8125
    lA, lB = np.log([0.9, 0.1]), np.log([0.1, 0.9])
    keys = np.array([0, 0, 0, 1])
    return {"s": (lA[keys], lB[keys])}


def test_select_w_finds_known_optimum():
    w, obj, (grid, curve) = blend.select_w(_bernoulli_groups())
    assert abs(w - 0.8125) <= 0.005
    assert np.isclose(obj, 0.75 * np.log(0.75) + 0.25 * np.log(0.25), atol=1e-4)
    assert np.isclose(obj, curve.max())


def test_select_w_boundary():
    lA, lB = np.log([0.6, 0.4]), np.log([0.4, 0.6])
    w, _, _ = blend.select_w({"s": (lA[[0, 0]], lB[[0, 0]])})
    assert w == 1.0
    w, _, _ = blend.select_w({"s": (lA[[1, 1]], lB[[1, 1]])})
    assert w == 0.0


def test_select_w_weights_sources_equally():
    lA, lB = np.log([0.9, 0.1]), np.log([0.1, 0.9])
    big = np.array([0] * 90)       # a large source of form 0 favours A
    small = np.array([1] * 10)     # a small source of form 1 favours B
    groups = {"big": (lA[big], lB[big]), "small": (lA[small], lB[small])}
    w_src, _, _ = blend.select_w(groups, mean_over="sources")
    w_pair, _, _ = blend.select_w(groups, mean_over="pairs")
    # sources: maximise 0.5 log(0.1 + 0.8 w) + 0.5 log(0.9 - 0.8 w): P_w(form 0) = 1/2, w = 0.5
    assert abs(w_src - 0.5) <= 0.005
    # pairs: 0.9 log(.) + 0.1 log(.): P_w(form 0) = 0.9, w = 1
    assert w_pair == 1.0
    manual = [0.5 * (blend.mix2(*groups["big"], w).mean() + blend.mix2(*groups["small"], w).mean())
              for w in (0.3, w_src, 0.7)]
    assert manual[1] >= max(manual[0], manual[2])


def test_em_matches_grid_for_two_components():
    g = _bernoulli_groups()
    w, obj = blend.em_weights({s: list(v) for s, v in g.items()})
    assert np.isclose(w[0], 0.8125, atol=1e-4) and np.isclose(w.sum(), 1.0)
    assert np.isclose(obj, blend.source_mean(g, 0.8125), atol=1e-8)


def test_em_three_components_beats_simplex_grid():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(6), size=3)
    data = {"a": rng.integers(0, 6, 40), "b": rng.integers(0, 3, 15)}
    groups = {s: [np.log(P[k][x]) for k in range(3)] for s, x in data.items()}
    w, obj = blend.em_weights(groups)
    best = -np.inf
    for i, j in itertools.product(range(21), repeat=2):
        if i + j <= 20:
            v = np.array([i, j, 20 - i - j]) / 20
            best = max(best, np.mean([blend.mix_logp(ls, v).mean() for ls in groups.values()]))
    assert obj >= best - 1e-9


def test_ols_novel_recovers_coefficients():
    nodes = np.array([3, 5, 8, 10, 12, 15, 20, 25], float)
    novel = np.array([0, 0, 0, 1, 0, 1, 1, 1], bool)
    y = 1.0 - 0.7 * nodes - 4.0 * novel
    b, g = gd.ols_novel(y, nodes, novel)
    assert np.isclose(b, -0.7) and np.isclose(g, -4.0)
    assert gd.ols_novel(y, nodes, np.zeros(8, bool)) == (None, None)


def test_gap_stats():
    raw = np.array([-2.0, -3.0, -10.0, -12.0])
    nodes = np.array([3, 4, 9, 11], float)
    seen = np.array([True, True, False, False])
    red = np.array([False, False, True, False])
    o = gd.gap_stats(raw, nodes, seen, red, sym0=raw - 1.0)
    assert np.isclose(o["raw_seen"], -2.5) and np.isclose(o["raw_novel"], -11.0) and np.isclose(o["gap"], 8.5)
    assert np.isclose(o["raw_novel_nonredundant"], -12.0)
    assert np.isclose(o["vs_sym0_seen"], 1.0) and np.isclose(o["vs_sym0_novel"], 1.0)
    assert o["by_size"]["1-5"] == {"n_seen": 2, "n_novel": 0, "seen": -2.5, "novel": None}
    assert o["by_size"]["6-9"]["n_novel"] == 1


def _temporal_steps():
    spec = importlib.util.spec_from_file_location("temporal_steps",
                                                  os.path.join(ROOT, "workflow", "scripts", "temporal_steps.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_temporal_report_exclusion_and_by_source():
    ts = _temporal_steps()
    F = {"support": ["a", "b", "c"],
         "folds": [{"train": ["a"], "test": ["a", "b", "c", "b"], "novel": [False, True, True, True],
                    "test_sources": ["modeldb", "modeldb", "jws", "jws"]}]}
    lp = np.log([0.5, 0.25, 0.25])
    scores = {"p": ("tree", list(lp)), "sym0": ("tree", list(lp - 1))}
    keep = np.array([s != "modeldb" for s in F["folds"][0]["test_sources"]])
    rows = ts.run_metrics(F, scores, keep)
    assert np.isclose(rows["p"]["raw"], np.log(0.25)) and rows["p"]["raw_seen"] is None
    allr = ts.run_metrics(F, scores)
    assert np.isclose(allr["p"]["raw_seen"], np.log(0.5))
    bs = ts.novel_by_source(F, scores, {"a": 1, "b": 3, "c": 5}, ["p", "sym0"])
    assert bs["jws"]["novel_pairs"] == 2 and bs["jws"]["novel_distinct"] == 2
    assert np.isclose(bs["jws"]["nodes_novel"], 4.0) and np.isclose(bs["modeldb"]["novel"]["p"], np.log(0.25))
    gap, counts = ts.gap_metrics(F, scores, {"a": 1, "b": 3, "c": 5}, lambda k: k == "c", keep)
    assert counts["novel_pairs"] == 2 and counts["redundant_novel_pairs"] == 1 and counts["redundant_novel"] == ["c"]
    assert np.isclose(gap["p"]["raw_novel_nonredundant"], np.log(0.25))
