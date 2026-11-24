import pytest
"""Themes-and-variations sampler against exact enumeration."""
import itertools
import math

import numpy as np

from hsr.candidates import edit_graph, from_skeletons, from_strings, skeleton_from_key
from hsr.gibbs import GibbsConfig
from hsr.grammar import Grammar
from hsr.themes import EditKernel, ThemeConfig, _p0, run_gibbs_themes


def _exact(L, log_p, kernel, gamma, gamma0, rho):
    J, C = L.shape
    P0 = _p0(log_p)
    Pz = np.zeros((J, C))
    Pt = np.zeros((J, C))
    total = 0.0
    for z in itertools.product(range(C), repeat=J):
        labels = sorted(set(z))
        sizes = [z.count(e) for e in labels]
        crp = gamma ** len(labels) * np.prod([math.factorial(s - 1) for s in sizes])
        lik = np.exp(sum(L[j, z[j]] for j in range(J)))
        for th in itertools.product(range(C), repeat=len(labels)):
            m = np.zeros(C)
            w = 1.0
            for i, t in enumerate(th):
                w *= (m[t] + gamma0 * P0[t]) / (i + gamma0)
                m[t] += 1
                idx, kv = kernel.row(t, rho)
                k = dict(zip(idx.tolist(), kv.tolist())).get(labels[i], 0.0)
                w *= k
            if w == 0:
                continue
            tot = crp * lik * w
            total += tot
            tmap = dict(zip(labels, th))
            for j in range(J):
                Pz[j, z[j]] += tot
                Pt[j, tmap[z[j]]] += tot
    return Pz / total, Pt / total


def test_themes_sampler_matches_enumeration():
    rng = np.random.default_rng(3)
    J, C = 3, 4
    L = rng.normal(0, 1.0, size=(J, C))
    g = Grammar()
    counts = np.abs(rng.normal(2, 1, size=(C, g.n_rules))).round()
    log_p = g.log_prob(counts, g.prior_mean())
    kernel = EditKernel(C, [(0, 1), (0, 2), (1, 2)])
    gamma, gamma0, rho = 0.8, 1.3, 0.6
    Pz, Pt = _exact(L, log_p, kernel, gamma, gamma0, rho)
    cfg = GibbsConfig(n_sweeps=40000, burn_in=1000, n_chains=1, gamma_init=gamma, update_gamma=False,
                      update_theta=False, seed=5)
    tcfg = ThemeConfig(rho_init=rho, update_rho=False, gamma0_init=gamma0, update_gamma0=False)
    res = run_gibbs_themes(L, counts, g, kernel, cfg, tcfg)
    assert np.abs(res.assign_probs - Pz).max() < 0.02
    assert np.abs(res.theme_probs - Pt).max() < 0.02


def test_themes_without_block_move_matches_enumeration():
    rng = np.random.default_rng(8)
    J, C = 3, 4
    L = rng.normal(0, 1.5, size=(J, C))
    g = Grammar()
    counts = np.abs(rng.normal(2, 1, size=(C, g.n_rules))).round()
    log_p = g.log_prob(counts, g.prior_mean())
    kernel = EditKernel(C, [(3, 0), (3, 1)])
    Pz, Pt = _exact(L, log_p, kernel, 1.0, 0.7, 0.4)
    cfg = GibbsConfig(n_sweeps=40000, burn_in=1000, n_chains=1, gamma_init=1.0, update_gamma=False,
                      update_theta=False, relabel=False, seed=2)
    tcfg = ThemeConfig(rho_init=0.4, update_rho=False, gamma0_init=0.7, update_gamma0=False)
    res = run_gibbs_themes(L, counts, g, kernel, cfg, tcfg)
    assert np.abs(res.assign_probs - Pz).max() < 0.02
    assert np.abs(res.theme_probs - Pt).max() < 0.02


def test_kernel_rows_normalised():
    k = EditKernel(5, [(0, 1), (0, 2), (2, 3)])
    for t in range(5):
        idx, v = k.row(t, 0.3)
        assert abs(v.sum() - 1) < 1e-12
    w = np.array([1.0, 0, 0, 0, 0])
    Q = k.push(w, 0.3)
    assert np.allclose(Q, [0.3, 0.35, 0.35, 0, 0])


def test_edit_graph_links_parents_to_variants():
    g = Grammar()
    c = from_strings(["C*x1/(C+x1)", "C*exp(C*x1)"], g)
    _, edges = edit_graph(c, [0, 1])
    keys = c.keys
    pairs = {(keys[a], keys[b]) for a, b in edges}
    assert ("c0*x1/(c1 + x1)", "c0*x1**c1/(c2 + x1**c1)") in pairs
    assert ("c0*exp(c1*x1)", "c0*exp(c1*x1**c2)") in pairs
    # specialisation: Hill as a source gets Michaelis-Menten as a parent
    c2 = from_skeletons([skeleton_from_key("c0*x1**c1/(c2 + x1**c1)")], g)
    _, e2 = edit_graph(c2, [0])
    k2 = c2.keys
    assert ("c0*x1/(c1 + x1)", "c0*x1**c1/(c2 + x1**c1)") in {(k2[a], k2[b]) for a, b in e2}


def test_edit_graph_handles_reprinted_keys():
    # x1/(c0 + x1)/2 canonicalises to "x1/(2*(c0 + x1))", which sympify re-prints as
    # "x1/(2*c0 + 2*x1)"; the edges must still resolve (this raised KeyError before)
    g = Grammar()
    c = from_skeletons([skeleton_from_key("c1*x1/(c0 + x1)")], g)
    import hsr.candidates as hc
    orig = hc.one_step_edits
    sym = {v.name: v for v in c.skeletons[0].expr.free_symbols}
    x1, c0 = sym["x1"], sym["c0"]
    hc.one_step_edits = lambda sk, library, n_vars: [x1 / (c0 + x1) / 2]
    try:
        new, edges = edit_graph(c, [0])
    finally:
        hc.one_step_edits = orig
    assert new and all(0 <= a < len(c.keys) and 0 <= b < len(c.keys) for a, b in edges)


def test_edit_graph_frees_numeric_exponents():
    g = Grammar()
    c = from_skeletons([skeleton_from_key("c0*x1**2/(c1 + x1**2)"), skeleton_from_key("c0*exp(-2*x1)")], g)
    _, edges = edit_graph(c, [0, 1])
    pairs = {(c.keys[a], c.keys[b]) for a, b in edges}
    assert ("c0*x1**2/(c1 + x1**2)", "c0*x1**c1/(c2 + x1**c1)") in pairs
    assert ("c0*exp(-2*x1)", "c0*exp(c1*x1)") in pairs


def test_weighted_kernel_rows_follow_grammar():
    log_w = np.log([1.0, 0.6, 0.2, 0.2, 1.0])
    k = EditKernel(5, [(0, 1), (0, 2), (2, 3), (2, 4)], log_w=log_w)
    idx, v = k.row(0, 0.4)
    assert abs(v.sum() - 1) < 1e-12
    assert np.allclose(dict(zip(idx, v))[1], 0.6 * 0.6 / 0.8)
    idx, v = k.row(2, 0.4)
    assert np.allclose(dict(zip(idx, v))[3], 0.6 * 0.2 / 1.2)
    # push agrees with the rows
    w = np.array([0.3, 0, 0.7, 0, 0])
    Q = k.push(w, 0.4)
    ref = np.zeros(5)
    for t in (0, 2):
        i, v = k.row(t, 0.4)
        ref[i] += w[t] * v
    assert np.allclose(Q, ref)


def test_themes_weighted_kernel_matches_enumeration():
    rng = np.random.default_rng(11)
    J, C = 3, 4
    L = rng.normal(0, 1.0, size=(J, C))
    g = Grammar()
    counts = np.abs(rng.normal(2, 1, size=(C, g.n_rules))).round()
    log_p = g.log_prob(counts, g.prior_mean())
    kernel = EditKernel(C, [(0, 1), (0, 2), (0, 3), (1, 2)], log_w=log_p)
    Pz, Pt = _exact(L, log_p, kernel, 0.9, 1.1, 0.5)
    cfg = GibbsConfig(n_sweeps=40000, burn_in=1000, n_chains=1, gamma_init=0.9, update_gamma=False,
                      update_theta=False, seed=7)
    tcfg = ThemeConfig(rho_init=0.5, update_rho=False, gamma0_init=1.1, update_gamma0=False)
    res = run_gibbs_themes(L, counts, g, kernel, cfg, tcfg)
    assert np.abs(res.assign_probs - Pz).max() < 0.02
    assert np.abs(res.theme_probs - Pt).max() < 0.02


@pytest.mark.parametrize("single_site", [False, True])
def test_themes_split_merge_matches_enumeration(single_site):
    """Split-merge with themes (and the block move; with or without single-site
    updates) leaves the exact posterior over skeletons and themes invariant."""
    rng = np.random.default_rng(21)
    J, C = 4, 4
    L = rng.normal(0, 1.0, size=(J, C))
    L[:2, 0] += 1.5
    L[2:, 2] += 1.5
    g = Grammar()
    counts = np.abs(rng.normal(2, 1, size=(C, g.n_rules))).round()
    log_p = g.log_prob(counts, g.prior_mean())
    kernel = EditKernel(C, [(0, 1), (0, 2), (1, 2), (3, 0)])
    gamma, gamma0, rho = 1.2, 0.9, 0.55
    Pz, Pt = _exact(L, log_p, kernel, gamma, gamma0, rho)
    cfg = GibbsConfig(n_sweeps=30000, burn_in=1000, n_chains=1, gamma_init=gamma, update_gamma=False,
                      update_theta=False, seed=7, split_merge=4, single_site=single_site)
    tcfg = ThemeConfig(rho_init=rho, update_rho=False, gamma0_init=gamma0, update_gamma0=False)
    res = run_gibbs_themes(L, counts, g, kernel, cfg, tcfg)
    assert np.abs(res.assign_probs - Pz).max() < 0.02
    assert np.abs(res.theme_probs - Pt).max() < 0.02


def test_themes_split_merge_exact_with_restricted_label_shortlist(monkeypatch):
    """As the plain test, with the sequential-allocation label shortlist cut
    to the members' own best labels."""
    import hsr.gibbs as hg
    monkeypatch.setattr(hg, "SHORTLIST_TOP", 0)
    test_themes_split_merge_matches_enumeration(False)
