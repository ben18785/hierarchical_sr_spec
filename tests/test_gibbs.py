import itertools

import numpy as np
import pytest
from scipy import integrate, special

from hsr.gibbs import GibbsConfig, escobar_west_gamma, run_gibbs
from hsr.grammar import Grammar


def one_var_grammar(d):
    """A grammar whose 'skeletons' are just x1..xd, each using a single V
    rule, so P_theta(e) = theta_V[e] sums to one over the candidate set."""
    g = Grammar(n_vars=d)
    counts = np.zeros((d, g.n_rules))
    for e in range(d):
        counts[e, g.rule_index[("V", f"x{e + 1}")]] = 1
    return g, counts


def test_sharp_likelihood_recovers_known_assignment():
    rng = np.random.default_rng(0)
    g, counts = one_var_grammar(20)
    J = 12
    truth = rng.choice([3, 7, 11], size=J)
    L = np.full((J, 20), -1000.0)
    L[np.arange(J), truth] = 0.0
    res = run_gibbs(L, counts, g, GibbsConfig(n_sweeps=200, burn_in=50, n_chains=2))
    assert np.array_equal(res.top, truth)
    assert np.allclose(res.assign_probs[np.arange(J), truth], 1.0)
    assert np.all(res.n_forms == len(set(truth)))
    assert res.chain_disagreement() == 0.0


def test_all_minus_inf_row_does_not_crash():
    g, counts = one_var_grammar(4)
    L = np.zeros((3, 4))
    L[1] = -np.inf
    res = run_gibbs(L, counts, g, GibbsConfig(n_sweeps=30, burn_in=10, n_chains=1))
    assert np.isclose(res.assign_probs[1].sum(), 1.0)


def _exact_posterior(L, logp, gamma):
    """Exact P(z | L) ∝ Polya-urn(z) * prod_j exp(L[j, z_j]) by enumeration."""
    J, C = L.shape
    p = np.exp(logp)
    probs = {}
    for z in itertools.product(range(C), repeat=J):
        lw = 0.0
        n = np.zeros(C)
        for i, e in enumerate(z):
            lw += np.log(n[e] + gamma * p[e]) - np.log(gamma + i)
            n[e] += 1
        lw += sum(L[j, e] for j, e in enumerate(z))
        probs[z] = lw
    keys = list(probs)
    lw = np.array([probs[k] for k in keys])
    w = np.exp(lw - special.logsumexp(lw))
    marg = np.zeros((J, C))
    for k, wk in zip(keys, w):
        for j, e in enumerate(k):
            marg[j, e] += wk
    return marg


@pytest.mark.parametrize("relabel", [False, True])
def test_sampler_matches_exact_posterior_on_tiny_problem(relabel):
    """Validation: with theta and gamma fixed, the Gibbs marginals equal the
    exact posterior obtained by enumerating every assignment (with and
    without the block relabel move)."""
    rng = np.random.default_rng(1)
    g, counts = one_var_grammar(3)
    J = 4
    L = rng.normal(0, 1.0, (J, 3))
    theta = g.prior_mean()
    gamma = 0.7
    cfg = GibbsConfig(n_sweeps=12000, burn_in=500, n_chains=2, update_theta=False, update_gamma=False,
                      gamma_init=gamma, seed=3, relabel=relabel)
    res = run_gibbs(L, counts, g, cfg)
    exact = _exact_posterior(L, g.log_prob(counts, theta), gamma)
    assert np.max(np.abs(res.assign_probs - exact)) < 0.02


def test_relabel_move_exact_with_large_base_probabilities():
    """The relabel conditional includes the prod (k + gamma P(e)) term, which
    matters when gamma P(e) is not small; check against enumeration with a
    3-candidate base measure and a large gamma."""
    rng = np.random.default_rng(8)
    g, counts = one_var_grammar(3)
    J = 5
    L = rng.normal(0, 0.7, (J, 3))
    gamma = 4.0
    cfg = GibbsConfig(n_sweeps=15000, burn_in=500, n_chains=2, update_theta=False, update_gamma=False,
                      gamma_init=gamma, seed=9, relabel=True)
    res = run_gibbs(L, counts, g, cfg)
    exact = _exact_posterior(L, g.log_prob(counts, g.prior_mean()), gamma)
    assert np.max(np.abs(res.assign_probs - exact)) < 0.02


def test_relabel_moves_a_stuck_cluster():
    """Ten datasets all start on skeleton 0. Jointly they prefer skeleton 1 by
    10 nats, but a single dataset leaving the cluster pays log(9 / gamma P(1)),
    so single-site Gibbs stays stuck while the relabel move escapes."""
    g, counts = one_var_grammar(3)
    J = 10
    L = np.zeros((J, 3))
    L[:, 1] = 1.0
    L[:, 2] = -50
    base = dict(n_sweeps=300, burn_in=50, n_chains=1, update_theta=False, update_gamma=False, gamma_init=1e-3)
    start = np.zeros(J, dtype=int)
    stuck = run_gibbs(L, counts, g, GibbsConfig(relabel=False, **base), z_init=start)
    moved = run_gibbs(L, counts, g, GibbsConfig(relabel=True, **base), z_init=start)
    assert np.all(stuck.assign_probs[:, 0] > 0.95)
    assert np.all(moved.assign_probs[:, 1] > 0.95)


def _gamma_posterior_mean(K, n, a, b):
    """E[gamma | K] with p(gamma | K) ∝ Gamma(a, rate b) gamma^K Gamma(gamma)/Gamma(gamma+n)."""
    def logf(gm):
        return ((a - 1) * np.log(gm) - b * gm + K * np.log(gm) + special.gammaln(gm) - special.gammaln(gm + n))
    grid_max = 60.0
    z, _ = integrate.quad(lambda gm: np.exp(logf(gm)), 1e-10, grid_max, limit=200)
    m, _ = integrate.quad(lambda gm: gm * np.exp(logf(gm)), 1e-10, grid_max, limit=200)
    return m / z


@pytest.mark.parametrize("K,n", [(1, 30), (3, 30), (10, 30)])
def test_escobar_west_targets_correct_posterior(K, n):
    rng = np.random.default_rng(5)
    a, b = 1.0, 1.0
    gm = 1.0
    draws = []
    for i in range(30000):
        gm = escobar_west_gamma(rng, gm, K, n, a, b)
        if i > 500:
            draws.append(gm)
    exact = _gamma_posterior_mean(K, n, a, b)
    assert abs(np.mean(draws) - exact) < 0.05 * exact + 0.02


def test_theta_posterior_mean_uses_distinct_skeletons_once():
    """theta_V | z ~ Dir(alpha + counts of distinct skeletons in use): with
    every dataset fixed on x1 and x2, the V counts are (1, 1, 0, ...), not
    the number of datasets."""
    g, counts = one_var_grammar(4)
    J = 10
    L = np.full((J, 4), -1000.0)
    L[:5, 0] = 0.0
    L[5:, 1] = 0.0
    res = run_gibbs(L, counts, g, GibbsConfig(n_sweeps=4000, burn_in=100, n_chains=1, seed=6))
    mean_v = res.theta["V"].mean(axis=(0, 1))
    expected = (np.ones(4) + np.array([1, 1, 0, 0])) / 6.0
    assert np.allclose(mean_v, expected, atol=0.02)


def test_population_weights_sum_below_one_and_track_counts():
    g, counts = one_var_grammar(5)
    J = 20
    L = np.full((J, 5), -1000.0)
    L[:15, 0] = 0.0
    L[15:, 1] = 0.0
    res = run_gibbs(L, counts, g, GibbsConfig(n_sweeps=500, burn_in=100, n_chains=2))
    w = res.pop_weight
    assert w.sum() <= 1.0 + 1e-9
    assert w[0] > 0.6 and w[1] > 0.15 and w[0] > w[1]


def test_rhat_helper():
    rng = np.random.default_rng(7)
    from hsr.gibbs import GibbsResult
    same = rng.normal(0, 1, (4, 1000))
    shifted = same + np.arange(4)[:, None] * 3
    r = GibbsResult.rhat(None, same)
    assert r < 1.01
    assert GibbsResult.rhat(None, shifted) > 1.5


def test_burn_in_validation():
    g, counts = one_var_grammar(2)
    with pytest.raises(ValueError):
        run_gibbs(np.zeros((2, 2)), counts, g, GibbsConfig(n_sweeps=10, burn_in=10))
