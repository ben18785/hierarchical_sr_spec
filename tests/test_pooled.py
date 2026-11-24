import itertools

import numpy as np
import pytest
from scipy import integrate, special, stats

from hsr.gibbs import GibbsConfig, run_gibbs
from hsr.grammar import Grammar
from hsr.likelihood import LikelihoodTable
from hsr.pooled import IndependentModel, PooledModel, grid_log_weights, member_terms
from tests.test_gibbs import one_var_grammar


def fake_table(L, phis, vars_):
    """LikelihoodTable from per-(dataset, candidate) Gaussian summaries.
    phis[e]: (J, k); vars_[e]: (J, k) diagonal variances."""
    J, C = L.shape
    covs = []
    for e in range(C):
        k = phis[e].shape[1]
        cov = np.zeros((J, k + 1, k + 1))
        for j in range(J):
            cov[j, :k, :k] = np.diag(vars_[e][j])
            cov[j, k, k] = 0.01
        covs.append(cov)
    return LikelihoodTable(L, L.copy(), phis, np.zeros((J, C)), covs)


def test_member_term_reduces_to_L_when_population_equals_reference_prior():
    """With mu = 0 and s = tau_ref the population prior is the reference
    prior, so log p(y | e, mu, s) must equal L."""
    rng = np.random.default_rng(0)
    B, k = 5, 2
    L = rng.normal(0, 3, B)
    m = rng.normal(0, 1, (B, k))
    R = rng.normal(0, 0.3, (B, k, k))
    V = np.einsum("bij,bkj->bik", R, R) + 0.05 * np.eye(k)
    A, a, c = member_terms(L, m, V, s=3.0, tau_ref=3.0)
    assert np.allclose(c, L)


def test_member_term_matches_numerical_integral():
    L, m, v, s, tau_ref, mu = -2.0, 0.7, 0.2, 0.5, 3.0, 0.3
    A, a, c = member_terms(np.array([L]), np.array([[m]]), np.array([[[v]]]), s, tau_ref)
    closed = -0.5 * A[0, 0, 0] * mu**2 + a[0, 0] * mu + c[0]

    def f(phi):
        return (np.exp(L) * stats.norm.pdf(phi, m, np.sqrt(v)) / stats.norm.pdf(phi, 0, tau_ref)
                * stats.norm.pdf(phi, mu, s))
    val, _ = integrate.quad(f, -10, 10, limit=200)
    assert np.isclose(closed, np.log(val), atol=1e-8)


def test_cluster_marginal_matches_brute_force():
    """M_e(S) for k = 1 against nested numerical integration over mu and the s grid."""
    rng = np.random.default_rng(1)
    J = 3
    L = rng.normal(0, 1, (J, 1))
    phis = [rng.normal(1.0, 0.3, (J, 1))]
    vars_ = [rng.uniform(0.05, 0.3, (J, 1))]
    tab = fake_table(L, phis, vars_)
    model = PooledModel(tab, [1], tau_ref=3.0, tau_mu=3.0)
    logw = grid_log_weights(model.grid)
    total = []
    for g, s in enumerate(model.grid):
        def integrand(mu):
            out = stats.norm.pdf(mu, 0, 3.0)
            for j in range(J):
                A, a, c = member_terms(L[j], phis[0][j:j + 1], np.array([[vars_[0][j]]]), s, 3.0)
                out *= np.exp(-0.5 * A[0, 0, 0] * mu**2 + a[0, 0] * mu + c[0])
            return out
        val, _ = integrate.quad(integrand, -15, 15, limit=400, points=[1.0])
        total.append(logw[g] + np.log(val))
    assert np.isclose(model.marginal(0, [0, 1, 2]), special.logsumexp(total), atol=1e-6)
    assert np.isclose(model.marginal_all([0, 1, 2])[0], model.marginal(0, [0, 1, 2]))
    assert np.allclose(model.singletons()[:, 0], [model.marginal(0, [j]) for j in range(J)])


def test_independent_model_is_sum_of_L():
    L = np.array([[1.0, -np.inf], [2.0, 0.5]])
    m = IndependentModel(L)
    assert m.marginal(0, [0, 1]) == 3.0
    assert m.marginal(1, [0, 1]) == -np.inf
    assert np.allclose(m.marginal_all([1]), [2.0, 0.5])


def _exact_posterior(model, logp, gamma, J, C):
    probs = {}
    for z in itertools.product(range(C), repeat=J):
        lw = 0.0
        n = np.zeros(C)
        for i, e in enumerate(z):
            lw += np.log(n[e] + gamma * np.exp(logp[e])) - np.log(gamma + i)
            n[e] += 1
        for e in set(z):
            lw += model.marginal(e, [j for j in range(J) if z[j] == e])
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
def test_sampler_with_pooled_model_matches_enumeration(relabel):
    """Validation: with the pooled model, Gibbs marginals equal the exact
    posterior P(z) ∝ urn(z) prod_e exp(M_e(S_e))."""
    rng = np.random.default_rng(2)
    g, counts = one_var_grammar(3)
    J = 4
    L = rng.normal(0, 1, (J, 3))
    phis = [rng.normal(1, 0.5, (J, 1)), rng.normal(-1, 0.5, (J, 2)), np.zeros((J, 0))]
    vars_ = [rng.uniform(0.05, 0.5, (J, 1)), rng.uniform(0.05, 0.5, (J, 2)), np.zeros((J, 0))]
    model = PooledModel(fake_table(L, phis, vars_), [1, 2, 0])
    gamma = 0.8
    cfg = GibbsConfig(n_sweeps=12000, burn_in=500, n_chains=2, update_theta=False, update_gamma=False,
                      gamma_init=gamma, relabel=relabel, seed=5)
    res = run_gibbs(L, counts, g, cfg, model=model)
    exact = _exact_posterior(model, g.log_prob(counts, g.prior_mean()), gamma, J, 3)
    assert np.max(np.abs(res.assign_probs - exact)) < 0.02


def test_pooling_rescues_a_shared_constant():
    """Ten datasets each weakly favour a constant-free skeleton (0) over a
    one-constant skeleton (1) whose constant is about 1.5 in every dataset.
    Independently each pays the Occam penalty for its constant; pooled, the
    shared constant is learned once and skeleton 1 wins."""
    J = 10
    rng = np.random.default_rng(3)
    # skeleton 1: posterior under N(0, 3^2) prior centred near 1.5 with sd 0.3
    phis = [np.zeros((J, 0)), 1.5 + rng.normal(0, 0.05, (J, 1))]
    vars_ = [np.zeros((J, 0)), np.full((J, 1), 0.3**2)]
    # L for skeleton 1 = Laplace evidence under N(0,9): fit gain 1.5 nats minus Occam ~ log(3/0.3) + 0.5*1.5^2/9
    L1 = 1.5 - np.log(3 / 0.3) - 0.5 * 1.5**2 / 9
    L = np.column_stack([np.zeros(J), np.full(J, L1)])
    tab = fake_table(L, phis, vars_)
    g, counts = one_var_grammar(2)
    cfg = GibbsConfig(n_sweeps=400, burn_in=100, n_chains=2, update_theta=False)
    ind = run_gibbs(L, counts, g, cfg)
    pooled = run_gibbs(L, counts, g, cfg, model=PooledModel(tab, [0, 1]))
    assert ind.assign_probs[:, 0].mean() > 0.9
    assert pooled.assign_probs[:, 1].mean() > 0.9


def test_posterior_mu_recovers_shared_mean():
    J = 20
    phis = [np.full((J, 1), 2.0)]
    vars_ = [np.full((J, 1), 0.1)]
    model = PooledModel(fake_table(np.zeros((J, 1)), phis, vars_), [1])
    mean, sd, wg = model.posterior_mu(0, list(range(J)))
    assert abs(mean[0] - 2.0) < 0.05 and sd[0] < 0.1
    assert np.isclose(wg.sum(), 1.0)
    assert wg[model.grid <= 0.25].sum() > 0.5   # identical constants: small spread favoured


def test_non_identified_skeleton_is_not_pooled():
    """A skeleton with a direction the data barely constrain (posterior variance
    close to the reference prior's) in every dataset gets M_e(S) = sum_j L_je,
    so it cannot gain from pooling; a well-identified one still can."""
    J = 12
    rng = np.random.default_rng(4)
    phis = [1.0 + rng.normal(0, 0.02, (J, 2)), 1.0 + rng.normal(0, 0.02, (J, 2))]
    vars_ = [np.tile([0.01, 0.01], (J, 1)), np.tile([0.01, 8.0], (J, 1))]
    L = np.full((J, 2), -3.0)
    pm = PooledModel(fake_table(L, phis, vars_), [2, 2])
    assert list(pm.unpooled) == [False, True]
    members = list(range(J))
    assert pm.marginal(1, members) == pytest.approx(L[:, 1].sum())
    assert pm.marginal_all(members)[1] == pytest.approx(L[:, 1].sum())
    assert np.allclose(pm.singletons()[:, 1], L[:, 1])
    assert pm.marginal(0, members) > L[:, 0].sum() + 10


def test_split_merge_with_pooled_model_matches_enumeration():
    """Split-merge and relabel only (no single-site updates) with the pooled
    model: marginals equal the exact posterior."""
    rng = np.random.default_rng(4)
    g, counts = one_var_grammar(3)
    J = 4
    L = rng.normal(0, 1, (J, 3))
    phis = [rng.normal(1, 0.5, (J, 1)), rng.normal(-1, 0.5, (J, 2)), np.zeros((J, 0))]
    vars_ = [rng.uniform(0.05, 0.5, (J, 1)), rng.uniform(0.05, 0.5, (J, 2)), np.zeros((J, 0))]
    model = PooledModel(fake_table(L, phis, vars_), [1, 2, 0])
    gamma = 0.8
    cfg = GibbsConfig(n_sweeps=12000, burn_in=500, n_chains=2, update_theta=False, update_gamma=False,
                      gamma_init=gamma, relabel=True, split_merge=4, single_site=False, seed=7)
    res = run_gibbs(L, counts, g, cfg, model=model)
    exact = _exact_posterior(model, g.log_prob(counts, g.prior_mean()), gamma, J, 3)
    assert np.max(np.abs(res.assign_probs - exact)) < 0.02
