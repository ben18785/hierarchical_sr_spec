

def test_complete_pooling_and_single_curve_posteriors():
    import numpy as np
    from scipy.special import logsumexp
    from hsr import baselines
    from hsr.grammar import Grammar
    g = Grammar()
    rng = np.random.default_rng(0)
    counts = np.abs(rng.normal(2, 1, size=(4, g.n_rules))).round()
    L = rng.normal(0, 2, size=(5, 4))
    L[2, 3] = -np.inf                      # a failed fit rules the form out for everyone
    P = baselines.complete_pooling_posterior(L, counts, g)
    lw = g.log_prob(counts, g.prior_mean()) + L.sum(axis=0)
    assert np.allclose(P, np.exp(lw - logsumexp(lw))) and P[3] == 0 and abs(P.sum() - 1) < 1e-12
    P1 = baselines.single_curve_posterior(L[0], counts, g)
    lw1 = g.log_prob(counts, g.prior_mean()) + L[0]
    assert np.allclose(P1, np.exp(lw1 - logsumexp(lw1)))
