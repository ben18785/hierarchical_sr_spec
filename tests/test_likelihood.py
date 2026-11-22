import numpy as np
import pytest
from scipy import integrate, stats

from hsr.canonical import canonicalise
from hsr.likelihood import (CompiledSkeleton, compute_likelihoods, fit_skeleton, log_joint,
                            log_ml_importance, predictive_logpdf, scale_dataset, scale_population)


def compiled(s):
    return CompiledSkeleton(canonicalise(s))


def make_data(rng, J=4, n=20, sd=0.05):
    x = rng.uniform(0.1, 3, (J, n))
    a = rng.uniform(1, 2, J)
    b = -rng.uniform(0.5, 1, J)
    y = a[:, None] * np.exp(b[:, None] * x) + rng.normal(0, sd, (J, n))
    return x, y, a, b


def test_scale_dataset_preserves_skeleton_family():
    x = np.linspace(0.1, 5, 30)
    y = 3 * np.exp(-0.5 * x)
    xs, ys, sx, sy = scale_dataset(x, y)
    assert np.isclose(np.sqrt(np.mean(ys**2)), 1.0)
    # same form with rescaled constants: y/sy = (3/sy) exp(-0.5*sx * x/sx)
    assert np.allclose(ys, (3 / sy) * np.exp(-0.5 * sx[0] * xs[:, 0]))


def test_scale_population_modes():
    xs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    ys = [np.array([1.0, 1.0]), np.array([10.0, 10.0])]
    X, Y, sx, sy = scale_population(xs, ys, "pooled")
    assert np.isclose(sy[0], sy[1])
    assert np.isclose(np.sqrt(np.mean(np.concatenate(Y) ** 2)), 1.0)
    X, Y, sx, sy = scale_population(xs, ys, "none")
    assert sy == [1.0, 1.0] and np.allclose(Y[1], 10.0)
    X, Y, sx, sy = scale_population(xs, ys, "per_dataset")
    assert np.allclose(Y[0], 1.0) and np.allclose(Y[1], 1.0)
    with pytest.raises(ValueError):
        scale_population(xs, ys, "bogus")


def test_gradient_and_hessian_match_finite_differences():
    cs = compiled("C*x1/(C+x1*x1)")
    rng = np.random.default_rng(0)
    X = rng.uniform(0.2, 2, (1, 7, 1))
    phi = np.array([[1.3, 0.7]])
    f, J = cs.f_and_grad(X, phi)
    H = cs.hess(X, phi)
    h = 1e-6
    for i in range(2):
        d = np.zeros((1, 2)); d[0, i] = h
        fp, Jp = cs.f_and_grad(X, phi + d)
        fm, Jm = cs.f_and_grad(X, phi - d)
        assert np.allclose((fp - fm) / (2 * h), J[..., i], atol=1e-6)
        assert np.allclose((Jp - Jm) / (2 * h), H[..., i, :], atol=1e-5)


def test_map_recovers_parameters():
    rng = np.random.default_rng(1)
    x, y, a, b = make_data(rng, sd=0.01, n=40)
    res = fit_skeleton(compiled("C*exp(C*x1)"), x, y)
    assert np.allclose(res.phi[:, 0], a, rtol=0.03)
    assert np.allclose(res.phi[:, 1], b, rtol=0.05)
    assert np.allclose(np.exp(res.log_sigma), 0.01, rtol=0.5)
    assert np.all(np.isfinite(res.log_ml))


def test_true_form_beats_wrong_forms_with_enough_data():
    rng = np.random.default_rng(2)
    x, y, *_ = make_data(rng, n=40, sd=0.02)
    lml = {s: fit_skeleton(compiled(s), x, y).log_ml for s in ["C*exp(C*x1)", "C*x1+C", "C*x1/(C+x1)", "C*x1"]}
    for s, v in lml.items():
        if s != "C*exp(C*x1)":
            assert np.all(lml["C*exp(C*x1)"] > v), s


def test_bic_equals_closed_form_for_linear_model():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, (2, 15))
    y = 2 * x + rng.normal(0, 0.1, x.shape)
    res = fit_skeleton(compiled("C*x1"), x, y)
    for j in range(2):
        beta = np.sum(x[j] * y[j]) / np.sum(x[j] ** 2)
        s2 = np.mean((y[j] - beta * x[j]) ** 2)
        ll = np.sum(stats.norm.logpdf(y[j], beta * x[j], np.sqrt(s2)))
        assert np.isclose(res.max_loglik[j], ll, atol=1e-6)
        assert np.isclose(res.bic[j], ll - 1.0 * np.log(15), atol=1e-6)


def test_zero_constant_skeleton():
    rng = np.random.default_rng(4)
    x = rng.uniform(0.1, 2, (3, 10))
    y = np.sqrt(x) + rng.normal(0, 0.1, x.shape)
    res = fit_skeleton(compiled("sqrt(x1)"), x, y)
    assert res.phi.shape == (3, 0)
    assert np.all(np.isfinite(res.log_ml))


def test_invalid_evaluations_give_minus_inf():
    x = np.linspace(0.1, 2, 10)[None, :]
    y = np.ones_like(x)
    cs = compiled("sqrt(x1 - x1*x1*x1)")   # negative for x > 1: NaN, no constants to rescue it
    res = fit_skeleton(cs, x, y)
    assert res.log_ml[0] == -np.inf and res.bic[0] == -np.inf


def test_masking_matches_unpadded_fit():
    rng = np.random.default_rng(5)
    x, y, *_ = make_data(rng, J=2, n=12)
    cs = compiled("C*exp(C*x1)")
    full = fit_skeleton(cs, x[:1, :8], y[:1, :8])
    M = np.zeros((1, 12)); M[0, :8] = 1
    padded = fit_skeleton(cs, x[:1], y[:1], M)
    assert np.isclose(full.log_ml[0], padded.log_ml[0], atol=1e-6)


def _quadrature_log_ml(cs, x, y, mode, cov):
    """2-D quadrature of exp(log joint) over (phi, s) in a box around the mode."""
    sd = np.sqrt(np.diag(cov))
    ref = log_joint(cs, x, y, mode[None, :])[0]

    def integrand(s, phi):
        return np.exp(log_joint(cs, x, y, np.array([[phi, s]]))[0] - ref)

    val, err = integrate.dblquad(integrand, mode[0] - 10 * sd[0], mode[0] + 10 * sd[0],
                                 mode[1] - 10 * sd[1], mode[1] + 10 * sd[1], epsabs=1e-10, epsrel=1e-8)
    return np.log(val) + ref


@pytest.mark.slow
@pytest.mark.parametrize("n", [5, 10, 40])
def test_laplace_and_importance_sampling_against_quadrature(n):
    """Validation (spec section 8): compare Laplace with brute-force
    integration. Importance sampling must match quadrature tightly; Laplace
    is allowed a looser tolerance that shrinks with n."""
    rng = np.random.default_rng(10 + n)
    x = rng.uniform(0.1, 2, n)
    y = 1.5 * x + rng.normal(0, 0.3, n)
    cs = compiled("C*x1")
    res = fit_skeleton(cs, x[None], y[None])
    mode = np.r_[res.phi[0], res.log_sigma[0]]
    exact = _quadrature_log_ml(cs, x, y, mode, res.cov[0])
    est, se, ess = log_ml_importance(cs, x, y, mode, res.cov[0], n_draws=40000)
    assert abs(est - exact) < max(4 * se, 0.01)
    tol = {5: 0.35, 10: 0.15, 40: 0.05}[n]
    assert abs(res.log_ml[0] - exact) < tol, (res.log_ml[0], exact)


@pytest.mark.slow
def test_laplace_vs_importance_sampling_nonlinear():
    rng = np.random.default_rng(11)
    x, y, *_ = make_data(rng, J=3, n=20, sd=0.05)
    cs = compiled("C*exp(C*x1)")
    res = fit_skeleton(cs, x, y)
    for j in range(3):
        mode = np.r_[res.phi[j], res.log_sigma[j]]
        est, se, ess = log_ml_importance(cs, x[j], y[j], mode, res.cov[j], n_draws=20000, seed=j)
        assert ess > 1000
        assert abs(est - res.log_ml[j]) < 0.5


def test_predictive_logpdf_plug_in_matches_normal():
    cs = compiled("C*x1")
    xn = np.array([1.0, 2.0])
    yn = np.array([2.1, 3.9])
    lp = predictive_logpdf(cs, xn, yn, np.array([2.0]), np.log(0.5), np.full((2, 2), np.nan),
                           np.array([1.0]), 1.0)
    assert np.allclose(lp, stats.norm.logpdf(yn, 2 * xn, 0.5))
    # rescaling: data in units where x and y were divided by 2
    lp2 = predictive_logpdf(cs, xn, yn, np.array([2.0]), np.log(0.25), np.full((2, 2), np.nan),
                            np.array([2.0]), 2.0)
    assert np.allclose(lp2, stats.norm.logpdf(yn, 2 * xn, 0.5))


def test_compute_likelihoods_table_shapes():
    rng = np.random.default_rng(6)
    x, y, *_ = make_data(rng, J=3, n=10)
    sks = [canonicalise(s) for s in ["x1", "C*x1", "C*exp(C*x1)"]]
    tab = compute_likelihoods(sks, x, y)
    assert tab.laplace.shape == (3, 3) and tab.bic.shape == (3, 3)
    assert tab.phi[2].shape == (3, 2) and tab.cov[2].shape == (3, 3, 3)
    both = tab.extend(tab)
    assert both.laplace.shape == (3, 6) and len(both.phi) == 6
