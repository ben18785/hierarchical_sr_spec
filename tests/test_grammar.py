import numpy as np
import pytest
import sympy as sp

from hsr.canonical import parse_skeleton, rule_counts, structural_size
from hsr.grammar import Grammar


@pytest.fixture
def g():
    return Grammar()


def test_prior_mean_normalised(g):
    for nt, p in g.prior_mean().items():
        assert np.isclose(p.sum(), 1.0)
        assert np.allclose(p, g.alpha[nt] / g.alpha[nt].sum())


def test_rule_indexing_is_flat_and_complete(g):
    assert g.n_rules == 3 + 3 + 4 + 4 + 1
    assert sorted(g.rule_index.values()) == list(range(g.n_rules))


def test_bad_alpha_length_raises():
    with pytest.raises(ValueError):
        Grammar(alpha={"E": [1.0, 1.0]})


def test_log_prob_hand_example_x(g):
    th = g.prior_mean()
    v = g.counts_vector(rule_counts(parse_skeleton("x1")))
    expected = np.log(th["E"][2]) + np.log(th["F"][2]) + np.log(th["T"][2]) + np.log(th["V"][0])
    assert np.isclose(g.log_prob(v, th), expected)


def test_log_prob_hand_example_exp(g):
    # C*exp(C*x1): E->F x2, F->F*T x2, F->T x2, T->C x2, T->R(E), R->exp, T->V, V->x1
    th = g.prior_mean()
    v = g.counts_vector(rule_counts(parse_skeleton("C*exp(C*x1)")))
    E, F, T, R = th["E"], th["F"], th["T"], th["R"]
    expected = (2 * np.log(E[2]) + 2 * np.log(F[0]) + 2 * np.log(F[2]) + 2 * np.log(T[3])
                + np.log(T[1]) + np.log(R[0]) + np.log(T[2]))
    assert np.isclose(g.log_prob(v, th), expected)


def test_log_prob_vectorised(g):
    th = g.prior_mean()
    rows = np.array([g.counts_vector(rule_counts(parse_skeleton(s))) for s in ["x1", "C*x1", "x1+C"]])
    lp = g.log_prob(rows, th)
    assert lp.shape == (3,)
    assert lp[0] > lp[1] and lp[0] > lp[2]


def test_sample_respects_size_cap(g):
    rng = np.random.default_rng(0)
    for _ in range(500):
        out = g.sample(rng, max_size=8)
        if out is not None:
            assert out[1] <= 8


def test_sample_strings_parse_and_sizes_match(g):
    """For parenthesis-free derivations the sympy tree has the same structure,
    so the rule counter must reproduce the sampler's size."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(3000):
        out = g.sample(rng, max_size=10)
        if out is None:
            continue
        s, size, nc = out
        expr = parse_skeleton(s)
        assert s.count("C") == nc
        if "(" in s or "-" in s or "/" in s:
            continue
        expr = sp.sympify(s.replace("C", "c"), locals={"x1": sp.Symbol("x1", positive=True)},
                          evaluate=False)
        assert structural_size(rule_counts(expr)) == size, s
        checked += 1
    assert checked > 50


def test_sample_frequency_matches_probability(g):
    """Empirical frequency of the derivation 'x1' equals P_theta(x1)."""
    rng = np.random.default_rng(2)
    th = g.prior_mean()
    n = 20000
    hits = sum(1 for _ in range(n) if (o := g.sample(rng, th, max_size=40)) is not None and o[0] == "x1")
    p = th["E"][2] * th["F"][2] * th["T"][2]
    assert abs(hits / n - p) < 4 * np.sqrt(p * (1 - p) / n)


def test_enumeration_contains_small_forms_and_respects_constant_cap(g):
    out = g.enumerate(6, max_consts=1)
    for s in ["x1", "C*x1", "x1+C", "x1*x1", "x1/C", "exp(x1)"]:
        assert s in out
    assert all(s.count("C") <= 1 for s in out)
    assert "C*C" not in out  # variable-free composites pruned


def test_enumeration_count_without_pruning_matches_recursion():
    """With a constant-free grammar (no pruning applies) the number of
    derivations of each size follows the CFG recursion."""
    g = Grammar(alpha={"T": [0.5, 1, 3, 2]})
    out = g.enumerate(7, max_consts=0)
    # sizes 3: x1 ; 5: x1*x1, x1/x1, x1+x1, x1-x1 ; 6: (x1) is pruned? no: (x1) has a var -> kept
    # count with the recursion restricted to constant-free strings
    N = 7
    T = [0] * (N + 1); F = [0] * (N + 1); E = [0] * (N + 1)
    for s in range(1, N + 1):
        T[s] = (1 if s == 1 else 0) + 5 * E[s - 1]
        F[s] = T[s - 1] + 2 * sum(F[a] * T[s - 1 - a] for a in range(1, s - 1))
        E[s] = F[s - 1] + 2 * sum(E[a] * F[s - 1 - a] for a in range(1, s - 1))
    assert len(out) == sum(E[: N + 1])


def test_dirichlet_update_concentrates(g):
    rng = np.random.default_rng(3)
    counts = np.zeros(g.n_rules)
    counts[g.rule_index[("R", "sin")]] = 1e5
    th = g.sample_theta(rng, counts)
    assert th["R"][2] > 0.99
