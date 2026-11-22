from collections import Counter

import numpy as np
import pytest
import sympy as sp

from hsr.candidates import CandidateSet, skeleton_from_key
from hsr.canonical import canonicalise, parse_skeleton, rule_counts, structural_size
from hsr.grammar import Grammar


def rc(s, canon=True):
    expr = canonicalise(s).expr if canon else parse_skeleton(s)
    return {k: v for k, v in rule_counts(expr).items() if v}


# ----------------------------------------------------------- rule counts
def test_counts_variable():
    assert rc("x1") == {("E", "F"): 1, ("F", "T"): 1, ("T", "V"): 1, ("V", "x1"): 1}


def test_counts_sum_of_three_terms():
    # x1 + x1*x1 + C: two E+F, one E->F
    c = rc("x1 + x1*x1 + C", canon=False)
    assert c[("E", "E+F")] == 2 and c[("E", "F")] == 1
    assert c[("F", "T")] == 3 and c[("F", "F*T")] == 1
    assert c[("T", "V")] == 3 and c[("T", "C")] == 1


def test_counts_difference():
    c = rc("x1 - x1*x1", canon=False)
    assert c[("E", "E-F")] == 1
    assert c.get(("E", "E+F"), 0) == 0
    assert c[("E", "F")] == 1


def test_counts_exponential_form():
    assert rc("C*exp(C*x1)") == {("E", "F"): 2, ("F", "T"): 2, ("F", "F*T"): 2, ("T", "C"): 2,
                                 ("T", "R(E)"): 1, ("R", "exp"): 1, ("T", "V"): 1, ("V", "x1"): 1}


def test_counts_saturating_form():
    c = rc("C*x1/(C+x1)")
    assert c == {("E", "F"): 2, ("F", "F*T"): 1, ("F", "F/T"): 1, ("F", "T"): 3, ("T", "C"): 2,
                 ("T", "V"): 2, ("V", "x1"): 2, ("T", "(E)"): 1, ("E", "E+F"): 1}
    assert structural_size(Counter(c)) == 13


def test_counts_rise_and_fall_form_has_size_15():
    c = rc("C*x1/(C+x1*x1)")
    assert c[("F", "F*T")] == 2  # C*x1 and x1*x1 (power 2 = one multiplication)
    assert structural_size(Counter(c)) == 15


def test_counts_integer_power():
    c = rc("x1*x1*x1", canon=False)  # x1**3: two multiplications
    assert c[("F", "F*T")] == 2 and c[("T", "V")] == 3


def test_counts_unary_minus_is_multiplication_by_constant():
    c = rc("-x1", canon=False)
    assert c[("F", "F*T")] == 1 and c[("T", "C")] == 1 and c[("T", "V")] == 1


def test_counts_sqrt_and_reciprocal():
    c = rc("sqrt(x1)", canon=False)
    assert c[("R", "sqrt")] == 1 and c[("T", "R(E)")] == 1
    c = rc("1/x1", canon=False)  # counted as C/x1
    assert c[("F", "F/T")] == 1 and c[("T", "C")] == 1


def test_counts_sum_inside_function_uses_R_E_directly():
    c = rc("exp(x1 + C)*x1", canon=False)
    assert c.get(("T", "(E)"), 0) == 0 and c[("E", "E+F")] == 1


def test_counts_sum_as_factor_needs_parentheses():
    c = rc("x1*(x1 + C)", canon=False)
    assert c[("T", "(E)")] == 1


# ----------------------------------------------------- canonicalisation
MERGES = [
    ["C*exp(C*x1)", "exp(C*x1)*C", "C*C*exp(x1*C)", "exp(C+C*x1)", "C*exp(C*x1-x1)", "exp(C*(x1+C))"],
    ["C*x1/(C+x1)", "x1*C/(x1+C)", "x1/(C*x1+C)", "C*x1/(C*x1+C)", "x1/(C+x1)/C", "C/(C+1/x1)",
     "C*x1/(C-x1)"],
    ["C*x1/(C+x1*x1)", "C*x1/(C+C*x1*x1)", "x1/(C*x1*x1+C)", "C/(C/x1+x1)", "C/(C/x1-x1)"],
    ["C*x1", "C*(C*x1)", "x1*C/C", "C*x1+C*x1", "-C*x1"],
    ["C+x1", "x1-C", "C+C+x1", "C-C+x1"],
    ["C*log(x1)+C", "C+C*log(x1)", "log(C*x1)*C", "C*(C+log(x1)/C)", "C*(C+log(sqrt(x1)))", "C*(C-log(x1))"],
    ["C*sqrt(x1)", "sqrt(C*x1)"],
    ["C*x1+C", "C*(x1+C)"],
    ["exp(C*x1)", "C**x1", "exp(x1*C+x1)"],
    ["2*x1", "x1+x1"],
    ["x1", "x1*x1/x1", "sqrt(x1)*sqrt(x1)"],
]


@pytest.mark.parametrize("group", MERGES, ids=[g[0] for g in MERGES])
def test_equivalent_forms_merge(group):
    keys = {canonicalise(s).key for s in group}
    assert len(keys) == 1, keys


DISTINCT = [
    ("C*x1", "C*x1+C"),
    ("x1", "C*x1"),
    ("exp(-x1)", "C*exp(C*x1)"),
    ("C-x1", "C+x1"),
    ("C*x1/(C+x1)", "C*x1/(C+x1*x1)"),
    ("(x1+C)*(x1+C)", "x1*x1+C*x1+C"),   # 2 vs 3 free constants
    ("2*x1", "C*x1"),                     # a literal is not a free constant
]


@pytest.mark.parametrize("a,b", DISTINCT)
def test_distinct_forms_do_not_merge(a, b):
    assert canonicalise(a).key != canonicalise(b).key


@pytest.mark.parametrize("s", ["C", "C*C", "exp(C)", "x1/x1", "x1-x1", "log(C)+C"])
def test_variable_free_skeletons_are_dropped(s):
    assert canonicalise(s) is None


def test_constant_cap():
    assert canonicalise("C*x1+C*x1*x1+C*exp(C*x1)", max_consts=4) is not None
    assert canonicalise("C*x1+C*sin(x1)+C*exp(x1)+C/x1+C*log(x1)", max_consts=4) is None


def test_constants_renamed_in_canonical_order():
    sk = canonicalise("x1*C/(x1+C)")
    assert sorted(c.name for c in sk.expr.free_symbols if c.name.startswith("c")) == ["c0", "c1"]
    assert sk.n_consts == 2
    assert sk.display == "C*x1/(C + x1)"


def test_repeated_constant_is_kept_shared():
    # x/(c*x + 1) = d*x/(x + d): one free constant appearing twice
    sk = canonicalise("x1/(C*x1+x1/x1)")
    assert sk.n_consts == 1


def _family_residual(src, dst, rng, n_tries=4):
    """Fit dst's constants to noiseless curves of src; return worst relative RMS misfit."""
    from hsr.likelihood import CompiledSkeleton, _lm
    a, b = CompiledSkeleton(src), CompiledSkeleton(dst)
    x = np.linspace(0.3, 3.0, 40)
    worst = 0.0
    for _ in range(n_tries):
        phi = rng.uniform(0.5, 1.5, a.k) * rng.choice([-1, 1], a.k)
        y = a.f(x[None, :, None], phi[None, :])[0]
        if not np.all(np.isfinite(y)) or np.ptp(y) < 1e-3:
            continue
        S = 30
        starts = rng.normal(0, 1.5, (S, b.k))
        X = np.broadcast_to(x[None, :, None], (S, 40, 1))
        Y = np.broadcast_to(y[None, :], (S, 40))
        M = np.ones((S, 40))
        phib, s, obj = _lm(b, X, Y, M, starts, np.full(S, 40.0), 3.0, False, 400, 1e-14)
        with np.errstate(all="ignore"):
            fit = b.f(X, phib)
            rms = np.sqrt(np.mean((fit - Y) ** 2, axis=1))
        worst = max(worst, np.nanmin(rms) / (np.std(y) + 1e-12))
    return worst


@pytest.mark.slow
def test_canonicalisation_preserves_function_family():
    """Validation: for random grammar derivations, the canonical skeleton can
    reproduce curves of the original expression (canonicalisation never
    loses a function)."""
    g = Grammar()
    rng = np.random.default_rng(4)
    checked = 0
    while checked < 25:
        out = g.sample(rng, max_size=11)
        if out is None or out[2] == 0 or out[2] > 2:
            continue
        sk = canonicalise(out[0])
        if sk is None:
            continue
        raw = parse_skeleton(out[0])
        from hsr.canonical import Skeleton
        src = Skeleton(key=str(raw), expr=raw, n_consts=len([c for c in raw.free_symbols if c.name.startswith("c")]))
        err = _family_residual(src, sk, rng)
        assert err < 1e-3, (out[0], sk.key, err)
        checked += 1


@pytest.fixture(scope="module")
def shipped():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "hsr", "data", "candidates.json")
    return CandidateSet.from_json(path)


def test_shipped_candidates_round_trip_and_are_idempotent(shipped):
    rng = np.random.default_rng(5)
    idx = rng.choice(len(shipped), 60, replace=False)
    for i in idx:
        sk = shipped.skeletons[i]
        assert skeleton_from_key(sk.key).key == sk.key
        again = canonicalise(sk.expr)
        assert again is not None and again.key == sk.key, (sk.key, again and again.key)


def test_shipped_candidates_contain_benchmark_forms(shipped):
    keys = set(shipped.keys)
    assert canonicalise("C*exp(C*x1)").key in keys
    assert canonicalise("C*x1/(C+x1)").key in keys
    assert all(s.n_consts <= 4 for s in shipped.skeletons)


def test_probe_merge_merges_reparameterisations_only():
    """c0/sqrt(sin(c1*x)) and c0*sqrt(1/sin(c1*x)) are the same function but
    the symbolic rules keep them apart; probe fitting must merge them and
    nothing else. (Families equal only up to a boundary of parameter space,
    such as c*x/(a + x) vs c/(a + 1/x), are deliberately not merged.)"""
    from hsr.candidates import from_skeletons, merge_by_probes
    g = Grammar()
    a = canonicalise("C/sqrt(sin(C*x1))")
    b = canonicalise("C*sqrt(1/sin(C*x1))")
    assert a.key != b.key
    others = [canonicalise(s) for s in ["C*x1/(C+x1*x1)", "C*exp(C*x1)", "C*x1+C", "C*sqrt(sin(C*x1))"]]
    cands = from_skeletons([a, b] + others, g)
    merged = merge_by_probes(cands, n_probes=4, n=25)
    assert len(merged) == 5
    assert merged.stats["n_probe_merged"] == 1
    assert (a.key in merged.keys) != (b.key in merged.keys)
