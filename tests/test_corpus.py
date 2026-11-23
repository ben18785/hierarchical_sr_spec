"""Corpus reduction and the corpus-informed grammar prior (no network)."""
import numpy as np

from hsr.corpus import _to_sympy, corpus_alpha, curated_keys, project, rule_totals
from hsr.grammar import Grammar


def test_reduce_kinetic_laws():
    expr, syms = _to_sympy("comp1 * Vmax * S / (Km + S)")
    assert project(expr, ["S"], syms) == {"c0*x1/(c1 + x1)"}
    expr, syms = _to_sympy("k1 * A * B - k2 * C")
    assert project(expr, ["A", "B", "C"], syms) == {"c0*x1 + c1"}
    # SBML names that clash with sympy are safe, and power uses ^
    expr, syms = _to_sympy("beta * S^n / (K^n + S^n)")
    assert project(expr, ["S"], syms) == {"c0*x1**c1/(c2 + x1**c1)"}
    assert _to_sympy("piecewise(1, x > 2, 0)") is None


def test_curated_forms_reduce():
    cur = curated_keys()
    assert cur["a*x/(1 + a*h*x)"] == ["c0*x1/(c1 + x1)"]
    assert cur["a*exp(-(x/l)**k)"] == ["c0*exp(c1*x1**c2)"]


def test_corpus_alpha_follows_frequencies():
    g = Grammar()
    freq = {"c0*x1/(c1 + x1)": 10, "c0*tanh(c1*x1)": 1}
    tot, used = rule_totals(freq, g)
    assert used == 11
    alpha = corpus_alpha(freq, g, kappa=100, floor=0.1)
    R = dict(zip(g.options["R"], alpha["R"]))
    assert R["tanh"] > R["exp"]           # only tanh is used in this toy corpus
    for nt in g.nonterminals:
        assert np.all(alpha[nt] >= 0.1)
        assert abs(alpha[nt].sum() - (100 + 0.1 * len(alpha[nt]))) < 1e-9
