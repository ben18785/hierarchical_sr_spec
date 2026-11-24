from collections import Counter

import numpy as np

from hsr.candidates import skeleton_from_key
from hsr.canonical import rule_counts
from hsr.grammar import Grammar
from hsr.grammar_compare import context_counts, log_evidence, project


def test_context_counts_collapse_to_rule_counts():
    for key in ["c0*x1/(c1 + x1)", "c0*exp(c1*x1) + c2", "c0*x1**c1/(c2 + x1**c1)", "c0*sqrt(x1)*tanh(c1*x1)",
                "c0*log(c1 + exp(c2*x1))", "c0 - x1/c1"]:
        e = skeleton_from_key(key).expr
        cf = Counter({(nt, o): n for ((), nt, o), n in project(context_counts(e), "cf").items()})
        assert +cf == +rule_counts(e), key


def test_contexts_record_position():
    c = project(context_counts(skeleton_from_key("c0*x1/(c1 + x1)").expr), "term")
    assert c[(("root", "num"), "T", "V")] == 1           # the x in the numerator
    assert c[(("root", "num+"), "T", "V")] == 1          # the x in c1 + x1 (a term of a sum)
    assert c[(("root", "den"), "T", "(E)")] == 1         # the bracketed denominator


def test_cf_evidence_is_dirichlet_multinomial():
    from scipy.special import gammaln
    g = Grammar()
    items = [context_counts(skeleton_from_key(k).expr) for k in ["c0*x1", "c0*x1 + c1"]]
    tot = 0.0
    for nt in g.nonterminals:
        n = np.zeros(len(g.options[nt]))
        for c in items:
            for (_, nt2, o), m in project(c, "cf").items():
                if nt2 == nt:
                    n[g.options[nt].index(o)] += m
        if n.sum():
            a = g.alpha[nt]
            tot += gammaln(a.sum()) - gammaln(a.sum() + n.sum()) + np.sum(gammaln(a + n) - gammaln(a))
    assert np.isclose(log_evidence(items, "cf"), tot)
