"""Scale-invariance (units) filter."""
from hsr.candidates import edit_graph, from_skeletons, skeleton_from_key
from hsr.grammar import Grammar
from hsr.properties import scale_invariant


def test_scale_invariance_classifies_known_forms():
    yes = ["c0*x1", "c0*exp(c1*x1) + c2", "c0*x1**c1/(c2 + x1**c1)", "c0*x1**4/(c1 + x1**4)",
           "c0*log(x1) + c1", "c0/(c1 + exp(c2*x1))", "-c0*exp(c1*x1) + c0*exp(c2*x1)", "c0*exp(c1*x1**c2)"]
    no = ["x1", "sin(x1)", "c0*tanh(3*x1)", "(c0 + 3*x1)**c1", "c0*tanh(sqrt(x1))**c1", "x1**c0",
          "exp(c0*x1)", "c0*log(c1 + x1)", "c0*exp(x1**2 + x1)"]
    for k in yes:
        assert scale_invariant(skeleton_from_key(k)), k
    for k in no:
        assert not scale_invariant(skeleton_from_key(k)), k


def test_admit_filter_applies_to_growth():
    g = Grammar()
    c = from_skeletons([skeleton_from_key("c0*x1/(c1 + x1)")], g)
    c.admit = scale_invariant
    new, edges = edit_graph(c, [0])
    assert new and all(scale_invariant(c.skeletons[i]) for i in new)
    assert all(0 <= a < len(c.keys) and 0 <= b < len(c.keys) for a, b in edges)
