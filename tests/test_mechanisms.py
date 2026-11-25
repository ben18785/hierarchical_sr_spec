import numpy as np

from hsr.mechanisms import Mechanism, family_string, mechanism_form, random_mechanism


def _mm():
    # E + S <-> ES -> E + P
    m = Mechanism(n_states=2)
    m._reverse = {}
    m.add(0, 1, True)
    m.add(1, 0, False)
    rel = m.add(1, 0, False)
    m.release = [rel]
    return m


def test_michaelis_menten():
    rec = mechanism_form(_mm())
    assert rec["key"] == "c0*x1/(c1 + x1)"
    assert rec["num_support"] == [1] and rec["den_support"] == [0, 1]


def test_substrate_inhibition_dead_end():
    m = _mm()
    # ES + S <-> ESS (dead end)
    m.n_states = 3
    m.add(1, 2, True)
    m.add(2, 1, False)
    rec = mechanism_form(m)
    assert rec["num_support"] == [1] and rec["den_support"] == [0, 1, 2]


def test_family_string_fixes_top_denominator_coefficient():
    assert family_string([1], [0, 1]) == "(C*x1)/(C + x1)"


def test_random_mechanisms_give_forms():
    rng = np.random.default_rng(3)
    keys = [mechanism_form(random_mechanism(rng), rng) for _ in range(10)]
    assert sum(k is not None for k in keys) >= 8
