import numpy as np
import sympy as sp

from hsr.mechanisms import (CompartmentSystem, Mechanism, ThermalMechanism, X, compartment_form,
                            exponential_structure, family_string, flux_polys, generate_class, mechanism_form,
                            random_mechanism, thermal_form)


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


# ------------------------------------------------------------ compartments
def _cs(n, transfers, elims, observed, infusion=False):
    return CompartmentSystem(n_comp=n, transfers=transfers, eliminations=elims, observed=observed,
                             infusion=infusion)


def test_one_compartment_bolus_and_infusion():
    assert compartment_form(_cs(1, [], [(0, "k0")], 0))["key"] == "c0*exp(c1*x1)"
    rec = compartment_form(_cs(1, [], [(0, "k0")], 0, infusion=True))
    assert rec["key"] == "c0*exp(c1*x1) + c2"


def test_two_compartments_difference_of_exponentials():
    rec = compartment_form(_cs(2, [(0, 1, "k0")], [(1, "k1")], 1))
    assert rec["key"] == "c0*exp(c1*x1) + c2*exp(c3*x1)"


def test_closed_system_has_constant_term():
    rec = compartment_form(_cs(2, [(0, 1, "k0")], [], 1))
    assert rec["key"] == "c0*exp(c1*x1) + c2"


def test_trap_under_infusion_grows_linearly():
    rec = compartment_form(_cs(2, [(0, 1, "k0")], [(0, "k1")], 1, infusion=True))
    assert {"rate": "zero", "powers": [0, 1]} in rec["terms"]


def test_transit_chain_gives_t_exp_terms():
    # 0 -> 1 -> 2 with one shared rate k: the amount in 1 is k t exp(-k t)
    rec = compartment_form(_cs(3, [(0, 1, "k0"), (1, 2, "k0")], [(2, "k1")], 1))
    assert rec["key"] == "c0*x1*exp(c1*x1)"
    rec = compartment_form(_cs(3, [(0, 1, "k0"), (1, 2, "k0")], [(2, "k1")], 2))
    assert any(t["powers"] == [0, 1] for t in rec["terms"])


def test_cycle_can_oscillate():
    sysm = _cs(3, [(0, 1, "k0"), (1, 2, "k1"), (2, 0, "k2")], [], 1)
    _, real = exponential_structure(sysm.laplace({"k0": 1, "k1": 1, "k2": 1}))
    assert not real


# ------------------------------------------------------------- temperature
def _mm_thermal(saturating, eyring=False):
    return ThermalMechanism(_mm(), eyring=eyring, saturating=saturating)


def test_flux_polys_match_numeric_flux():
    m = _mm()
    N, D = flux_polys(m, X)
    vals = {"k0": 3, "k1": 5, "k2": 7}
    sub = {sp.Symbol(k, positive=True): v for k, v in vals.items()}
    assert sp.simplify(N.subs(sub) / D.subs(sub) - m.flux(vals)) == 0


def test_thermal_michaelis_menten():
    assert thermal_form(_mm_thermal(True))["key"] == "c0*exp(c1/x1)"        # kcat(T), Arrhenius
    assert thermal_form(_mm_thermal(True, eyring=True))["key"] == "c0*x1*exp(c1/x1)"
    fixed = thermal_form(_mm_thermal(False))
    assert (fixed["n_num_terms"], fixed["n_den_terms"]) == (1, 3)


def test_thermal_inactivation_adds_denominator_term():
    th = _mm_thermal(True)
    th.mech.n_states = 3
    th.mech.add(1, 2, False)      # inactivation of the bound enzyme matters at saturation
    th.mech.add(2, 1, False)
    rec = thermal_form(th)
    assert rec["n_den_terms"] >= 2 and "exp" in rec["key"]


def test_generate_classes_count_drops():
    for cls in ("compartment", "temperature"):
        recs, drops = generate_class(cls, 12, seed=1)
        assert len(recs) + sum(drops.values()) == 12
        assert all(r["n_consts"] <= 8 for r in recs)
