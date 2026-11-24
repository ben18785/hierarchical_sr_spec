import numpy as np

from hsr.candidates import skeleton_from_key
from hsr.equivalence import Equivalence, class_summary
from hsr.likelihood import compute_likelihoods


def _setup():
    rng = np.random.default_rng(0)
    J, n = 4, 15
    X = np.tile(np.linspace(0.1, 3, n), (J, 1))[..., None]
    Y = np.array([2 * np.exp(-0.7 * X[j, :, 0]) + rng.normal(0, 0.02, n) for j in range(J)])
    keys = ["c0*exp(c1*x1)", "c0*exp(c1*x1 + 1)", "c0*x1 + c1", "c0/(c1 + x1)"]
    sks = [skeleton_from_key(k) for k in keys]
    table = compute_likelihoods(sks, X, Y, n_starts=6, seed=0)
    return sks, table, X


def test_identical_functions_are_equivalent_and_distinct_ones_are_not():
    sks, table, X = _setup()
    eq = Equivalence(sks, table, X)
    assert eq.equivalent(0, 1)          # e^1 is absorbed by c0: the same function family
    assert not eq.equivalent(0, 2)      # a straight line cannot follow an exponential decay
    probs = np.tile([0.5, 0.3, 0.2, 0.0], (4, 1))
    cls = eq.classes(probs)
    assert cls[0] == cls[1] == 0 and cls[2] == 2


def test_class_summary_sums_members():
    sks, table, X = _setup()
    probs = np.tile([0.5, 0.3, 0.2, 0.0], (4, 1))
    z = np.array([[[0, 1, 1, 2], [1, 1, 0, 0]]])            # 1 chain, 2 draws, 4 datasets
    out = class_summary(probs, {0: 0, 1: 0, 2: 2}, sks, z_samples=z, true_idx=[1, 1, 2, 0])
    assert out["top_class_keys"] == ["c0*exp(c1*x1)"] * 4
    assert out["top_class_prob"] == [0.8] * 4
    assert out["class_members"] == {"c0*exp(c1*x1)": ["c0*exp(c1*x1)", "c0*exp(c1*x1 + 1)"]}
    assert out["n_classes_posterior"] == {1: 0.5, 2: 0.5}
    assert out["class_recovery"] == 0.75
    assert out["class_coassignment"][0][3] == 0.5


def test_end_behaviour_labels():
    from hsr.equivalence import end_behaviour
    from hsr.likelihood import CompiledSkeleton
    cases = [("c0*x1/(c1 + x1)", [2.0, 1.0], "levels_off", "zero"),
             ("c0*exp(c1*x1)", [2.0, -0.7], "to_zero", "nonzero"),
             ("c0*exp(c1*x1)", [2.0, 0.7], "unbounded", "nonzero"),
             ("c0*x1**c1", [1.0, 0.5], "unbounded", "zero"),
             ("c0*log(x1)", [1.0], "unbounded", "unbounded"),
             ("c0*sin(x1)", [1.0], "other", "zero")]
    for key, phi, right, left in cases:
        r, l = end_behaviour(CompiledSkeleton(skeleton_from_key(key)), [phi], 0.1, 3.0, 1.0)
        assert (r[0], l[0]) == (right, left), key


def test_in_range_lookalikes_with_different_ends_are_kept_apart():
    """A power law fits saturating data on a short range, but grows without
    bound where Michaelis-Menten levels off: they must not share a class."""
    rng = np.random.default_rng(1)
    J, n = 4, 12
    X = np.tile(np.linspace(0.5, 2.0, n), (J, 1))[..., None]
    Y = np.array([3 * X[j, :, 0] / (4 + X[j, :, 0]) + rng.normal(0, 0.02, n) for j in range(J)])
    sks = [skeleton_from_key(k) for k in ["c0*x1/(c1 + x1)", "c0*x1**c1"]]
    table = compute_likelihoods(sks, X, Y, n_starts=6, seed=0)
    eq = Equivalence(sks, table, X)
    assert eq.end_labels(0, 0)[0].get("levels_off", 0) > 0.5
    assert eq.end_labels(1, 0)[0].get("unbounded", 0) > 0.5
    assert not eq.equivalent(0, 1)
    es = eq.end_summary(np.tile([0.5, 0.5], (J, 1)))
    assert abs(es["end_right"][0].get("levels_off", 0) - 0.5) < 0.2
