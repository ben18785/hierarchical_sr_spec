import numpy as np

from hsr import baselines
from hsr.metrics import (adjusted_rand_index, expected_calibration_error, form_recovery,
                         reliability_curve, reliability_pairs, weight_error)
from tests.test_gibbs import one_var_grammar


def test_ari_known_values():
    assert adjusted_rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == 1.0
    assert np.isclose(adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]), -0.5)
    # hand-computed: (2 - 1.2) / (4.5 - 1.2)
    a = [0, 0, 0, 1, 1, 1]
    b = [0, 0, 1, 1, 2, 2]
    assert np.isclose(adjusted_rand_index(a, b), 0.24242424242424246)
    assert adjusted_rand_index([0, 0, 0], [1, 1, 1]) == 1.0


def test_recovery_and_weight_error():
    assert form_recovery(["a", "b", "c"], ["a", "x", "c"]) == 2 / 3
    assert np.isclose(weight_error({"a": 0.5, "b": 0.5}, ["a", "b", "c"], [0.6, 0.3, 0.1]),
                      (0.1 + 0.2 + 0.1) / 3)


def test_reliability_pairs_include_true_even_if_small():
    P = np.array([[0.9, 0.1, 0.0], [0.5, 0.4999, 0.0001]])
    p, y = reliability_pairs(P, [0, 2])
    assert y.sum() == 2
    assert 0.0001 in np.round(p, 4)
    curve = reliability_curve([0.05, 0.95, 0.95], [0, 1, 1], bins=10)
    assert curve[0] == (0.05, 0.0, 1) and curve[-1] == (0.95, 1.0, 2)
    assert np.isclose(expected_calibration_error([0.05, 0.95, 0.95], [0, 1, 1]), 0.05)


def test_baselines():
    g, counts = one_var_grammar(3)
    L = np.array([[0.0, -5.0, -1.0], [-1.0, 0.0, -2.0], [-0.1, -9.0, 0.0]])
    assert list(baselines.no_pooling(L, counts, g)) == [0, 1, 2]
    assert baselines.complete_pooling(L) == 0  # column sums: -1.1, -14, -3
    P = baselines.no_pooling_posterior(L, counts, g)
    assert np.allclose(P.sum(axis=1), 1.0)
    res, idx = baselines.oracle(L, counts, g, [0, 2])
    assert res.assign_probs.shape == (3, 2)
