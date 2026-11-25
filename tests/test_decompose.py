import numpy as np

from hsr.decompose import evidence_decomposition


def test_changed_choice_split_into_base_and_sharing():
    # dataset 0: data prefer form 1 by 3 nats, method chose form 0; the base
    # measure favours form 0 by 1 nat, so sharing supplied the other 2
    L = np.array([[0.0, 3.0, -5.0],
                  [4.0, 1.0, -5.0]])
    lp = np.array([-2.0, -3.0, -1.0])
    probs = np.array([[0.9, 0.1, 0.0],
                      [0.8, 0.2, 0.0]])
    d = evidence_decomposition(probs, L, lp, keys=["a", "b", "c"])
    assert d["dec_frac_changed"] == 0.5
    assert np.isclose(d["dec_data_cost_mean"], 3.0)
    assert np.isclose(d["dec_base_gain_mean"], 1.0)
    assert np.isclose(d["dec_sharing_mean"], 2.0)
    assert d["dec_frac_base_decides"] == 0.0
    # dataset 1 kept the data's choice: margin 3 over b; base margin -2 + 3 = 1
    assert np.isclose(d["dec_data_margin_median"], 3.0)
    assert np.isclose(d["dec_base_margin_median"], 1.0)
    assert d["dec_datasets"][0]["chosen"] == "a" and d["dec_datasets"][0]["rival"] == "b"


def test_base_decides_and_nonfinite_evidence():
    L = np.array([[0.0, 0.5, -np.inf, np.nan]])
    lp = np.array([0.0, -2.0, 0.0, 0.0])
    probs = np.array([[1.0, 0.0, 0.0, 0.0]])
    d = evidence_decomposition(probs, L, lp)
    assert d["dec_frac_changed"] == 1.0
    assert d["dec_frac_base_decides"] == 1.0
    assert np.isclose(d["dec_sharing_mean"], 0.5 - 2.0)
