"""Convergence diagnostics."""
import numpy as np

from hsr.diagnostics import chain_tv, ess_bulk, split_rhat


def test_rhat_and_ess_on_iid_and_stuck_chains():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=(4, 1000))
    assert split_rhat(iid) < 1.01
    assert ess_bulk(iid) > 2500
    shifted = iid + np.array([[0], [0], [0], [3]])
    assert split_rhat(shifted) > 1.3
    ar = np.zeros((4, 1000))
    for t in range(1, 1000):
        ar[:, t] = 0.95 * ar[:, t - 1] + rng.normal(size=4)
    assert ess_bulk(ar) < 400
    const_same = np.ones((4, 100))
    assert split_rhat(const_same) == 1.0
    const_diff = np.ones((4, 100)) * np.array([[1], [1], [2], [1]])
    assert split_rhat(const_diff) == np.inf


def test_chain_tv():
    P = np.zeros((2, 3, 2))
    P[0, :, 0] = 1
    P[1, :2, 0] = 1
    P[1, 2, 1] = 1
    mean, mx = chain_tv(P)
    assert np.isclose(mx, 1.0) and np.isclose(mean, 1 / 3)


def test_chain_report_flags_stuck_chain():
    from types import SimpleNamespace
    from hsr.diagnostics import chain_report
    rng = np.random.default_rng(0)
    moving = rng.integers(0, 3, size=(50, 5))
    stuck = np.zeros((50, 5), int)
    res = SimpleNamespace(z_samples=np.stack([moving, stuck]), log_lik=np.zeros((2, 50)),
                          log_post=np.vstack([np.zeros(50), -10 * np.ones(50)]),
                          n_forms=np.ones((2, 50)))
    rep = chain_report(res, keys=["a", "b", "c"])
    assert rep[0]["moves"] > 0.5 and rep[1]["moves"] == 0 and rep[1]["n_states"] == 1
    assert rep[1]["top"] == [["a", 1.0]] and rep[1]["log_post"] == -10
