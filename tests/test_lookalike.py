import numpy as np

from hsr import lookalike as la


def test_dataset_scores_rank_mass_and_map():
    L = np.array([0.0, -1.0, -3.0, -np.inf, -10.0])
    lp = np.log(np.array([0.1, 0.6, 0.2, 0.05, 0.05]))
    s = la.dataset_scores(L, lp, t=1, deltas=[2, 5])
    assert s[2]["in_set"] and s[2]["set_size"] == 2          # candidates 0 and 1
    assert s[2]["rank"] == 1 and s[2]["top"]
    assert np.isclose(s[2]["mass"], 0.6 / 0.7)
    assert s[5]["set_size"] == 3 and np.isclose(s[5]["mass"], 0.6 / 0.9)
    post = L + lp
    p = np.exp(post[1] - np.logaddexp.reduce(post[np.isfinite(post)]))
    assert np.isclose(s["p_post"], p) and s["map"]


def test_flat_prior_mass_is_one_over_set_size():
    L = np.array([0.0, -0.5, -1.0, -8.0])
    s = la.dataset_scores(L, np.zeros(4), t=2, deltas=[2])
    assert np.isclose(s[2]["mass"], 1 / 3) and np.isclose(s[2]["gain"], 0.0)
    assert s[2]["rank"] == 2.0                                # ties count half: 1 + 0 + 0.5 * 2
    assert not s["map"]


def test_truth_outside_set():
    s = la.dataset_scores(np.array([0.0, -6.0]), np.zeros(2), t=1, deltas=[2, 5])
    assert not s[2]["in_set"] and not s[5]["in_set"]
    summ = la.summarise([s], [2, 5])
    assert summ["in_set_2"] == 0.0 and summ["mass_2"] is None


def _folds():
    support = ["c0*x1", "c0*x1/(c1 + x1)", "c0*exp(c1*x1)", "c0*exp(c1*x1) + c2", "x1", "c0*x1**2 + c1"]
    mech = {"support": support,
            "folds": [{"train": ["c0*x1", "c0*x1/(c1 + x1)", "x1"],
                       "test": ["c0*x1/(c1 + x1)", "c0*exp(c1*x1) + c2", "c0*exp(c1*x1)"],
                       "test_sources": ["rnls", "rnls", "mech_compartment"]}]}
    temporal = {"support": support[:4] + ["x1"],
                "folds": [{"train": ["c0*x1"], "test": ["c0*exp(c1*x1)", "c0*x1"]}]}
    trees = {k: ["n"] * (3 + len(k) // 4) for k in support}
    return mech, temporal, trees


def test_make_design_strata_and_candidates():
    mech, temporal, trees = _folds()
    cfg = {"seed": 0, "max_consts": 3, "max_nodes": 30, "reps": 2, "settings": [[8, 0.05]],
           "strata": {"holdout_seen": 1, "holdout_novel": 1, "mech_compartment": 1, "temporal_novel": 1}}
    d = la.make_design(mech, temporal, trees, ["rnls"], ["mech_compartment"], cfg)
    strata = {t["stratum"]: t for t in d["true_forms"]}
    assert strata["holdout_seen"]["key"] == "c0*x1/(c1 + x1)" and strata["holdout_seen"]["seen"]
    assert strata["holdout_novel"]["key"] == "c0*exp(c1*x1) + c2" and not strata["holdout_novel"]["seen"]
    assert strata["temporal_novel"]["key"] == "c0*exp(c1*x1)"
    assert len(d["datasets"]) == 4 * 2 and all(len(x["x"]) == 8 for x in d["datasets"])
    assert set(d["candidates"]) == set(mech["support"])
    assert d["in_temporal"][d["candidates"].index("c0*x1**2 + c1")] is False
    X, Y, M = la.padded(d)
    assert X.shape == (8, 8, 1) and M.sum() == 64


def test_evidence_shard_prefers_truth_on_clean_data():
    mech, temporal, trees = _folds()
    cfg = {"seed": 0, "max_consts": 3, "max_nodes": 30, "reps": 1, "settings": [[30, 0.01]],
           "strata": {"holdout_novel": 1}}
    d = la.make_design(mech, temporal, trees, ["rnls"], [], cfg)
    keys, L, _ = la.evidence_shard(d, 0, 1, n_starts=4, progress=False)
    assert keys == d["candidates"]
    truth = keys.index(d["true_forms"][0]["key"])
    assert np.isfinite(L[0, truth]) and L[0, truth] >= L[0, keys.index("c0*x1")]


def test_temper_is_proper_and_interpolates():
    lp = np.log(np.array([0.5, 0.3, 0.2]))
    for tau in (0.0, 0.25, 1.0, 1.5):
        q = np.exp(la.temper(lp, tau))
        assert np.isclose(q.sum(), 1.0) and np.all(q > 0)
    assert np.allclose(np.exp(la.temper(lp, 0.0)), 1 / 3)
    assert np.allclose(np.exp(la.temper(lp, 1.0)), [0.5, 0.3, 0.2])
    assert np.allclose(np.exp(la.temper(lp + 7.0, 1.0)), [0.5, 0.3, 0.2])      # renormalised over the set
    q2 = np.exp(la.temper(lp, 2.0))
    assert np.allclose(q2, np.array([0.25, 0.09, 0.04]) / 0.38)
    # zero-probability forms stay at zero for tau > 0; tau = 0 is flat on every candidate
    lz = np.array([0.0, -np.inf, -1.0])
    assert np.exp(la.temper(lz, 0.5))[1] == 0 and np.isclose(np.exp(la.temper(lz, 0.5)).sum(), 1)
    assert np.allclose(np.exp(la.temper(lz, 0.0)), 1 / 3)


def test_log_p_true_taus_matches_dataset_scores():
    L = np.array([0.0, -1.0, -3.0, -np.inf, -10.0])
    lp = np.log(np.array([0.1, 0.6, 0.2, 0.05, 0.05]))
    taus = [0.0, 0.5, 1.0, 1.5]
    got = la.log_p_true_taus(L, la.temper_all(lp, taus), 1)
    for tau, g in zip(taus, got):
        s = la.dataset_scores(L, la.temper(lp, tau), 1, [2])
        assert np.isclose(g, s["log_p_post"]) and np.isclose(np.exp(g), s["p_post"])
    assert s["post_rank"] == 1.0
    assert np.all(np.isneginf(la.log_p_true_taus(L, la.temper_all(lp, taus), 3)))   # failed true-form fit


def test_select_tau_loto():
    taus = [0.0, 1.0, 2.0]
    # form 0 prefers tau 2, forms 1 and 2 prefer tau 1
    logp = np.array([[-3.0, -2.0, -1.0], [-3.0, -1.0, -2.0], [-3.0, -1.0, -2.5], [-3.0, -1.5, -2.0]])
    folds = np.array([0, 1, 2, 2])
    r = la.select_tau_loto(logp, folds, taus)
    assert r["tau_full"] == 1.0
    assert r["tau_folds"] == {0: 1.0, 1: 1.0, 2: 1.0}
    assert np.allclose(r["cv"], [-2.0, -1.0, -1.0, -1.5])
    r2 = la.select_tau_loto(logp[:2], folds[:2], taus)          # leaving form 1 out leaves form 0: tau 2
    assert r2["tau_folds"] == {0: 1.0, 1: 2.0} and np.allclose(r2["cv"], [-2.0, -2.0])
