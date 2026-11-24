import numpy as np

from hsr import fixed_prior as fp


def test_tree_priors_rank_corpus_forms_and_floor_uncodable(monkeypatch):
    """The adaptor and HPY tree priors (hsr.fixed_prior) give finite log
    priors, rank a frequent training form above an exotic one, and give
    uncodable keys the floor (smallest finite value minus 10)."""
    train = ["c0*x1/(c1 + x1)"] * 30 + ["c0*exp(c1*x1)"] * 10 + ["c0*x1"] * 10
    monkeypatch.setattr(fp, "_combined_training_keys", lambda exclude, weighting="equal": list(train))
    monkeypatch.setattr(fp, "_TREE_MODELS", {})
    keys = ["c0*x1/(c1 + x1)", "c0*sin(c1*tanh(x1))", "not a key ((("]
    for v in fp.TREE_VARIANTS:
        lp = fp.tree_log_prior(keys, v)
        assert np.all(np.isfinite(lp))
        assert lp[0] > lp[1]
        assert np.isclose(lp[2], min(lp[:2]) - 10.0)
    # the adaptor caches whole forms, so a repeated form gains over the plain HPY
    assert fp.tree_log_prior(keys[:1], "adaptor")[0] > fp.tree_log_prior(keys[:1], "hpy")[0]
