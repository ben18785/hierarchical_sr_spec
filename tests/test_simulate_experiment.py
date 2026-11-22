import numpy as np
import pytest

from hsr.candidates import from_strings
from hsr.canonical import canonicalise
from hsr.experiment import run_replicate
from hsr.gibbs import GibbsConfig
from hsr.grammar import Grammar
from hsr.likelihood import CompiledSkeleton
from hsr.simulate import BENCHMARK_FORMS, X_HIGH, X_LOW, all_A_population, benchmark_population


@pytest.mark.parametrize("name", list(BENCHMARK_FORMS))
def test_benchmark_curves_are_visibly_nonlinear(name):
    """Constants are drawn so that a straight line explains < 97% of the
    curve's variance on the observed range."""
    string, draw = BENCHMARK_FORMS[name]
    cs = CompiledSkeleton(canonicalise(string))
    rng = np.random.default_rng(0)
    x = np.linspace(X_LOW, X_HIGH, 200)
    for _ in range(50):
        phi = draw(rng)
        f = cs.f(x[None, :, None], phi[None])[0]
        r2 = np.corrcoef(x, f)[0, 1] ** 2
        assert r2 < 0.97, (name, phi, r2)


def test_constant_order_matches_canonical_skeleton():
    # the drawn constants are in canonical order: scale first, then the other
    x = np.array([1.0])
    for name, (string, draw) in BENCHMARK_FORMS.items():
        cs = CompiledSkeleton(canonicalise(string))
        phi = np.array([2.0, 0.5])
        f = cs.f(x[None, :, None], phi[None])[0, 0]
        ref = {"A": 2 * np.exp(0.5), "B": 2 / 1.5, "C": 2 / 1.5}[name]
        assert np.isclose(f, ref)
    assert BENCHMARK_FORMS["A"][1](np.random.default_rng(1))[1] < 0  # decaying


def test_population_generation():
    rng = np.random.default_rng(2)
    pop = benchmark_population(rng, n=10, noise_frac=0.05, J=200)
    assert len(pop.datasets) == 200
    freq = np.bincount(pop.labels, minlength=3) / 200
    assert np.allclose(freq, [0.6, 0.3, 0.1], atol=0.08)
    ds = pop.datasets[0]
    assert ds.x.shape == (10,) and ds.x_in.shape == (20,) and ds.x_out.shape == (20,)
    assert np.all(ds.x_out >= X_HIGH)
    popA = all_A_population(rng, 5, 0.15, J=10)
    assert set(popA.true_keys) == {canonicalise("C*exp(C*x1)").key}


@pytest.fixture(scope="module")
def tiny_candidates():
    strings = ["x1", "C*x1", "C*x1+C", "C*exp(C*x1)", "C*x1/(C+x1)", "C*log(x1)+C", "C*sqrt(x1)",
               "exp(-x1)", "C*x1*x1+C", "C/(C+x1)"]
    return from_strings(strings, Grammar())


@pytest.mark.parametrize("setting", ["mixture", "all_A", "grammar"])
def test_run_replicate_smoke(tiny_candidates, setting):
    cfg = GibbsConfig(n_sweeps=60, burn_in=20, n_chains=2)
    rows = run_replicate(setting, 10, 0.05, 0, tiny_candidates, cfg, J=8, n_starts=3, expand=False)
    methods = {(r["method"], r["kind"]) for r in rows}
    for m in ["hier", "no_pool", "no_pool_bma", "complete_pool", "oracle"]:
        assert (m, "laplace") in methods and (m, "bic") in methods
    for r in rows:
        assert 0 <= r["recovery"] <= 1
        assert np.isfinite(r["lpd_in"]) and np.isfinite(r["lpd_out"])
    if setting == "mixture":
        # form C is not in the tiny set, so it is appended for the oracle only
        assert rows[0]["true_forms_in_C"][canonicalise("C*x1/(C+x1*x1)").key] is False


def test_run_replicate_with_edits_and_sensitivity(tiny_candidates):
    cfg = GibbsConfig(n_sweeps=40, burn_in=10, n_chains=2)
    rows = run_replicate("mixture", 10, 0.05, 1, tiny_candidates, cfg, J=6, n_starts=2, expand=True,
                         sensitivity=True, kinds=("laplace",))
    assert rows[0]["n_added_by_edits"] > 0
    assert any(r["method"].startswith("hier[alpha") for r in rows)


# ---------------------------------------------------------------- families
from hsr.families import FAMILIES, SETTINGS, family_metrics, family_population


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_variant_reduces_to_parent_at_null(family):
    fam = FAMILIES[family]
    x = np.linspace(0.1, 5, 50)
    p = fam["parent"].draw(np.random.default_rng(0))
    null = np.append(p, 1.0)      # extension coefficient at its null value
    assert np.allclose(fam["variant"].f(x, null), fam["parent"].f(x, p))


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_keys_are_canonical_and_distinct(family):
    fam = FAMILIES[family]
    kp, kv = fam["parent"].key, fam["variant"].key
    assert kp != kv
    assert canonicalise(kp).key == kp and canonicalise(kv).key == kv
    assert canonicalise(kv).n_consts == canonicalise(kp).n_consts + 1


def test_family_populations_and_metrics():
    rng = np.random.default_rng(1)
    pop = family_population(rng, "saturation_mixture", 10, 0.05, J=200)
    frac = np.mean([k == pop.variant_key for k in pop.true_keys])
    assert abs(frac - 0.3) < 0.08
    assert len(pop.form_keys) == 2
    allp = family_population(rng, "decay_parent", 10, 0.05, J=5)
    assert set(allp.true_keys) == {allp.parent_key}
    m = family_metrics(allp, [allp.variant_key, allp.parent_key, "other", allp.parent_key, allp.parent_key])
    assert m["family_recovery"] == 0.8 and m["variant_when_parent"] == 0.2 and m["variant_detected"] is None
    assert set(SETTINGS) >= {"saturation_extended", "decay_mixture"}


def test_run_replicate_family_pooled_and_growth(tiny_candidates):
    cfg = GibbsConfig(n_sweeps=40, burn_in=10, n_chains=2)
    rows = run_replicate("saturation_mixture", 10, 0.05, 0, tiny_candidates, cfg, J=8, n_starts=2,
                         grow_rounds=1, pooled=True, kinds=("laplace",))
    methods = {r["method"] for r in rows}
    assert "hier_pooled" in methods and "hier" in methods
    r = [r for r in rows if r["method"] == "hier_pooled"][0]
    assert "family_recovery" in r and "variant_when_parent" in r
    assert r["n_added_by_edits"] > 0


def test_run_replicate_family_themes(tiny_candidates):
    cfg = GibbsConfig(n_sweeps=40, burn_in=10, n_chains=2)
    rows = run_replicate("decay_mixture", 10, 0.05, 0, tiny_candidates, cfg, J=6, n_starts=2,
                         grow_rounds=1, themes=True, theme_top_k=2, kinds=("laplace",))
    methods = {r["method"] for r in rows}
    assert {"hier_themes", "hier_pooled", "hier"} <= methods
    r = [r for r in rows if r["method"] == "hier_themes"][0]
    assert len(r["theme_top_keys"]) == 6 and len(r["top_keys"]) == 6
    assert "theme_family" in r and "extension_detected" in r or "extension_when_parent" in r
    assert r["n_kernel_edges"] > 0 and r["n_added_by_themes"] >= 0
