"""The ecology catalogue hold-out source (hsr/data/sources/ecocat2_curves.csv ->
hsr/data/corpus_ecocat2.json); no network, no R."""
import json
import os
import re

import pytest
import sympy as sp

from hsr import corpus_sources as cs
from hsr import grammar_compare as gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "hsr", "data", "sources", "ecocat2_curves.csv")
CORPUS = os.path.join(ROOT, "hsr", "data", "corpus_ecocat2.json")
COLUMNS = ["id", "package", "function", "model", "variable", "params", "defs", "formula", "r_call", "ranges",
           "reference", "notes"]


@pytest.fixture(scope="module")
def rows():
    return cs.catalogue_rows(SRC)


@pytest.fixture(scope="module")
def corpus():
    with open(CORPUS) as fh:
        return json.load(fh)


def test_catalogue_columns_and_ids(rows):
    with open(SRC) as fh:
        assert fh.readline().strip().split("|") == COLUMNS
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert r["model"] and r["variable"] and r["reference"], r["id"]
        assert bool(r["package"]) == bool(r["r_call"].strip()), r["id"]


def test_every_formula_parses_with_its_parameters(rows):
    for r in rows:
        expr, x, params = cs.catalogue_expr(r)
        assert expr.has(x), r["id"]
        # every free symbol is x or a declared parameter
        assert expr.free_symbols <= {x} | set(params.values()), r["id"]
        for part in filter(None, (r["ranges"] or "").split(";")):
            name = part.split("=")[0].strip()
            assert name == "x" or name in params, (r["id"], name)


def test_packages_are_pinned_in_config(rows):
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(ROOT, "config", "ecocat2.yaml")) as fh:
        cfg = yaml.safe_load(fh)["ecocat2"]
    known = set(cfg["cran"]) | set(cfg["installed"])
    assert {r["package"] for r in rows if r["package"]} <= known
    assert set(cfg["sourced"]) == set(cfg["cran"])
    for pkg, members in cfg["sourced"].items():
        assert all(m.startswith(f"{pkg}/R/") for m in members)


def test_committed_corpus_is_verified(rows, corpus):
    """Every row with an R implementation agreed with it; rows without one are
    kept unverified; the only row left out is not expressible in the grammar."""
    v = corpus["verification"]
    assert {r["id"] for r in rows} == set(v)
    for r in rows:
        assert v[r["id"]]["formula"] == r["formula"]
        assert v[r["id"]]["verified"] == bool(r["r_call"].strip()), r["id"]
        if not v[r["id"]]["in_corpus"]:
            assert v[r["id"]]["key"] is None, r["id"]
    keys = {k for ks in corpus["per_model"].values() for k in ks}
    assert keys == set(corpus["freq"]) and all(cs._expressible(k) for k in keys)
    assert set(corpus["per_model"]) == {r["model"] for r in rows if v[r["id"]]["in_corpus"]}
    assert re.match(r"R=4\.5\.3;vegan=", corpus["versions"])


@pytest.mark.parametrize("rid,key", [("ml_gompertz", "c0*exp(c1*x1)"),
                                     ("vg_arrhenius", "c0*x1**c1"),
                                     ("tr_frequency", "c0*x1/(c1 + x1)"),
                                     ("sr_arrhenius", "c0*exp(c1/x1)")])
def test_known_keys(rows, rid, key):
    r = next(r for r in rows if r["id"] == rid)
    assert cs.catalogue_key(r) == key


def test_hold_out_not_a_training_source():
    """ecocat2 is scored as a hold-out only: never in the training sources."""
    assert "ecocat2" not in gc.SOURCES
    assert os.path.basename(gc.HOLDOUT_SOURCES["ecocat2"]) == "corpus_ecocat2.json"


def test_eval_csv_defs_substitute(rows):
    """defs are substituted (Heligman-Pollard: q = eta/(1 + eta))."""
    r = next(r for r in rows if r["id"] == "ml_hp")
    expr, x, p = cs.catalogue_expr(r)
    vals = {p["A"]: sp.Rational(1, 1000), p["B"]: sp.Rational(1, 100), p["C"]: sp.Rational(1, 10),
            p["D"]: sp.Rational(1, 1000), p["E"]: 10, p["F_"]: 20, p["G"]: sp.Rational(1, 10000),
            p["H"]: sp.Rational(11, 10)}
    eta = float((p["A"] ** ((x + p["B"]) ** p["C"]) + p["G"] * p["H"] ** x
                 + p["D"] * sp.exp(-p["E"] * sp.log(x / p["F_"]) ** 2)).subs(vals).subs(x, 30))
    assert abs(float(expr.subs(vals).subs(x, 30)) - eta / (1 + eta)) < 1e-12
