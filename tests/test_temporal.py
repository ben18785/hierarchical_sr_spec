"""Model dates (hsr.model_dates) and temporal hold-out folds (hsr.temporal); no network."""
import gzip
import json

import pytest

from hsr import grammar_compare as gc
from hsr import model_dates as md
from hsr import temporal


# ------------------------------------------------------------------ dates
def test_years_in_ignores_ids_and_implausible_numbers():
    assert md.years_in("Hemond P, Epstein D (2008)") == [2008]
    assert md.years_in("BIOMD0000002010") == []
    assert md.years_in("model 12345 of 1750 and 2999") == []
    assert md.years_in("Land et al 2012") == [2012]


def test_biomodels_year_prefers_the_publication():
    rec = {"publication": {"year": 2018}, "history": {"revisions": [{"submitted": 1488540300}]},
           "firstPublished": 1725285415}
    assert md.biomodels_year(rec) == (2018, "publication")
    rec["publication"] = {"year": None}
    assert md.biomodels_year(rec) == (2017, "first submission")      # 1488540300 = 2017-03-03
    del rec["history"]
    assert md.biomodels_year(rec) == (2024, "first published")
    assert md.biomodels_year({}) == (None, "undated")
    # milliseconds are accepted as well
    assert md.biomodels_year({"history": {"revisions": [{"submitted": 1488540300000}]}})[0] == 2017


def test_modeldb_year_paper_then_name_then_created():
    rec = {"name": "CA3 (Hemond et al. 2008)", "created": "2007-11-28T03:20:04",
           "model_paper": {"value": [{"object_name": "Hemond P, Jaffe DB (2008)"},
                                     {"object_name": "Hemond P (2006)"}]}}
    assert md.modeldb_year(rec) == (2006, "paper")
    del rec["model_paper"]
    assert md.modeldb_year(rec) == (2008, "name")
    rec["name"] = "CA3 pyramidal neuron"
    assert md.modeldb_year(rec) == (2007, "created")


def test_jws_year():
    assert md.jws_year({"year": 2013}) == (2013, "manuscript")
    assert md.jws_year({"year": None}) == (None, "undated")


def test_cellml_allodb_and_catalogue_years(tmp_path):
    assert md.cellml_year("w__dcowan__Land_et_al_2012", ["a.cellml"]) == 2012
    assert md.cellml_year("workspace__35f", ["gupta_2009.cellml"]) == 2009
    assert md.cellml_year("workspace__35f", ["a_2009.cellml", "b_2011.cellml"]) is None
    assert md.cellml_year("workspace__35f", ["model.cellml"]) is None
    assert md.allodb_year("allodb:barney_1977_bdac") == 1977
    assert md.allodb_year("allodb:nodate_xyz") is None
    p = tmp_path / "cat.csv"
    p.write_text("id|model|reference\nr1|M|Smith A (2004) J 1:1\nr2|M|Jones B (1999) J 2:2\nr3|N|no year\n")
    assert md.catalogue_years(str(p), "model") == {"M": 1999}


def test_sabiork_years_from_cached_pages(tmp_path):
    data = {"data": [{"publication": {"publication_id": 7, "year": 2003}},
                     {"general": {"publication_id": 8}, "publication": {"year": None}},
                     {"publication": {"publication_id": 7, "year": 2001}}]}
    with gzip.open(tmp_path / "p.json.gz", "wt") as fh:
        json.dump(data, fh)
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [{"file": "p.json.gz"}]}))
    assert md.sabiork_years(str(tmp_path)) == {"SABIORK_PUB7": 2001}


# ------------------------------------------------------------------ folds
@pytest.fixture
def toy(tmp_path):
    corpora = {"a": {"m1": ["c0*x1"], "m2": ["c0*x1", "exp(x1)"], "m3": ["log(x1)"], "m4": ["x1**2"]},
               "b": {"n1": ["c0*x1"], "n2": ["c0/x1"]},
               "u": {"lib1": ["sqrt(x1)"]}}
    sources = {}
    for name, pm in corpora.items():
        p = tmp_path / f"corpus_{name}.json"
        p.write_text(json.dumps({"per_model": pm}))
        sources[name] = str(p)
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"keys": ["c0 + x1"]}))
    dates = {"years": {"a": {"m1": 2000, "m2": 2012, "m3": 2016}, "b": {"n1": 2005, "n2": 2020}},
             "undated_sources": ["u"]}
    return sources, str(cand), dates


def test_make_temporal_split(toy):
    sources, cand, dates = toy
    F = temporal.make_temporal({}, dates, 2010, "pooled", sources=sources, candidates=cand)
    fold = F["folds"][0]
    # trained on the models before 2010 and the undated source; m4 (undated, dated source) dropped
    assert sorted(fold["train"]) == ["c0*x1", "c0*x1", "sqrt(x1)"]
    assert fold["train_labels"] == ["a", "b", "u"]
    assert fold["test"] == ["c0*x1", "exp(x1)", "log(x1)", "c0/x1"]
    assert fold["test_models"] == ["a:m2", "a:m3", "b:n2"]
    assert fold["novel"] == [False, True, True, True]
    assert fold["test_years"] == [2012, 2012, 2016, 2020]
    assert F["sizes"]["a"]["dropped_undated"] == 1
    assert "x1**2" in F["support"] and "c0 + x1" in F["support"]
    # later cutoff: more training, fewer test pairs
    G = temporal.make_temporal({}, dates, 2015, "pooled", sources=sources, candidates=cand)["folds"][0]
    assert G["test"] == ["log(x1)", "c0/x1"] and G["novel"] == [True, True]


def test_make_temporal_equal_weighting_matches_weight_sources(toy):
    sources, cand, dates = toy
    F = temporal.make_temporal({}, dates, 2015, "equal", sources=sources, candidates=cand)
    fold = F["folds"][0]
    # a has 3 training pairs, b and u one each: replicated three times
    assert fold["train_labels"].count("a") == 3 and fold["train_labels"].count("b") == 3
    parts = {"a": ["c0*x1", "c0*x1", "exp(x1)"], "b": ["c0*x1"], "u": ["sqrt(x1)"]}
    assert fold["train"] == gc.weight_sources(parts, "equal")


def test_make_temporal_needs_dates_for_every_source(toy):
    sources, cand, dates = toy
    dates = {"years": {"a": dates["years"]["a"]}, "undated_sources": ["u"]}
    with pytest.raises(ValueError):
        temporal.make_temporal({}, dates, 2010, sources=sources, candidates=cand)


def test_hpyv_score_takes_labels_from_the_fold(toy):
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("hpyv_score", os.path.join(root, "workflow", "scripts",
                                                                            "hpyv_score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sources, cand, dates = toy
    F = temporal.make_temporal({}, dates, 2010, "equal", sources=sources, candidates=cand)
    labels, test_src = mod.train_sources(F, F["folds"][0], {})
    assert labels == F["folds"][0]["train_labels"] and test_src == "temporal"
