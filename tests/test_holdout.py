"""Hold-out sources: never in training, scored after training on SOURCES."""
import importlib.util
import json
import os

import numpy as np
import pytest
import yaml

from hsr import grammar_compare as gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _corpus(path, per_model):
    path.write_text(json.dumps({"per_model": per_model}))
    return str(path)


def test_holdouts_are_not_training_sources():
    with open(os.path.join(ROOT, "config", "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    assert set(cfg["holdout_sources"]) == set(gc.HOLDOUT_SOURCES)
    assert not set(gc.HOLDOUT_SOURCES) & set(gc.SOURCES)
    assert len(gc.SOURCES) == 12
    assert not {os.path.abspath(p) for p in gc.HOLDOUT_SOURCES.values()} & {os.path.abspath(p) for p in
                                                                            gc.SOURCES.values()}


def test_make_holdout(tmp_path):
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"keys": ["c0*x1"]}))
    a = _corpus(tmp_path / "a.json", {"m1": ["c0*x1/(c1 + x1)"], "m2": ["c0*x1/(c1 + x1)", "c0*exp(c1*x1)"]})
    b = _corpus(tmp_path / "b.json", {"n1": ["c0*x1**c1"]})
    h1 = _corpus(tmp_path / "h1.json", {"p1": ["c0*x1/(c1 + x1)", "c0 + c1*x1"], "p2": ["c0 + c1*x1"]})
    h2 = _corpus(tmp_path / "h2.json", {"q1": ["c0*log(x1)"]})
    F = gc.make_holdout({}, {"h1": h1, "h2": h2}, "pooled", sources={"a": a, "b": b}, candidates=str(cand))
    fold = F["folds"][0]
    assert sorted(fold["train"]) == sorted(["c0*x1/(c1 + x1)", "c0*x1/(c1 + x1)", "c0*exp(c1*x1)", "c0*x1**c1"])
    assert fold["test"] == ["c0 + c1*x1", "c0*x1/(c1 + x1)", "c0 + c1*x1", "c0*log(x1)"]
    assert fold["test_sources"] == ["h1", "h1", "h1", "h2"]
    assert F["holdout_models"] == {"h1": 2, "h2": 1}
    assert set(fold["test"]) <= set(F["support"]) and "c0*x1" in F["support"]
    # equal weighting replicates the smaller source; the test set does not change
    E = gc.make_holdout({}, {"h1": h1, "h2": h2}, "equal", sources={"a": a, "b": b}, candidates=str(cand))
    assert E["folds"][0]["train"].count("c0*x1**c1") == 3 and E["folds"][0]["test"] == fold["test"]
    # the test-source label is parsed as in hpyv_score.train_sources
    assert E["test_corpus"].endswith("; equal)") and E["test_corpus"].split(" ")[0] == "holdouts"
    with pytest.raises(ValueError):
        gc.make_holdout({}, {"a": a}, "pooled", sources={"a": a, "b": b}, candidates=str(cand))


def test_holdout_report_seen_novel():
    spec = importlib.util.spec_from_file_location("holdout_report",
                                                  os.path.join(ROOT, "workflow", "scripts", "holdout_report.py"))
    hr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hr)
    F = {"support": ["a", "b", "c"],
         "folds": [{"train": ["a"], "test": ["a", "b", "b", "c"], "test_sources": ["h1", "h1", "h1", "h2"]}]}
    lp = {"p": np.log(np.array([0.5, 0.25, 0.25]))}
    out, counts = hr.summarise(F, lp, {"p": "tree"})
    assert counts["h1"] == {"pairs": 3, "distinct": 2, "novel_pairs": 2, "novel_distinct": 1,
                            "redundant_novel_pairs": 0, "redundant_novel_distinct": 0}
    assert counts["all"]["pairs"] == 4
    s = out["p"]["h1"]
    assert np.isclose(s["raw"], np.mean(np.log([0.5, 0.25, 0.25])))
    assert np.isclose(s["raw_distinct"], np.mean(np.log([0.5, 0.25])))
    assert np.isclose(s["raw_seen"], np.log(0.5)) and np.isclose(s["raw_novel"], np.log(0.25))
    assert np.isclose(out["p"]["h2"]["raw_seen"] or 0, 0) and out["p"]["h2"]["raw_seen"] is None


def test_redundant_constants_detected():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).parents[1] / "workflow" / "scripts" / "holdout_report.py"
    spec = importlib.util.spec_from_file_location("holdout_report", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.redundant("c0*x1 + c1*x1")
    assert m.redundant("c0*c1*x1/(c2 + x1)")
    assert not m.redundant("c0*x1/(c1 + x1)")
    assert not m.redundant("c0*exp(c1*x1)")
