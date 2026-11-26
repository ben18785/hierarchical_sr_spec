"""Paper-1 tables and figures (workflow/scripts/paper1_data.py, paper1_report.py)."""
import importlib.util
import json
import os
import shutil

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "paper1")


def _module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "workflow", "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_split_options_and_by_node():
    d = _module("paper1_data")
    assert d.split_options(["--a", "1", "2", "--b", "x"]) == {"a": ["1", "2"], "b": ["x"]}
    lp = np.array([-1.0, -3.0, -2.0, -np.inf, -5.0])
    nodes = np.array([3, 3, 5, 5, np.nan])
    seen = np.array([True, True, False, False, True])
    b = d.by_node(lp, nodes, seen)
    assert b["seen"] == {"3": [2, -2.0, round(float(np.std([-1, -3], ddof=1)), 4)]}
    assert b["novel"] == {"5": [1, -2.0, 0.0]}   # -inf and unknown sizes are left out


@pytest.fixture(scope="module")
def report():
    pytest.importorskip("matplotlib")
    return _module("paper1_report")


def test_formatting(report):
    assert report.num(-6.306) == "$-6.31$"
    assert report.num(0.5, 1) == "0.5"
    assert report.num(None) == "--" and report.num(float("nan")) == "--"
    assert report.intc(19848) == "19{,}848"
    assert report.sci(2.6e-11) == "$2.6\\times 10^{-11}$"
    assert report.pname("tree_ad3 + hpy3_depth") == "\\texttt{tree\\_ad3} + \\texttt{hpy3\\_depth}"
    assert report.pname("flat") == "flat"


def test_tabular_checks_columns(report, tmp_path):
    p = tmp_path / "t.tex"
    report.tabular(str(p), "lr", ["a & b"], [["x", "1"], "MID", ("GROUP", "g")])
    txt = p.read_text()
    assert "\\toprule" in txt and "\\bottomrule" in txt and "\\multicolumn{2}{l}{\\emph{g}}" in txt
    with pytest.raises(ValueError):
        report.tabular(str(p), "lr", ["a & b"], [["x"]])


def test_uniform_symbols_is_improper(report):
    q, m = report.uniform_symbols_finite()
    # 2 leaves, 6 unary, 5 binary symbols: q = (2 + 6q + 5q^2) / 13, smallest root 0.4
    assert m == pytest.approx(16 / 13)
    assert q == pytest.approx(0.4, abs=1e-9)


def test_binned_pools_node_counts(report):
    rng = np.random.default_rng(1)
    x = {3: rng.normal(-2, 1, 6), 5: rng.normal(-4, 1, 9)}
    by_node = {str(k): [v.size, float(v.mean()), float(v.std(ddof=1))] for k, v in x.items()}
    b = report.binned(by_node)                     # nodes 3 and 5 fall in bins [1, 4) and [4, 7)
    allv = np.concatenate(list(x.values()))
    assert b.shape == (2, 3)
    assert b[0, 1] == pytest.approx(x[3].mean()) and b[1, 1] == pytest.approx(x[5].mean())
    by_node = {"4": by_node["3"], "5": by_node["5"]}  # both in [4, 7): pooled mean and sd
    b = report.binned(by_node)
    assert b[0, 1] == pytest.approx(allv.mean())
    assert b[0, 2] == pytest.approx(1.96 * allv.std(ddof=1) / np.sqrt(allv.size))


@pytest.mark.skipif(not os.path.exists(os.path.join(OUT, "data", "corpus.json")), reason="no paper-1 data")
def test_committed_outputs_regenerate(report, tmp_path):
    """The committed tables and README are what the code makes from the committed data."""
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(ROOT, "config", "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    shutil.copytree(os.path.join(OUT, "data"), tmp_path / "data")
    report.main(str(tmp_path), os.path.join(ROOT, "results", "grammar"), json.dumps(cfg["paper1"]),
                ",".join(cfg["hpyv"]["run"]))
    made = sorted(f for f in os.listdir(tmp_path) if f.endswith((".tex", ".md", ".pdf")))
    committed = sorted(f for f in os.listdir(OUT) if f.endswith((".tex", ".md", ".pdf")))
    assert made == committed
    for f in made:
        if not f.endswith(".pdf"):
            assert (tmp_path / f).read_text() == open(os.path.join(OUT, f)).read(), f
