"""Hold-out source rnls (hsr.rnls, workflow/scripts/rnls_extract.R); no network."""
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile

import pytest

from hsr import rnls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tsv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _manifest(d, files):
    entries = []
    for name, data in files.items():
        (d / name).write_bytes(data)
        entries.append({"file": name, "sha256": hashlib.sha256(data).hexdigest()})
    (d / "manifest.json").write_text(json.dumps({"date": "2026-01-01", "files": entries}))


def test_select_packages_keyword_filter(tmp_path):
    idx = tmp_path / "index"
    idx.mkdir()
    _manifest(idx, {"Views.rds": b"v", "packages.rds": b"p"})
    fields = ["view", "package", "version", "license", "title", "description"]
    _tsv(tmp_path / "views.tsv", [
        {"view": "Environmetrics", "package": "ecofit", "version": "1.0", "license": "GPL-3",
         "title": "Species Abundance Models", "description": "Fits curves."},
        {"view": "Agriculture", "package": "ecofit", "version": "1.0", "license": "GPL-3",
         "title": "Species Abundance Models", "description": "Fits curves."},
        {"view": "Environmetrics", "package": "optimx2", "version": "2.0", "license": "MIT",
         "title": "General Optimisers", "description": "Numerical optimisation."},
        {"view": "Epidemiology", "package": "gone", "version": "", "license": "", "title": "", "description": ""},
        {"view": "Epidemiology", "package": "epiX", "version": "0.1", "license": "GPL-2",
         "title": "Outbreak Tools", "description": "x"},
    ], fields)
    out = rnls.select_packages(str(idx), str(tmp_path / "views.tsv"), str(tmp_path / "pk.tsv"), exclude="epiX")
    assert [r["package"] for r in out] == ["ecofit"]
    assert out[0]["views"] == "Agriculture,Environmetrics" and out[0]["keyword"] == "species"


def test_formula_law():
    law, regs = rnls.formula_law("Asym/(1 + exp((xmid - Time)/scal))", ["Time"])
    assert regs == ("r_Time",) and "r_Asym" in law
    with pytest.raises(ValueError):
        rnls.formula_law("ifelse(x < c, a, b*x)", ["x"])
    with pytest.raises(ValueError):
        rnls.formula_law("a*d$x", ["x"])


def test_build(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _manifest(raw, {"p1_1.0.tar.gz": b"1", "p2_1.0.tar.gz": b"2"})
    _tsv(tmp_path / "pk.tsv", [{"package": "p1", "version": "1.0", "license": "GPL-3", "views": "A", "keyword": "x"},
                               {"package": "p2", "version": "1.0", "license": "MIT", "views": "A", "keyword": "x"}],
         ["package", "version", "license", "views", "keyword"])
    f = ["package", "file", "kind", "lhs", "rhs", "params", "regressors", "status"]
    _tsv(tmp_path / "f.tsv", [
        {"package": "p1", "file": "R/a.R", "kind": "nls", "lhs": "y", "rhs": "Vm * S/(K + S)", "params": "Vm,K",
         "regressors": "S", "status": "ok"},
        {"package": "p1", "file": "man/a.Rd", "kind": "nls", "lhs": "y", "rhs": "Vm * S/(K + S)", "params": "Vm,K",
         "regressors": "S", "status": "ok"},
        {"package": "p2", "file": "R/b.R", "kind": "nlme", "lhs": "y", "rhs": "a * conc/(b + conc)",
         "params": "a,b", "regressors": "conc", "status": "ok"},
        {"package": "p2", "file": "R/b.R", "kind": "nls", "lhs": "y", "rhs": "a * x^b * z", "params": "a,b",
         "regressors": "x,z", "status": "ok"},
        {"package": "p2", "file": "R/c.R", "kind": "nls", "lhs": "y", "rhs": "", "params": "", "regressors": "",
         "status": "formula not resolvable"},
        {"package": "p2", "file": "R/c.R", "kind": "nls", "lhs": "y", "rhs": "ifelse(x < c, a, b)",
         "params": "a,b,c", "regressors": "x", "status": "ok"},
    ], f)
    res = rnls.build(str(raw), str(tmp_path / "pk.tsv"), str(tmp_path / "f.tsv"), str(tmp_path / "c.json"), 1)
    assert res["freq"]["c0*x1/(c1 + x1)"] == 2                     # two packages
    assert set(res["per_model"]) == {"cran:p1", "cran:p2"}
    assert "c0*x1" in res["per_model"]["cran:p2"]                  # z projected with x constant
    st = res["stats"]
    assert st["found"] == 6 and st["duplicates_within_package"] == 1 and st["multi_regressor"] == 1
    assert st["drop_reasons"]["formula not resolvable"] == 1
    assert any(k.startswith("transcription") for k in st["drop_reasons"])
    assert res["licenses"]["cran:p2"] == "MIT"


R_CODE = """
fm <- y ~ Asym/(1 + exp((xmid - age)/scal))
fit1 <- nls(fm, data = d, start = list(Asym = 1, xmid = 2, scal = 3))
fit2 <- nlme::nlme(height ~ SSasymp(age, Asym, R0, lrc), data = Loblolly,
                   fixed = Asym + R0 + lrc ~ 1, random = Asym ~ 1)
mm <- function(conc, Vm, K) { .value <- Vm * conc / (K + conc); .value }
SSmm <- selfStart(mm, function(mCall, data, LHS, ...) NULL, c("Vm", "K"))
fit3 <- nls(rate ~ SSmm(conc, Vm, K), data = Puromycin)
fit4 <- nls(y ~ a * x^b, data = d)
"""


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not available")
def test_extract_R(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = R_CODE.encode()
        ti = tarfile.TarInfo("toy/R/fits.R")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    (tmp_path / "toy_1.0.tar.gz").write_bytes(buf.getvalue())
    _tsv(tmp_path / "pk.tsv", [{"package": "toy", "version": "1.0", "license": "GPL-3", "views": "A",
                                "keyword": "x"}], ["package", "version", "license", "views", "keyword"])
    out = tmp_path / "f.tsv"
    subprocess.run(["Rscript", os.path.join(ROOT, "workflow", "scripts", "rnls_extract.R"), str(tmp_path),
                    str(tmp_path / "pk.tsv"), str(out)], check=True, capture_output=True)
    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    got = {(r["kind"], r["rhs"].replace(" ", ""), r["regressors"], r["status"]) for r in rows}
    assert ("nls", "Asym/(1+exp((xmid-age)/scal))", "age", "ok") in got
    assert ("nlme", "Asym+(R0-Asym)*exp(-exp(lrc)*age)", "age", "ok") in got
    assert ("selfStart", "Vm*conc/(K+conc)", "conc", "ok") in got
    assert ("nls", "Vm*conc/(K+conc)", "conc", "ok") in got
    assert any(r["status"] == "no parameter names" for r in rows)
