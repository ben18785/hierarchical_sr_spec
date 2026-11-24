"""Parsing of the further corpora (SABIO-RK, CellML, curated catalogue); no network."""
import gzip
import hashlib
import importlib.util
import io
import json
import os
import urllib.error
import urllib.parse

import pytest
import sympy as sp

from hsr import corpus_sources as cs
from hsr.corpus import _to_sympy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_scientific_notation_is_a_number():
    expr, syms = _to_sympy("2.5e-3*S + 1E5 + K*.5e+2")
    assert set(syms) == {"S", "K"}
    assert abs(float(expr.subs({syms["S"]: 1, syms["K"]: 2})) - (0.0025 + 100 + 100000)) < 1e-9


def test_tidy_half_integer_floats_and_power_of_exp_log():
    s = sp.Symbol("s", positive=True)
    n = sp.Symbol("n", positive=True)
    assert cs.tidy(s**sp.Float(0.5)) == sp.sqrt(s)
    assert cs.tidy(sp.Float(2.0) * s**sp.Float(2.0)) == 2 * s**2
    assert cs.tidy(sp.exp(n * sp.log(s))) == s**n


def test_law_keys_one_key_per_variable():
    assert cs.law_keys("E*kcat*S/(Km+S)", ["E", "S"]) == {"c0*x1/(c1 + x1)", "c0*x1"}
    assert cs.law_keys("Vmax*A/(Km+A+(A^2)/Ki)", ["A"]) == {"c0*x1/(c1*x1**2 + c2 + x1)"}
    # S^0.5 is a square root, not a free power
    assert cs.law_keys("V*S^0.5/(K+S^0.5)", ["S"]) == {"c0*sqrt(x1)/(c1 + sqrt(x1))"}
    assert cs.law_keys("k1(1+S)", ["S"]) == set()       # implicit call: unparseable


def _entry(i, pub, formula, variables):
    return {"id": i, "publication": {"publication_id": pub},
            "kineticlaw": {"formula": formula,
                           "parameter": [{"name": v, "role": "Variable"} for v in variables]
                           + [{"name": "Km", "role": "Constant"}]}}


def _raw_dir(tmp_path, entries, corrupt=False):
    data = json.dumps({"meta": {"total_count": len(entries)}, "data": entries}).encode()
    with gzip.open(tmp_path / "entries_000001_001000.json.gz", "wb") as fh:
        fh.write(data + (b" " if corrupt else b""))
    man = {"date": "2026-01-01", "api": cs.SABIORK_API, "total_count": len(entries), "sha256_all": "x",
           "files": [{"file": "entries_000001_001000.json.gz", "sha256": hashlib.sha256(data).hexdigest()}]}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return str(tmp_path)


def test_sabiork_law_and_build(tmp_path):
    entries = [_entry(1, 7, "Vmax*S/(Km+S)", ["S"]), _entry(2, 7, "V*S/(K+S)", ["S", "A"]),
               _entry(3, 8, "Vmax*S/(Km+S)", ["S"]), {"id": 4, "kineticlaw": {"formula": None}}]
    assert cs.sabiork_law(entries[1]) == ("V*S/(K+S)", ["A", "S"])
    assert cs.sabiork_law(entries[3]) is None
    out = tmp_path / "c.json"
    res = cs.sabiork_build(_raw_dir(tmp_path, entries), str(out), processes=1)
    # one model per publication, each distinct form once per model
    assert res["per_model"] == {"SABIORK_PUB7": ["c0*x1/(c1 + x1)"], "SABIORK_PUB8": ["c0*x1/(c1 + x1)"]}
    assert res["freq"] == {"c0*x1/(c1 + x1)": 2}
    assert res["stats"]["entries"] == 4 and res["stats"]["entries_with_formula"] == 3
    assert json.loads(out.read_text())["per_model"] == res["per_model"]


def test_sabiork_build_checks_checksums(tmp_path):
    with pytest.raises(RuntimeError, match="checksum"):
        cs.sabiork_build(_raw_dir(tmp_path, [_entry(1, 7, "V*S/(K+S)", ["S"])], corrupt=True),
                         str(tmp_path / "c.json"), processes=1)


def _extract_module():
    spec = importlib.util.spec_from_file_location("cellml_extract",
                                                  os.path.join(ROOT, "workflow", "scripts", "cellml_extract.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cellml_generated_code_to_equations():
    code = """
def compute_computed_constants(voi, states, rates, constants, computed_constants, algebraic_variables):
    computed_constants[0] = constants[0]/constants[1]

def compute_rates(voi, states, rates, constants, computed_constants, algebraic_variables):
    algebraic_variables[0] = 0.1*(states[0]+25.0)/(exp((states[0]+25.0)/10.0)-1.0)
    rates[0] = algebraic_variables[0]*(1.0-states[1])-computed_constants[0]*states[1]

def compute_variables(voi, states, rates, constants, computed_constants, algebraic_variables):
    algebraic_variables[0] = 0.1*(states[0]+25.0)/(exp((states[0]+25.0)/10.0)-1.0)
    algebraic_variables[1] = constants[2] if lt_func(voi, 1.0) else 0.0
"""
    eqs = _extract_module().equations_of_code(code)
    assert "0.1*(x_s0+25.0)/(exp((x_s0+25.0)/10.0)-1.0)" in eqs
    assert "x_a0*(1.0-x_s1)-p_c0*x_s1" in eqs
    assert len(eqs) == 3                    # deduplicated; computed constants skipped
    rhs, var = cs.cellml_law("x_a0*(1.0-x_s1)-p_c0*x_s1")
    assert var == ["x_a0", "x_s1"]
    assert cs.law_keys(*cs.cellml_law("0.1*(x_s0+25.0)/(exp((x_s0+25.0)/10.0)-1.0)")) == \
        {"(c0 + c1*x1)/(c2 + exp(c3*x1))"}
    # piecewise definitions are not expressible: no key
    assert cs.law_keys(*cs.cellml_law("p_k2 if lt_func(x_voi, 1.0) else 0.0")) == set()


def test_cellml_build(tmp_path):
    E = {"date": "2026-01-01", "sha256_all": "x", "libcellml": "0.7.2",
         "workspaces": {"w1": [{"file": "a.cellml", "status": "ok", "equations": ["p_k0*x_s0/(p_k1+x_s0)"]},
                               {"file": "b.cellml", "status": "analysis failed", "equations": []}],
                        "w2": [{"file": "c.cellml", "status": "unresolved imports", "equations": []}]}}
    p = tmp_path / "eq.json"
    p.write_text(json.dumps(E))
    res = cs.cellml_build(str(p), str(tmp_path / "c.json"), processes=1)
    assert res["per_model"] == {"PMR:w1": ["c0*x1/(c1 + x1)"]}
    assert res["stats"]["workspaces"] == 2 and res["stats"]["workspaces_ok"] == 1
    assert res["stats"]["files_analysis"] == 1 and res["stats"]["files_unresolved"] == 1


ROW_PK2 = {"id": "pk2", "params": "D,V,k10,k12,k21",
           "defs": "s=k10 + k12 + k21; al=(s + sqrt(s**2 - 4*k10*k21))/2; be=(s - sqrt(s**2 - 4*k10*k21))/2",
           "formula": "D/V*((al - k21)/(al - be)*exp(-al*x) + (k21 - be)/(al - be)*exp(-be*x))",
           "check": "library"}


def test_catalogue_expr_and_key():
    expr, x, p = cs.catalogue_expr(ROW_PK2)
    assert expr.free_symbols == {x, *p.values()}
    # at t = 0 the concentration is D/V
    assert sp.simplify(expr.subs(x, 0) - p["D"] / p["V"]) == 0
    assert cs.catalogue_key(ROW_PK2) == "c0*exp(c1*x1) + c2*exp(c3*x1)"
    row = {"params": "Vmax,Km", "defs": "", "formula": "Vmax*x/(Km + x)"}
    assert cs.catalogue_key(row) == "c0*x1/(c1 + x1)"
    lw = {"params": "a,h,T", "defs": "", "formula": "x - LambertW(a*h*x*exp(-a*(T - h*x)))/(a*h)"}
    assert cs.catalogue_key(lw) is None


def test_catalogue_verify():
    row = {"params": "Vmax,Km", "defs": "", "formula": "Vmax*x/(Km + x)"}
    good = [{"x": str(x), "params": "2,1", "y": str(2 * x / (1 + x))} for x in (0.5, 1, 2, 3, 4, 5)]
    assert cs.catalogue_verify(row, good)[0]
    bad = [dict(e, y=str(float(e["y"]) * 1.01)) for e in good]
    assert not cs.catalogue_verify(row, bad)[0]
    assert not cs.catalogue_verify(row, good[:3])[0]        # too few points


def test_catalogue_file_is_well_formed():
    rows = cs.catalogue_rows(os.path.join(ROOT, "hsr", "data", "sources", "bio_catalogue.csv"))
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert r["check"] in ("library", "ode", "none")
        assert bool(r["r_call"]) == (r["check"] != "none"), r["id"]
        assert r["reference"], r["id"]
        cs.catalogue_expr(r)            # parses


def test_make_transfer_uses_every_form_of_a_model(tmp_path):
    from hsr.grammar_compare import make_transfer
    T = {"per_model": {"m1": ["c0*x1/(c1 + x1)", "c0*x1"], "m2": ["c0*x1"]}}
    p = tmp_path / "corpus_t.json"
    p.write_text(json.dumps(T))
    train = tmp_path / "corpus.json"
    train.write_text(json.dumps({"per_model": {"b1": ["c0*x1"], "b2": ["c0*exp(c1*x1)"]}}))
    cand = tmp_path / "candidates.json"
    cand.write_text(json.dumps({"keys": ["c0*x1"]}))
    F = make_transfer({}, str(p), corpus=str(train), candidates=str(cand))
    assert sorted(F["folds"][0]["test"]) == ["c0*x1", "c0*x1", "c0*x1/(c1 + x1)"]
    assert sorted(F["folds"][0]["train"]) == ["c0*exp(c1*x1)", "c0*x1"]


class _Resp(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_polite_client_retries_after_429(monkeypatch):
    calls, sleeps = [], []

    def fake_urlopen(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "3"}, None)
        return _Resp(b"ok")

    monkeypatch.setattr(cs.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cs.time, "sleep", lambda s: sleeps.append(s))
    data, _ = cs.PoliteClient(min_interval=0).get("https://example.org/x")
    assert data == b"ok" and len(calls) == 2 and 3 in sleeps
    assert calls[0].get_header("User-agent") == cs.USER_AGENT


def test_polite_client_does_not_retry_404(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(cs.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="404"):
        cs.PoliteClient(min_interval=0).get("https://example.org/x")
    assert len(calls) == 1


# ------------------------------------------------------------ R curve catalogues
def _curve_row(id_, model, formula, params="a,b", defs="", r_call="f(x)"):
    return {"id": id_, "package": "pkg", "function": id_, "model": model, "params": params, "defs": defs,
            "formula": formula, "r_call": r_call}


def _evals(id_, f, params=(1.3, 0.7), xs=None):
    xs = xs if xs is not None else [0.3 + 0.15 * i for i in range(25)]
    return [{"func": id_, "x": repr(x), "params": ",".join(map(str, params)), "y": repr(f(x, *params))} for x in xs]


def test_curve_verify_relative_to_y_not_one():
    import math
    row = _curve_row("r", "m", "a*exp(-b/(8.62e-5*(x + 273.15)))")
    good = _evals("r", lambda x, a, b: a * math.exp(-b / (8.62e-5 * (x + 273.15))))
    assert cs.curve_verify(row, good)[0]
    # a 1% error in values ~1e-20 passes catalogue_verify's max(1, |y|) scale but not curve_verify
    bad = [dict(e, y=repr(1.01 * float(e["y"]))) for e in good]
    assert cs.catalogue_verify(row, bad)[0]
    assert not cs.curve_verify(row, bad)[0]
    # zeros and NA from R are skipped; too few points is a failure
    few = [dict(e, y="NA") for e in good[:22]] + good[22:]
    assert cs.curve_verify(row, few) == (False, 3, cs.curve_verify(row, few)[2])


def test_curves_build_groups_rows_by_model(tmp_path):
    import csv
    import math
    rows = [_curve_row("p1", "Briere1999", "a*x*(x - b)*sqrt(3 - x)"),
            _curve_row("p2", "Briere1999", "a*x*(x - b)*sqrt(3 - x)"),       # same model, same form: counted once
            _curve_row("p3", "Arrhenius", "a*exp(-b/x)"),
            _curve_row("p4", "Wrong", "a*exp(-b*x)"),                         # disagrees with R: excluded
            _curve_row("p5", "NoRef", "a + b*x", r_call="")]                 # no implementation: kept, unverified
    src = tmp_path / "t_curves.csv"
    with open(src, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="|")
        w.writeheader()
        w.writerows(rows)
    br = lambda x, a, b: a * x * (x - b) * math.sqrt(3 - x)
    ev = (_evals("p1", br, xs=[0.3 + 0.1 * i for i in range(25)]) + _evals("p2", br, xs=[0.3 + 0.1 * i for i in range(25)])
          + _evals("p3", lambda x, a, b: a * math.exp(-b / x)) + _evals("p4", lambda x, a, b: a * math.exp(-b * x * 1.1)))
    evp = tmp_path / "ev.csv"
    with open(evp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["func", "x", "params", "y"])
        w.writeheader()
        w.writerows(ev)
    (tmp_path / "ev.csv.versions").write_text("R=4.5.1;pkg=1.0")
    res = cs.curves_build(str(src), str(evp), str(tmp_path / "c.json"))
    assert set(res["per_model"]) == {"Briere1999", "Arrhenius", "NoRef"}
    assert len(res["per_model"]["Briere1999"]) == 1
    assert res["per_model"]["Arrhenius"] == ["c0*exp(c1/x1)"]
    v = res["verification"]
    assert v["p1"]["verified"] and not v["p4"]["verified"] and not v["p4"]["in_corpus"]
    assert v["p5"]["in_corpus"] and not v["p5"]["verified"]
    assert res["stats"]["models"] == 3 and res["versions"] == "R=4.5.1;pkg=1.0"


@pytest.mark.parametrize("name", ["thermal", "ecocurves"])
def test_committed_curve_catalogues_are_verified(name):
    """Every row of the committed curve catalogues with an R implementation
    agreed with it (rows without one are marked unverified)."""
    rows = cs.catalogue_rows(os.path.join(ROOT, "hsr", "data", "sources", f"{name}_curves.csv"))
    with open(os.path.join(ROOT, "hsr", "data", f"corpus_{name}.json")) as fh:
        d = json.load(fh)
    assert {r["id"] for r in rows} == set(d["verification"])
    for r in rows:
        v = d["verification"][r["id"]]
        assert v["formula"] == r["formula"]
        assert v["verified"] == bool(r["r_call"].strip()), r["id"]
    keys = {k for ks in d["per_model"].values() for k in ks}
    assert keys == set(d["freq"]) and all(cs._expressible(k) for k in keys)


# -------------------------------------------------------------------- overlap
def test_overlap_table(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"per_model": {"m1": ["raw1", "f2"], "m2": ["f2"], "m3": ["bad"]}}))
    b.write_text(json.dumps({"per_model": {"n1": ["f2", "f3"]}}))
    gen = {"raw1": "f1", "bad": None}
    md = cs.overlap_table({"A": str(a), "B": str(b)}, gen, ["A"])
    assert "| A | 2 | 3 | 2 | 1 | - |" in md            # m3 has no parseable form
    assert "| B | 1 | 2 | 2 | 1 | 1 |" in md            # f3 is new relative to the baseline A
    assert "| A | 2 | 1 |" in md and "| B | 1 | 2 |" in md


# ------------------------------------------------------- BioModels by query
_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="m">
  <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
   <species id="P" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="V" value="1" constant="true"/><parameter id="K" value="0.5" constant="true"/>
  </listOfParameters>
  <listOfReactions>
   <reaction id="r" reversible="false">
    <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
    <listOfProducts><speciesReference species="P" stoichiometry="1" constant="true"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><divide/><apply><times/><ci>V</ci><ci>S</ci></apply><apply><plus/><ci>K</ci><ci>S</ci></apply></apply>
    </math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
""".encode()


class _FakeBioModels:
    """Stands in for PoliteClient: search pages, model records and files."""

    def __init__(self, models):
        self.models = models          # {id: bytes or None (no SBML)}
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        if "/search?" in url:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            off, n = int(q["offset"][0]), int(q["numResults"][0])
            ids = sorted(self.models)
            return json.dumps({"matches": len(ids), "models": [{"id": i} for i in ids[off:off + n]]}).encode(), {}
        if "/model/download/" in url:
            mid = urllib.parse.urlparse(url).path.rsplit("/", 1)[1]
            return self.models[mid], {}
        mid = urllib.parse.urlparse(url).path.rsplit("/", 1)[1]
        data = self.models[mid]
        main = [] if data is None else [{"name": f"{mid}_file.xml", "sha256sum": hashlib.sha256(data).hexdigest()}]
        return json.dumps({"name": f"model {mid}", "curationStatus": "NON_CURATED",
                           "files": {"main": main, "additional": [{"name": "x.cps"}]}}).encode(), {}


def test_biomodels_main_sbml():
    info = {"files": {"main": [{"name": "a.cps"}, {"name": "M.xml", "sha256sum": "ab"}]}}
    assert cs.biomodels_main_sbml(info) == ("M.xml", "ab")
    assert cs.biomodels_main_sbml({"files": {"main": [{"name": "m.m"}]}}) is None


def test_biomodels_fetch_and_build(tmp_path, monkeypatch):
    pytest.importorskip("libsbml")
    import urllib.parse  # noqa: F401  (used by the fake client)
    fake = _FakeBioModels({"MODEL1": _SBML, "MODEL2": None, "MODEL3": _SBML})
    monkeypatch.setattr(cs, "PoliteClient", lambda **kw: fake)
    man = cs.biomodels_fetch(str(tmp_path), query="q")
    ent = {e["id"]: e for e in man["entries"]}
    assert ent["MODEL2"]["error"] == "no main SBML file"
    assert ent["MODEL1"]["sha256"] == ent["MODEL1"]["sha256_server"]
    assert "filename=MODEL1_file.xml" in ent["MODEL1"]["url"]
    # second run: files present with the server's checksum are not downloaded again
    n = sum("/model/download/" in u for u in fake.urls)
    cs.biomodels_fetch(str(tmp_path), query="q")
    assert sum("/model/download/" in u for u in fake.urls) == n
    res = cs.biomodels_build(str(tmp_path), str(tmp_path / "c.json"), processes=1)
    assert res["per_model"] == {"MODEL1": ["c0*x1/(c1 + x1)"], "MODEL3": ["c0*x1/(c1 + x1)"]}
    assert res["stats"]["not_downloaded"] == 1 and res["stats"]["models_read"] == 2


def test_biomodels_fetch_rejects_checksum_mismatch(tmp_path, monkeypatch):
    fake = _FakeBioModels({"MODEL1": _SBML})
    orig = fake.get
    fake.get = lambda url: (b"corrupt", {}) if "/model/download/" in url else orig(url)
    monkeypatch.setattr(cs, "PoliteClient", lambda **kw: fake)
    man = cs.biomodels_fetch(str(tmp_path), query="q")
    assert man["entries"][0]["error"] == "checksum differs from the server's"
    assert not (tmp_path / "MODEL1.xml").exists()
