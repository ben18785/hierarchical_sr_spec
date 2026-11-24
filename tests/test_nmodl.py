"""NMODL equation extraction (hsr.nmodl) for the ModelDB corpus; no network."""
import io
import json
import zipfile

import pytest

from hsr import corpus_sources as cs
from hsr.nmodl import Module, _split_assignments, statements, strip_comments

MOD = """
TITLE test channel
COMMENT
  a = should_not_appear(v)
ENDCOMMENT
UNITS { (mV) = (millivolt) }
NEURON {
    SUFFIX tst
    USEION ca READ cai, cao WRITE ica
    RANGE gbar
}
PARAMETER { gbar = 0.01 (mho/cm2)  celsius = 6.3 (degC) }
STATE { m h }
ASSIGNED { ica (mA/cm2) minf mtau (ms) }
BREAKPOINT {
    SOLVE states METHOD cnexp
    ica = gbar*m*m*h*(v - 120 (mV))   : a comment
}
DERIVATIVE states {
    rates(v)
    m' = (minf - m)/mtau
}
PROCEDURE rates(v(mV)) {
    LOCAL a, b, q10
    TABLE minf, mtau FROM -100 TO 100 WITH 200
    q10 = 3^((celsius - 6.3)/10)
    a = 0.1*vtrap(-(v + 40), 10)  b = 4*exp(-(v + 65)/18)
    mtau = 1/(q10*(a + b))
    minf = a/(a + b)
}
FUNCTION vtrap(x, y) {
    if (fabs(x/y) < 1e-6) {
        vtrap = y*(1 - x/y/2)
    } else {
        vtrap = x/(exp(x/y) - 1)
    }
}
FUNCTION hinf(v) { hinf = 1/(1 + exp((v + 1.e-6)/7.)) }
"""


def test_strip_comments_and_statements():
    text = strip_comments(MOD)
    assert "should_not_appear" not in text and "a comment" not in text
    st = statements("if (x > 1) { y = 2 } else { y = 3 }\n a = f(v)  b=g(v)\n LOCAL q\n TABLE m FROM 1 TO 2")
    assert st == [("y", "2"), ("y", "3"), ("a", "f(v)"), ("b", "g(v)")]
    assert _split_assignments("tau = 1/(a + b) minf = a*tau") == ["tau = 1/(a + b)", "minf = a*tau"]
    assert _split_assignments("x = (a == b)") == ["x = (a == b)"]


def test_module_equations_inline_functions_and_locals():
    m = Module(MOD)
    assert m.variables == {"v", "t", "m", "h", "cai", "cao"}
    eqs, errors = m.equations()
    assert errors == []
    keys = {}
    for rhs, var in eqs:
        for k in cs.law_keys(rhs, var):
            keys.setdefault(tuple(var), set()).add(k)
    flat = {k for ks in keys.values() for k in ks}
    # vtrap's else branch, inlined: the Hodgkin-Huxley alpha_m
    assert "(c0 + c1*x1)/(c2 + exp(c3*x1))" in flat
    # the unit annotation 120 (mV) is a number; ica is linear in v and quadratic in m
    assert "c0*x1 + c1" in flat and "c0*x1**2" in flat
    # 1.e-6 is a number (not an identifier e), hinf a Boltzmann function
    assert "c0/(c1 + exp(c2*x1))" in flat
    # celsius is a constant, so q10 is: mtau is a function of v only
    assert all("n_celsius" not in v for _, v in eqs)


def test_modeldb_fetch_equations_build(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model/chan.mod", MOD)
        zf.writestr("model/copy.mod", MOD)            # identical content: parsed once
        zf.writestr("model/run.hoc", "load_file()")
        zf.writestr("__MACOSX/model/._chan.mod", "junk")
    data = buf.getvalue()

    class Fake:
        def get(self, url):
            if "/api/v1/models" in url:
                return json.dumps([12, 7]).encode(), {}
            return data, {}

    monkeypatch.setattr(cs, "PoliteClient", lambda **kw: Fake())
    man = cs.modeldb_fetch(str(tmp_path / "raw"))
    assert [e["id"] for e in man["entries"]] == [7, 12]
    assert man["entries"][0]["mod_files"] == ["model/chan.mod", "model/copy.mod"]
    E = cs.modeldb_equations(str(tmp_path / "raw"), str(tmp_path / "eq.json"), processes=1)
    assert E["n_files"] == 4 and E["n_distinct_files"] == 1
    res = cs.modeldb_build(str(tmp_path / "eq.json"), str(tmp_path / "c.json"), processes=1)
    assert set(res["per_model"]) == {"ModelDB:7", "ModelDB:12"}
    assert "(c0 + c1*x1)/(c2 + exp(c3*x1))" in res["per_model"]["ModelDB:7"]


def test_unparseable_assignment_is_reported_not_fatal():
    m = Module("PROCEDURE p(v) { LOCAL a\n a = v[0] + 1\n b = 2*v\n c = (v > 1)\n }")
    eqs, errors = m.equations()
    assert [e for e, _ in eqs] == ["2*n_v"]
    assert len(errors) == 2
