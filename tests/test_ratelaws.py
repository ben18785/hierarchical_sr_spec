"""Rate-law libraries (hsr.ratelaws): SBO MathML and COPASI function database; no network."""
import hashlib
import json

import sympy as sp

from hsr import ratelaws as rl

# a fragment of the SBO OWL export: MathML partly in the default (SBO) namespace
OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://biomodels.net/SBO/" xml:base="http://biomodels.net/SBO/"
     xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:mathml="http://www.w3.org/1998/Math/MathML">
 <owl:Class rdf:about="http://biomodels.net/SBO/SBO_0000001"><rdfs:label>rate law</rdfs:label></owl:Class>
 <owl:Class rdf:about="http://biomodels.net/SBO/SBO_0000360"><rdfs:label>quantity of an entity pool</rdfs:label></owl:Class>
 <owl:Class rdf:about="http://biomodels.net/SBO/SBO_0000196">
  <rdfs:subClassOf rdf:resource="http://biomodels.net/SBO/SBO_0000360"/><rdfs:label>concentration</rdfs:label></owl:Class>
 <owl:Class rdf:about="http://biomodels.net/SBO/SBO_0000515">
  <rdfs:subClassOf rdf:resource="http://biomodels.net/SBO/SBO_0000196"/><rdfs:label>conc of substrate</rdfs:label></owl:Class>
 <owl:Class rdf:about="http://biomodels.net/SBO/SBO_0000027"><rdfs:label>Michaelis constant</rdfs:label></owl:Class>
 <owl:Class rdf:about="http://biomodels.net/SBO/SBO_0000028">
  <rdfs:subClassOf rdf:resource="http://biomodels.net/SBO/SBO_0000001"/>
  <rdfs:label>MM</rdfs:label>
  <rdfs:seeAlso rdf:parseType="Literal"><mathml:math xmlns:mathml="http://www.w3.org/1998/Math/MathML">
   <semantics definitionURL="http://biomodels.net/SBO/#SBO:0000062"><lambda>
    <bvar><ci definitionURL="http://biomodels.net/SBO/#SBO:0000025">V</ci></bvar>
    <bvar><ci definitionURL="http://biomodels.net/SBO/#SBO:0000515">S</ci></bvar>
    <bvar><ci definitionURL="http://biomodels.net/SBO/#SBO:0000027">Km</ci></bvar>
    <apply><divide/><apply><times/><ci>V</ci><apply><power/><ci>S</ci><cn type="integer">2</cn></apply></apply>
     <apply><plus/><ci>Km</ci><ci>S</ci></apply></apply>
   </lambda></semantics></mathml:math></rdfs:seeAlso>
 </owl:Class>
 <owl:Class rdf:about="http://biomodels.net/SBO/SBO_0000047">
  <rdfs:subClassOf rdf:resource="http://biomodels.net/SBO/SBO_0000028"/>
  <rdfs:label>zeroth order</rdfs:label>
  <rdfs:seeAlso rdf:parseType="Literal"><mathml:math xmlns:mathml="http://www.w3.org/1998/Math/MathML">
   <lambda><bvar><ci definitionURL="http://biomodels.net/SBO/#SBO:0000048">k</ci></bvar>
    <apply><ci>k</ci></apply></lambda></mathml:math></rdfs:seeAlso>
 </owl:Class>
 <owl:Class rdf:about="http://biomodels.net/SBO/SBO_0000163">
  <rdfs:subClassOf rdf:resource="http://biomodels.net/SBO/SBO_0000001"/>
  <rdfs:label>indexed</rdfs:label>
  <rdfs:seeAlso rdf:parseType="Literal"><mathml:math xmlns:mathml="http://www.w3.org/1998/Math/MathML">
   <lambda><bvar><ci definitionURL="http://biomodels.net/SBO/#SBO:0000515">R</ci></bvar>
    <apply><product/><bvar><ci>i</ci></bvar><ci>R</ci></apply></lambda></mathml:math></rdfs:seeAlso>
 </owl:Class>
 <owl:Class rdf:about="http://biomodels.net/SBO/SBO_0000999">
  <rdfs:label>not a rate law</rdfs:label>
  <rdfs:seeAlso rdf:parseType="Literal"><mathml:math xmlns:mathml="http://www.w3.org/1998/Math/MathML">
   <lambda><bvar><ci definitionURL="http://biomodels.net/SBO/#SBO:0000515">S</ci></bvar><ci>S</ci></lambda>
  </mathml:math></rdfs:seeAlso>
 </owl:Class>
</rdf:RDF>
"""

CPP = """#include <string>
static const std::string functionDBxml = {
  R"xml(<?xml version='1.0' encoding='UTF-8'?>
<COPASI><ListOfFunctions>
  <Function key='Function_0' name='Henri-Michaelis-Menten (irreversible)' type='PreDefined'>
  <Expression>
  V*substrate/(Km+substrate)
  </Expression>
  <ListOfParameterDescriptions>
  <ParameterDescription key='P0' name='substrate' order='0' role='substrate'/>
  <ParameterDescription key='P1' name='Km' order='1' role='constant'/>
  <ParameterDescription key='P2' name='V' order='2' role='constant'/>
  </ListOfParameterDescriptions>
  </Function>)xml"
  R"xml("<Function key='Function_1' name='Mass action (reversible)' type='MassAction'>
  <Expression>
  k1*PRODUCT&lt;substrate_i&gt;-k2*PRODUCT&lt;product_j&gt;
  </Expression>
  <ListOfParameterDescriptions>
  <ParameterDescription key='P3' name='k1' order='0' role='constant'/>
  <ParameterDescription key='P4' name='substrate' order='1' role='substrate'/>
  <ParameterDescription key='P5' name='k2' order='2' role='constant'/>
  <ParameterDescription key='P6' name='product' order='3' role='product'/>
  </ListOfParameterDescriptions>
  </Function>
  <Function key='Function_2' name='Constant flux' type='PreDefined'>
  <Expression>v</Expression>
  <ListOfParameterDescriptions>
  <ParameterDescription key='P7' name='v' order='0' role='constant'/>
  </ListOfParameterDescriptions>
  </Function>
</ListOfFunctions></COPASI>)xml"};
"""


def test_sbo_laws_variables_and_drops():
    laws = {m: (laws, reason) for m, _, laws, reason in rl.sbo_laws(OWL.encode())}
    assert set(laws) == {"sbo:SBO_0000028", "sbo:SBO_0000047", "sbo:SBO_0000163"}   # rate-law branch only
    (formula, var), = laws["sbo:SBO_0000028"][0]
    assert var == ("S",)                                   # a concentration; V and Km are constants
    S, V, Km = sp.symbols("S V Km", positive=True)
    assert sp.simplify(sp.sympify(formula, locals={"S": S, "V": V, "Km": Km}) - V * S**2 / (Km + S)) == 0
    assert laws["sbo:SBO_0000047"] == ([], "no variable among the bound variables")
    assert laws["sbo:SBO_0000163"] == ([], "unsupported MathML: indexed product")


def test_mathml_e_notation_and_log():
    import xml.etree.ElementTree as ET
    m = ET.fromstring('<apply xmlns="http://www.w3.org/1998/Math/MathML"><log/><logbase><cn>2</cn></logbase>'
                      '<apply><times/><cn type="e-notation">1.5<sep/>3</cn><ci>x</ci></apply></apply>')
    syms = {}
    e = rl.mathml_to_sympy(m, syms)
    assert abs(float(e.subs(syms["x"], 2)) - __import__("math").log2(3000)) < 1e-9


def test_copasi_laws_chunks_and_mass_action():
    laws = {m: (laws, reason) for m, _, laws, reason in rl.copasi_laws(CPP.encode())}
    assert laws["copasi:Henri-Michaelis-Menten (irreversible)"][0] == [("V*substrate/(Km+substrate)", ("substrate",))]
    assert laws["copasi:Mass action (reversible)"][0] == [("k1*substrate-k2*product", ("product", "substrate"))]
    assert laws["copasi:Constant flux"] == ([], "no substrate, product or modifier")


def test_build(tmp_path):
    files = []
    for name, txt in (("SBO_OWL.owl", OWL), ("FunctionDB.xml.cpp", CPP)):
        (tmp_path / name).write_text(txt)
        files.append({"file": name, "url": "u", "sha256": hashlib.sha256(txt.encode()).hexdigest()})
    (tmp_path / "manifest.json").write_text(json.dumps({"date": "2026-01-01", "files": files}))
    out = tmp_path / "corpus.json"
    res = rl.build(str(tmp_path), str(out), 1)
    assert res["per_model"]["copasi:Henri-Michaelis-Menten (irreversible)"] == ["c0*x1/(c1 + x1)"]
    assert res["per_model"]["sbo:SBO_0000028"] == ["c0*x1**2/(c1 + x1)"]
    assert res["per_model"]["copasi:Mass action (reversible)"] == ["c0*x1 + c1"]
    assert res["stats"]["sbo_definitions"] == 3 and res["stats"]["copasi_definitions"] == 3
    assert set(res["dropped"]) == {"sbo:SBO_0000047", "sbo:SBO_0000163", "copasi:Constant flux"}
    assert json.loads(out.read_text())["freq"]["c0*x1/(c1 + x1)"] == 1
