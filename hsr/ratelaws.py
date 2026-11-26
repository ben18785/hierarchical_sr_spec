"""Hold-out source ``ratelaws``: canonical rate-law libraries.

  * Systems Biology Ontology (SBO): every term under SBO:0000001 "rate law"
    that carries a MathML definition (a ``lambda`` whose bound variables are
    annotated with SBO terms). Variables are the bound variables whose term
    is, or descends from, SBO:0000360 "quantity of an entity pool"
    (concentrations and particle numbers of reactants, products, enzymes,
    inhibitors, activators, modifiers), the participant roles SBO:0000010
    "reactant" / SBO:0000015 "substrate", or SBO:0000586 "model time"; every
    other bound variable (rate constants, Michaelis and inhibition constants,
    Hill coefficients, stoichiometries, ...) is a constant.
  * COPASI's built-in function database (``copasi/function/FunctionDB.xml.cpp``
    embeds it as a string): each function has an infix expression and
    parameter roles; variables are the substrates, products and modifiers
    (also volume and time if present). COPASI's mass-action laws use
    ``PRODUCT<substrate_i>`` over a variable list; it is read as one
    substrate (or product) ``substrate_i`` (projecting any product of
    distinct species onto one of them gives the same one-variable form).

One "model" per rate-law definition (``sbo:SBO_xxxxxxx``, ``copasi:<name>``);
each variable in turn becomes x1 (``hsr.corpus_sources.law_keys``), keys are
canonicalised and generalised.
"""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter

import sympy as sp

from . import corpus_sources as cs

MML = "{http://www.w3.org/1998/Math/MathML}"
OWL_CLASS = "{http://www.w3.org/2002/07/owl#}Class"
RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
RDF_RES = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
SBO_URI = "http://biomodels.net/SBO/"

RATE_LAW = "SBO_0000001"
# bound variables of these terms (or their descendants) are variables
VARIABLE_ROOTS = ("SBO_0000360", "SBO_0000010", "SBO_0000015", "SBO_0000586")
COPASI_VARIABLE_ROLES = ("substrate", "product", "modifier", "volume", "time")


class Unsupported(ValueError):
    pass


# ------------------------------------------------------------------ fetching
def fetch(raw_dir, sbo_url, copasi_url, client=None):
    """Download the SBO OWL export and COPASI's function database source
    file (each one GET) into ``raw_dir`` with a manifest (URLs, date, sha256)."""
    client = client or cs.PoliteClient(min_interval=1.5)
    os.makedirs(raw_dir, exist_ok=True)
    files = []
    for name, url in (("SBO_OWL.owl", sbo_url), ("FunctionDB.xml.cpp", copasi_url)):
        data, _ = client.get(url)
        with open(os.path.join(raw_dir, name), "wb") as fh:
            fh.write(data)
        files.append({"file": name, "url": url, "sha256": cs.sha256(data), "bytes": len(data)})
    man = {"source": "SBO rate-law branch (OWL export) and COPASI function database", "date": cs._today(),
           "files": files}
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return man


def _read_checked(raw_dir, man, name):
    f = next(f for f in man["files"] if f["file"] == name)
    with open(os.path.join(raw_dir, name), "rb") as fh:
        data = fh.read()
    if cs.sha256(data) != f["sha256"]:
        raise RuntimeError(f"checksum mismatch: {name}")
    return data


# ------------------------------------------------------------------ MathML
def _sbo_of(url):
    m = re.search(r"SBO:(\d+)", url or "")
    return f"SBO_{m.group(1)}" if m else None


def _cn(el):
    t = el.get("type", "real")
    parts = [p.strip() for p in el.itertext()]
    if t == "e-notation":
        return sp.Float(parts[0]) * 10 ** sp.Integer(int(parts[-1]))
    if t == "rational":
        return sp.Rational(int(parts[0]), int(parts[-1]))
    txt = "".join(parts)
    return sp.Integer(int(txt)) if re.fullmatch(r"[-+]?\d+", txt) else sp.Float(txt)


def _safe(name):
    """Identifier usable in a formula string (SBO uses names such as R*)."""
    return re.sub(r"\W", "_", name) or "_"


def _tag(el):
    """Local name of an element: MathML inside the SBO OWL export is partly in
    the MathML namespace and partly in the document's default (SBO) one."""
    return el.tag.rsplit("}", 1)[-1]


def mathml_to_sympy(el, syms):
    """sympy expression of a content-MathML element; ``syms`` {name: Symbol}
    (identifiers not in it become new positive symbols, added to it).
    Raises Unsupported for indexed sums/products (selector, sum, product),
    piecewise and other constructs outside arithmetic."""
    tag = _tag(el)
    if tag == "ci":
        name = _safe((el.text or "").strip())
        return syms.setdefault(name, sp.Symbol(name, positive=True))
    if tag == "cn":
        return _cn(el)
    if tag in ("semantics", "math"):
        return mathml_to_sympy(next(iter(el)), syms)
    if tag == "apply":
        kids = list(el)
        op = _tag(kids[0])
        if op in ("sum", "product", "selector"):
            raise Unsupported(f"indexed {op}")
        if len(kids) == 1:                    # <apply><ci>k</ci></apply>
            return mathml_to_sympy(kids[0], syms)
        q = {_tag(k): k for k in kids[1:] if _tag(k) in ("degree", "logbase")}
        args = [mathml_to_sympy(k, syms) for k in kids[1:] if _tag(k) not in ("degree", "logbase")]
        if op == "plus":
            return sp.Add(*args)
        if op == "times":
            return sp.Mul(*args)
        if op == "minus":
            return -args[0] if len(args) == 1 else args[0] - args[1]
        if op == "divide":
            return args[0] / args[1]
        if op == "power":
            return args[0] ** args[1]
        if op == "exp":
            return sp.exp(args[0])
        if op == "ln":
            return sp.log(args[0])
        if op == "log":
            base = mathml_to_sympy(next(iter(q["logbase"])), syms) if "logbase" in q else sp.Integer(10)
            return sp.log(args[0]) / sp.log(base)
        if op == "root":
            deg = mathml_to_sympy(next(iter(q["degree"])), syms) if "degree" in q else sp.Integer(2)
            return args[0] ** (1 / deg)
        if op == "tanh":
            return sp.tanh(args[0])
        raise Unsupported(op)
    raise Unsupported(tag)


def lambda_law(math_el):
    """(expression, {bound variable name: SBO term or None}) of an SBO MathML
    definition (``math`` > [``semantics``] > ``lambda``)."""
    lam = next(e for e in math_el.iter() if _tag(e) == "lambda")
    bvars, body = {}, None
    for k in lam:
        if _tag(k) == "bvar":
            ci = next(e for e in k if _tag(e) == "ci")
            bvars[_safe((ci.text or "").strip())] = _sbo_of(ci.get("definitionURL"))
        else:
            body = k
    syms = {n: sp.Symbol(n, positive=True) for n in bvars}
    return mathml_to_sympy(body, syms), bvars


def _ancestors(term, parents):
    out, todo = set(), [term]
    while todo:
        for p in parents.get(todo.pop(), ()):
            if p not in out:
                out.add(p)
                todo.append(p)
    return out


def sbo_terms(owl_bytes):
    """{term: {"label", "parents", "maths": [math elements]}} of the SBO OWL export."""
    root = ET.fromstring(owl_bytes)
    terms = {}
    for c in root.iter(OWL_CLASS):
        tid = (c.get(RDF_ABOUT) or "").replace(SBO_URI, "")
        lab = c.find(RDFS + "label")
        terms[tid] = {"label": (lab.text or "") if lab is not None else "",
                      "parents": [(p.get(RDF_RES) or "").replace(SBO_URI, "")
                                  for p in c.findall(RDFS + "subClassOf") if p.get(RDF_RES)],
                      "maths": [m for sa in c.findall(RDFS + "seeAlso") for m in sa.iter(MML + "math")]}
    return terms


def sbo_laws(owl_bytes):
    """[(model id, label, [(formula string, variables)], drop reason or None)]
    for every rate-law term (descendants of SBO:0000001) with MathML."""
    terms = sbo_terms(owl_bytes)
    parents = {t: v["parents"] for t, v in terms.items()}
    is_var = {}

    def variable(term):
        if term is None:
            return False
        if term not in is_var:
            is_var[term] = term in VARIABLE_ROOTS or bool(_ancestors(term, parents) & set(VARIABLE_ROOTS))
        return is_var[term]

    out = []
    for tid, t in sorted(terms.items()):
        if RATE_LAW not in _ancestors(tid, parents) or not t["maths"]:
            continue
        laws, reason = [], None
        for m in t["maths"]:
            try:
                expr, bvars = lambda_law(m)
            except Unsupported as e:
                reason = f"unsupported MathML: {e}"
                continue
            except (StopIteration, AttributeError, ValueError, TypeError) as e:
                reason = f"malformed MathML: {type(e).__name__}"
                continue
            var = tuple(sorted(n for n, s in bvars.items() if variable(s)))
            if not var:
                reason = "no variable among the bound variables"
                continue
            laws.append((sp.sstr(expr), var))
        out.append((f"sbo:{tid}", t["label"], sorted(set(laws)), None if laws else reason))
    return out


# ------------------------------------------------------------------ COPASI
_MASS_ACTION = re.compile(r"PRODUCT<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>")


def copasi_xml(cpp_bytes):
    """The CopasiML document embedded in FunctionDB.xml.cpp (a sequence of
    raw-string chunks R"xml(...)xml", concatenated by the compiler)."""
    txt = cpp_bytes.decode("utf-8")
    chunks = re.findall(r'R"xml\((.*?)\)xml"', txt, re.S)
    return ("".join(chunks) if chunks else txt).strip()


def copasi_laws(cpp_bytes):
    """[(model id, name, [(formula, variables)], drop reason or None)] of the
    COPASI function database; variables are the parameters with roles in
    COPASI_VARIABLE_ROLES."""
    root = ET.fromstring(copasi_xml(cpp_bytes))
    out = []
    for f in root.iter("Function"):
        name = f.get("name")
        expr = " ".join((f.findtext("Expression") or "").split())
        roles = {p.get("name"): p.get("role") for p in f.iter("ParameterDescription")}
        # PRODUCT<substrate_i> -> substrate (the parameter holding the list)
        expr = _MASS_ACTION.sub(lambda m: m.group(1).rsplit("_", 1)[0]
                                if m.group(1).rsplit("_", 1)[0] in roles else m.group(1), expr)
        var = tuple(sorted(n for n, r in roles.items() if r in COPASI_VARIABLE_ROLES))
        reason = None if var else "no substrate, product or modifier"
        out.append((f"copasi:{name}", name, [(expr, var)] if var else [], reason))
    return out


# ------------------------------------------------------------------ corpus
def build(raw_dir, out, processes=4):
    with open(os.path.join(raw_dir, "manifest.json")) as fh:
        man = json.load(fh)
    defs = (sbo_laws(_read_checked(raw_dir, man, "SBO_OWL.owl"))
            + copasi_laws(_read_checked(raw_dir, man, "FunctionDB.xml.cpp")))
    stats, dropped = Counter(), {}
    for mid, _, laws, reason in defs:
        src = mid.split(":")[0]
        stats[f"{src}_definitions"] += 1
        stats[f"{src}_parsed"] += bool(laws)
        if reason and not laws:
            dropped[mid] = reason
    law_cache = cs.reduce_laws([law for _, _, laws, _ in defs for law in laws], int(processes), stats=stats)
    by_model = {}
    for mid, _, laws, _ in defs:
        ks = set().union(*(law_cache[law] for law in laws)) if laws else set()
        if laws and not ks:
            dropped[mid] = "no one-variable key (parse or reduction failed)"
        stats[f"{mid.split(':')[0]}_reduced"] += bool(ks)
        if ks:
            by_model[mid] = ks
    stats["dropped"] = len(dropped)
    res = cs._corpus_dict(by_model, dict(stats))
    res.update(source="Canonical rate-law libraries: the Systems Biology Ontology rate-law branch (MathML of "
                      "SBO:0000001 and descendants) and COPASI's built-in function database; one model per "
                      "rate-law definition; each concentration/amount (SBO) or substrate/product/modifier "
                      "(COPASI) in turn is x1; literals generalised",
               dropped=dict(sorted(dropped.items())),
               labels={mid: lab for mid, lab, _, _ in defs},
               snapshot={"date": man["date"], "files": man["files"]})
    cs._write(res, out)
    return res
