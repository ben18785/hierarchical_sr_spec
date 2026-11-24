"""Extract the equations of CellML models with libcellml (cellml environment).

    python cellml_extract.py all <raw_dir> <out.json> [processes]
    python cellml_extract.py one <file.cellml>          (worker; prints JSON)

For each .cellml file of each workspace in the snapshot manifest, the model is
parsed (CellML 1.0/1.1 are translated to 2.0 by the non-strict parser), its
imports are resolved from the workspace's local files and flattened, and it is
analysed and turned into Python code by libcellml's generator. Every
assignment of compute_rates/compute_variables becomes one equation: its
right-hand side with states, algebraic variables and the variable of
integration renamed x_<k>/voi (variables) and constants and computed
constants renamed p_<k> (constants). Each file runs in its own process with a
time limit, since flattening or analysing a few very large models can stall.
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

TIMEOUT = 120
MAX_FILES = 60      # per workspace

_ARR = re.compile(r"\b(states|algebraic_variables|algebraic|constants|computed_constants|external_variables|external)\[(\d+)\]")


def _rename(m):
    kind, i = m.group(1), m.group(2)
    if kind in ("states",):
        return f"x_s{i}"
    if kind in ("algebraic_variables", "algebraic"):
        return f"x_a{i}"
    if kind in ("external_variables", "external"):
        return f"x_e{i}"
    if kind == "computed_constants":
        return f"p_c{i}"
    return f"p_k{i}"


def equations_of_code(code):
    """[rhs] of the assignments in compute_rates and compute_variables."""
    out, keep = [], False
    for line in code.splitlines():
        if line.startswith("def "):
            keep = line.startswith("def compute_rates") or line.startswith("def compute_variables")
            continue
        s = line.strip()
        if not keep or " = " not in s or not re.match(r"^(rates|algebraic_variables|algebraic|states|external_variables)\[\d+\] = ", s):
            continue
        rhs = s.split(" = ", 1)[1]
        rhs = _ARR.sub(_rename, rhs)
        rhs = re.sub(r"\bvoi\b", "x_voi", rhs)
        out.append(rhs)
    return sorted(set(out))


def one(path):
    import libcellml as lc
    path = os.path.abspath(path)
    res = {"file": path, "status": None, "equations": []}
    p = lc.Parser(False)
    with open(path, encoding="utf-8", errors="replace") as fh:
        m = p.parseModel(fh.read())
    if m is None or p.errorCount():
        res["status"] = "parse error"
        return res
    if m.hasUnresolvedImports():
        imp = lc.Importer(False)
        imp.resolveImports(m, os.path.dirname(path) + "/")
        if m.hasUnresolvedImports() or imp.errorCount():
            res["status"] = "unresolved imports"
            return res
        m = imp.flattenModel(m)
        if m is None:
            res["status"] = "flatten failed"
            return res
    a = lc.Analyser()
    a.analyseModel(m)
    am = a.analyserModel()
    if a.errorCount() or am is None or not am.isValid():
        res["status"] = "analysis failed"
        res["detail"] = a.error(0).description()[:200] if a.errorCount() else ""
        return res
    code = lc.Generator().implementationCode(am, lc.GeneratorProfile(lc.GeneratorProfile.Profile.PYTHON))
    res["equations"] = equations_of_code(code)
    res["status"] = "ok"
    return res


def _run(path):
    try:
        r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "one", path],
                           capture_output=True, text=True, timeout=TIMEOUT)
        if r.returncode != 0:
            return {"file": path, "status": "crashed", "equations": []}
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return {"file": path, "status": "timeout", "equations": []}
    except Exception as e:
        return {"file": path, "status": f"error {e!r}"[:200], "equations": []}


def all_(raw_dir, out, processes=4):
    import libcellml
    with open(os.path.join(raw_dir, "manifest.json")) as fh:
        man = json.load(fh)
    jobs = []
    for e in man["entries"]:
        if not e.get("dir"):
            continue
        for f in e.get("cellml_files", [])[:MAX_FILES]:
            jobs.append((e["dir"], f))
    with ThreadPoolExecutor(int(processes)) as ex:
        results = list(ex.map(lambda j: _run(os.path.join(raw_dir, j[0], j[1])), jobs))
    ws = {}
    for (d, f), r in zip(jobs, results):
        r["file"] = f
        ws.setdefault(d, []).append(r)
    res = {"libcellml": libcellml.versionString(), "date": man["date"], "sha256_all": man["sha256_all"],
           "max_files_per_workspace": MAX_FILES, "timeout_s": TIMEOUT, "workspaces": ws}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=0)
    from collections import Counter
    print(Counter(r["status"] for r in results))


if __name__ == "__main__":
    if sys.argv[1] == "one":
        print(json.dumps(one(sys.argv[2])))
    else:
        all_(*sys.argv[2:])
