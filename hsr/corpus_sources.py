"""Further equation corpora (beyond BioModels and the R curve libraries).

Each source is fetched as a dated snapshot (raw files cached under
``results/corpora/raw/<source>/`` with a manifest of URLs and sha256
checksums) and parsed into ``hsr/data/corpus_<name>.json`` in the format of
``hsr/data/corpus.json``: ``per_model`` {model id: [distinct keys]},
``freq`` {key: number of models using it}, ``stats`` and ``source``. Keys are
canonical one-variable skeletons with numeric literals generalised to free
constants (``hsr.grammar_compare.generalise``), as in corpus_curvelib.json.

Sources
  * SABIO-RK (kinetic rate laws of enzyme-catalysed reactions), via the
    export API https://sabiork.h-its.org/export-api/sabio (JSON). A "model" is
    one SABIO-RK publication: it contributes each distinct skeleton of its
    rate laws once, as a BioModels model does.
  * CellML / Physiome Model Repository: see ``cellml_*``.
  * A curated catalogue of PK/PD, functional-response and growth models
    (``hsr/data/sources/bio_catalogue.csv``): see ``catalogue_build``.

The reduction of a rate law to skeletons reuses ``hsr.corpus`` (``_to_sympy``,
``project``): each variable (a concentration) in turn becomes x1 and every
other symbol a constant.
"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import os
import re
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

import sympy as sp

USER_AGENT = "hsr-corpus/0.1 (python urllib; research corpus builder)"


# ------------------------------------------------------------------ fetching
class PoliteClient:
    """GET with a generic user agent, a minimum interval between requests,
    and retries with exponential backoff (honouring Retry-After on 429/503)."""

    def __init__(self, min_interval=1.5, tries=6, timeout=300, accept="*/*"):
        self.min_interval = min_interval
        self.tries = tries
        self.timeout = timeout
        self.accept = accept
        self._last = 0.0

    def get(self, url):
        """-> (bytes, headers). Raises RuntimeError after the last retry; a 404 is not retried."""
        err = None
        for k in range(self.tries):
            wait = self._last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": self.accept,
                                                       "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        data = gzip.decompress(data)
                    return data, dict(r.headers)
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code} {e.reason}"
                if e.code == 404:
                    break
                ra = e.headers.get("Retry-After") if e.headers else None
                if e.code in (429, 503) and ra and ra.isdigit():
                    time.sleep(min(int(ra), 120))
                    continue
            except Exception as e:  # network, TLS, timeout
                err = repr(e)
            time.sleep(min(5 * 2 ** k, 120))
        raise RuntimeError(f"GET {url} failed: {err}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _today():
    return datetime.date.today().isoformat()


# ------------------------------------------------------------------ SABIO-RK
SABIORK_API = "https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json"


def sabiork_url(query, page_size=1000, page=1):
    return SABIORK_API + "?" + urllib.parse.urlencode({"q": query, "pageSize": page_size, "page": page})


def sabiork_fetch(raw_dir, chunk=1000, client=None):
    """Download every SABIO-RK kinetic-law entry as JSON, in EntryID ranges of
    ``chunk`` (one request each; the API allows 60 requests per minute and at
    most 1000 entries per page). Writes gzipped pages and ``manifest.json``."""
    client = client or PoliteClient(min_interval=1.5)
    os.makedirs(raw_dir, exist_ok=True)
    data, hdr = client.get(sabiork_url("EntryID:[1 TO *]", page_size=1))
    total = int(json.loads(data)["meta"]["total_count"])
    files, got, lo, empty = [], 0, 1, 0
    while got < total:
        hi = lo + chunk - 1
        url = sabiork_url(f"EntryID:[{lo} TO {hi}]", page_size=chunk)
        data, _ = client.get(url)
        meta = json.loads(data)["meta"]
        if meta["total_pages"] > 1:
            raise RuntimeError(f"{url}: more than one page")
        n = int(meta["total_count"])
        name = f"entries_{lo:06d}_{hi:06d}.json.gz"
        with gzip.open(os.path.join(raw_dir, name), "wb") as fh:
            fh.write(data)
        files.append({"file": name, "url": url, "entries": n, "sha256": sha256(data)})
        got += n
        empty = empty + 1 if n == 0 else 0
        if empty > 20:
            raise RuntimeError(f"stopped after 20 empty ranges at EntryID {hi}: got {got} of {total}")
        print(f"EntryID {lo}-{hi}: {n} entries ({got}/{total})", flush=True)
        lo = hi + 1
    manifest = {"source": "SABIO-RK export API", "api": SABIORK_API, "date": _today(),
                "total_count": total, "files": files,
                "sha256_all": sha256("".join(f["sha256"] for f in files).encode())}
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def sabiork_law(entry):
    """(formula, [variable names]) of one SABIO-RK JSON entry, or None when it has no
    rate equation. Variables are the parameters whose role is 'Variable'
    (concentrations)."""
    kl = entry.get("kineticlaw") or {}
    f = kl.get("formula")
    if not f or not f.strip():
        return None
    var = sorted({p["name"] for p in kl.get("parameter") or [] if p.get("role") == "Variable" and p.get("name")})
    return f.strip(), var


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _pow_of_sym(e, s):
    """exp(A*log(s) + R) -> s**A * exp(R) for A free of s (s > 0), as
    ``_pow_of_x`` in workflow/scripts/curvelib_build.py."""
    arg = sp.expand(sp.expand_log(e.args[0], force=True))
    A = arg.coeff(sp.log(s))
    if A == 0 or A.has(s):
        return e
    return s**A * sp.exp(sp.expand(arg - A*sp.log(s)))


def tidy(expr):
    """Two canonicalisation aids applied before reduction: floats that are
    integers or half-integers become exact rationals (so that S^0.5 is a square
    root and S^2.0 a square, not a free power), and exp(b*log(s)) -> s**b."""
    def half(e):
        try:
            return e.is_Float and float(2 * e).is_integer()
        except (TypeError, ValueError):
            return False
    expr = expr.replace(half, lambda e: sp.Rational(int(round(float(2 * e))), 2))
    for s in sorted(expr.free_symbols, key=str):
        expr = expr.replace(lambda e: isinstance(e, sp.exp) and e.has(sp.log(s)), lambda e: _pow_of_sym(e, s))
    return expr


def law_keys(formula, variables, timeout=10):
    """Generalised one-variable keys of a rate law (``hsr.corpus.project``, then
    ``generalise``); an empty set if it cannot be parsed or reduced in time."""
    from .corpus import _to_sympy, project
    from .grammar_compare import generalise
    parsed = _to_sympy(formula)
    if parsed is None:
        return set()
    expr, syms = parsed
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        ks = project(tidy(expr), variables, syms)
        out = set()
        for k in ks:
            try:
                g = generalise(k)
            except Exception:
                g = None
            if g:
                out.add(g)
        return out
    except _Timeout:
        return set()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _expressible(key):
    """True if our grammars can score ``key`` (context counts exist)."""
    from .candidates import skeleton_from_key
    from .grammar_compare import context_counts
    try:
        context_counts(skeleton_from_key(key).expr)
        return True
    except Exception:
        return False


def _law_keys_star(args):
    return sorted(law_keys(*args))


def reduce_laws(laws, processes=4):
    """{(formula, variables): set of keys} for distinct laws, in parallel."""
    import multiprocessing as mp
    laws = sorted(set(laws))
    if processes <= 1:
        res = [_law_keys_star(law) for law in laws]
    else:
        with mp.get_context("fork").Pool(processes) as pool:
            res = pool.map(_law_keys_star, laws, chunksize=20)
    return {law: set(ks) for law, ks in zip(laws, res)}


def sabiork_build(raw_dir, out, processes=4):
    with open(os.path.join(raw_dir, "manifest.json")) as fh:
        man = json.load(fh)
    stats = Counter()
    rows = []
    for f in man["files"]:
        path = os.path.join(raw_dir, f["file"])
        with gzip.open(path, "rb") as fh:
            data = fh.read()
        if sha256(data) != f["sha256"]:
            raise RuntimeError(f"checksum mismatch: {path}")
        for e in json.loads(data)["data"]:
            stats["entries"] += 1
            law = sabiork_law(e)
            if law is None:
                continue
            stats["entries_with_formula"] += 1
            pub = (e.get("publication") or {}).get("publication_id") or (e.get("general") or {}).get("publication_id")
            mid = f"SABIORK_PUB{pub}" if pub is not None else f"SABIORK_ENTRY{e.get('id')}"
            rows.append((mid, (law[0], tuple(law[1]))))
    law_cache = reduce_laws([ck for _, ck in rows], int(processes))
    by_pub = {}
    for mid, ck in rows:
        ks = law_cache[ck]
        stats["entries_reduced"] += bool(ks)
        by_pub.setdefault(mid, set()).update(ks)
    stats = dict(stats)
    stats["distinct_laws"] = len(law_cache)
    stats["distinct_laws_reduced"] = sum(1 for v in law_cache.values() if v)
    stats["publications_with_formula"] = len(by_pub)
    res = _corpus_dict(by_pub, stats)
    res.update(source="SABIO-RK kinetic rate laws (export API, JSON); one model per publication; "
                      "each variable (concentration) in turn is x1; literals generalised",
               snapshot={"date": man["date"], "api": man["api"], "total_count": man["total_count"],
                         "sha256_all": man["sha256_all"]})
    _write(res, out)
    return res


def _corpus_dict(by_model, stats):
    """Corpus-file fields from {model id: set of generalised keys}: forms our
    grammars cannot score are listed under ``not_expressible`` (with the number
    of models using them) and left out of ``per_model``/``freq``, as in
    corpus_curvelib.json."""
    all_freq = Counter(k for ks in by_model.values() for k in ks)
    ok = {k: _expressible(k) for k in all_freq}
    per_model = {m: sorted(k for k in ks if ok[k]) for m, ks in sorted(by_model.items())}
    per_model = {m: ks for m, ks in per_model.items() if ks}
    freq = Counter(k for ks in per_model.values() for k in ks)
    stats = dict(stats)
    stats.update(models_with_keys=sum(1 for ks in by_model.values() if ks), models=len(per_model),
                 distinct_keys=len(all_freq), distinct_keys_expressible=len(freq),
                 model_key_pairs=sum(all_freq.values()), model_key_pairs_expressible=sum(freq.values()))
    return {"stats": stats, "freq": dict(freq.most_common()), "per_model": per_model,
            "not_expressible": {k: v for k, v in all_freq.most_common() if not ok[k]}}


def _write(res, out):
    with open(out, "w") as fh:
        json.dump(res, fh, indent=0, sort_keys=True)
    print(json.dumps(res["stats"], indent=1))


# ------------------------------------------------------------------- CellML
PMR = "https://models.physiomeproject.org"
PMR_JSON = "application/vnd.physiome.pmr2.json.1"


def _pmr_collection(client, url):
    data, _ = client.get(url)
    return json.loads(data)["collection"]


def _safe_member(name):
    parts = name.split("/")[1:]       # drop the archive's top directory
    if not parts or any(p in ("", "..", ".") for p in parts):
        return None
    return parts


def cellml_fetch(raw_dir, limit=None, max_bytes=40_000_000, client=None):
    """Snapshot of the CellML files of every Physiome Model Repository exposure.

    One exposure-listing request; then per exposure its info (workspace and
    changeset) and, once per workspace, the workspace archive at that
    changeset (``<workspace>/@@archive/<commit>/tgz``). Only ``.cellml``
    members are kept, under ``<raw_dir>/<workspace slug>/``; the manifest
    records each archive's URL and sha256. Archives over ``max_bytes`` are
    skipped."""
    import io
    import tarfile
    client = client or PoliteClient(min_interval=1.5, accept=PMR_JSON)
    arch = PoliteClient(min_interval=1.5, accept="application/octet-stream")
    os.makedirs(raw_dir, exist_ok=True)
    links = [l["href"] for l in _pmr_collection(client, PMR + "/exposure")["links"] if l.get("rel") == "bookmark"]
    if limit:
        links = links[:int(limit)]
    seen, entries = set(), []
    for i, href in enumerate(links):
        rec = {"exposure": href}
        entries.append(rec)
        try:
            col = _pmr_collection(client, href)
            item = {d["name"]: d["value"] for d in col["items"][0]["data"]}
            ws = [l["href"] for l in col.get("links", []) if l.get("rel") == "via"][0]
            rec.update(title=" ".join((item.get("title") or "").split()), commit=item.get("commit_id"), workspace=ws)
        except Exception as e:
            rec["error"] = f"exposure info: {e!r}"[:300]
            continue
        if ws in seen:
            rec["duplicate_workspace"] = True
            continue
        seen.add(ws)
        rec["archive"] = f"{ws}/@@archive/{rec['commit']}/tgz"
        try:
            data, _ = arch.get(rec["archive"])
        except Exception as e:
            rec["error"] = f"archive: {e!r}"[:300]
            continue
        rec.update(sha256=sha256(data), bytes=len(data))
        if len(data) > max_bytes:
            rec["error"] = "archive too large"
            continue
        slug = ws.rstrip("/").replace(PMR + "/", "").replace("/", "__")
        rec["dir"] = slug
        files = []
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                for mem in tf.getmembers():
                    parts = _safe_member(mem.name)
                    if not (mem.isfile() and parts and mem.name.lower().endswith(".cellml")):
                        continue
                    dest = os.path.join(raw_dir, slug, *parts)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as fh:
                        fh.write(tf.extractfile(mem).read())
                    files.append("/".join(parts))
        except Exception as e:
            rec["error"] = f"untar: {e!r}"[:300]
        rec["cellml_files"] = sorted(files)
        print(f"[{i + 1}/{len(links)}] {slug}: {len(files)} cellml files", flush=True)
    manifest = {"source": "Physiome Model Repository (CellML)", "listing": PMR + "/exposure", "date": _today(),
                "n_exposures": len(links), "entries": entries,
                "sha256_all": sha256("".join(e.get("sha256", "") for e in entries).encode())}
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def cellml_law(rhs):
    """(rhs, variables) of one equation extracted by workflow/scripts/cellml_extract.py:
    identifiers x_* (states, algebraic variables, time) are variables, p_* constants."""
    import re
    return rhs, sorted(set(re.findall(r"\bx_[a-z]+\d*\b", rhs)))


def cellml_build(eq_json, out, processes=4):
    """Corpus from the extracted CellML equations; one model per workspace."""
    with open(eq_json) as fh:
        E = json.load(fh)
    stats = Counter()
    rows = []
    for ws, files in E["workspaces"].items():
        stats["workspaces"] += 1
        ok = [f for f in files if f["status"] == "ok"]
        stats["files"] += len(files)
        for f in files:
            stats["files_" + f["status"].split(" ")[0].replace(" ", "_")] += 1
        stats["workspaces_ok"] += bool(ok)
        for f in ok:
            for rhs in f["equations"]:
                law = cellml_law(rhs)
                if law[1]:
                    rows.append((f"PMR:{ws}", (law[0], tuple(law[1]))))
    stats["equations"] = len(rows)
    law_cache = reduce_laws([ck for _, ck in rows], int(processes))
    by_ws = {}
    for mid, ck in rows:
        ks = law_cache[ck]
        stats["equations_reduced"] += bool(ks)
        by_ws.setdefault(mid, set()).update(ks)
    stats = dict(stats)
    stats["distinct_equations"] = len(law_cache)
    stats["distinct_equations_reduced"] = sum(1 for v in law_cache.values() if v)
    res = _corpus_dict(by_ws, stats)
    res.update(source="Physiome Model Repository CellML models (libcellml-generated equations); one model per "
                      "workspace; each variable (state, algebraic variable or time) in turn is x1; literals generalised",
               snapshot={"date": E["date"], "listing": PMR + "/exposure", "sha256_all": E["sha256_all"],
                         "libcellml": E["libcellml"]})
    _write(res, out)
    return res


# ------------------------------------------------- curated biological catalogue
def catalogue_rows(src):
    import csv
    with open(src) as fh:
        return list(csv.DictReader(fh, delimiter="|"))


def catalogue_expr(row):
    """(sympy expression, x symbol, {param name: symbol}) of one catalogue row;
    ``defs`` ("name=expr; ...") are substituted in order."""
    x = sp.Symbol("x", positive=True)
    params = {p: sp.Symbol(p, positive=True) for p in row["params"].split(",")}
    local = {**params, "x": x, "LambertW": sp.LambertW, "pi": sp.pi}
    for d in filter(None, (s.strip() for s in (row.get("defs") or "").split(";"))):
        name, rhs = d.split("=", 1)
        local[name.strip()] = sp.sympify(rhs, locals=local)
    return sp.sympify(row["formula"], locals=local), x, params


def catalogue_key(row):
    """Generalised one-variable key of a catalogue row (``hsr.corpus.project``
    after ``tidy``), or None if it is not expressible."""
    from .corpus import project
    from .grammar_compare import generalise
    expr, x, params = catalogue_expr(row)
    try:
        ks = project(tidy(expr), ["x"], {"x": x, **params})
    except Exception:
        ks = set()
    if len(ks) != 1:
        return None
    try:
        key = generalise(next(iter(ks)))
    except Exception:
        return None
    return key if key and _expressible(key) else None


def catalogue_verify(row, evals, tol=1e-6):
    """Compare the transcribed formula with the R evaluations of its
    independent implementation: (verified, n points, max relative error)."""
    import numpy as np
    expr, x, params = catalogue_expr(row)
    syms = list(params.values())
    f = sp.lambdify([x] + syms, expr, modules=["numpy", "scipy"])
    errs = []
    for e in evals:
        yv = float(e["y"]) if e["y"] not in ("NA", "") else np.nan
        if not np.isfinite(yv):
            continue
        with np.errstate(all="ignore"):
            yc = complex(f(float(e["x"]), *[float(v) for v in e["params"].split(",")]))
        yp = yc.real if abs(yc.imag) < 1e-12 else np.nan
        if np.isfinite(yp):
            errs.append(abs(yp - yv) / max(1.0, abs(yv)))
    ok = len(errs) >= 5 and max(errs) < tol
    return bool(ok), len(errs), (float(max(errs)) if errs else None)


def catalogue_build(src, r_eval, out):
    """Corpus of the curated catalogue: one model per row. Rows whose check
    is "library" or "ode" are kept only if they agree with the R evaluation;
    rows with check "none" are kept with verified = false."""
    import csv
    rows = catalogue_rows(src)
    with open(r_eval) as fh:
        ev = list(csv.DictReader(fh))
    with open(r_eval + ".versions") as fh:
        versions = fh.read().strip()
    by_model, verification = {}, {}
    for r in rows:
        if r["check"] in ("library", "ode"):
            ok, n, err = catalogue_verify(r, [e for e in ev if e["func"] == r["id"]])
        else:
            ok, n, err = False, 0, None
        key = catalogue_key(r)
        keep = (ok or r["check"] == "none") and key is not None
        note = "" if key else "not expressible in the grammar"
        if r["check"] != "none" and not ok:
            note = (note + "; " if note else "") + "failed numerical verification"
        verification[r["id"]] = {"verified": ok, "check": r["check"], "n_points": n, "max_rel_err": err,
                                 "key": key, "in_corpus": keep, "note": note, "formula": r["formula"],
                                 "reference": r["reference"], "domain": r["domain"]}
        if keep:
            by_model[r["id"]] = {key}
    res = {"stats": {"forms": len(rows), "checked": sum(r["check"] != "none" for r in rows),
                     "verified": sum(v["verified"] for v in verification.values()),
                     "expressible": sum(v["key"] is not None for v in verification.values()),
                     "in_corpus": len(by_model)},
           "per_model": {m: sorted(ks) for m, ks in sorted(by_model.items())},
           "freq": dict(Counter(k for ks in by_model.values() for k in ks).most_common()),
           "verification": verification, "versions": versions,
           "source": "Curated catalogue of PK/PD, enzyme and microbial kinetics, functional-response, growth, "
                     "stock-recruitment and temperature models (hsr/data/sources/bio_catalogue.csv); verified "
                     "numerically against R packages or the defining ODE where possible"}
    _write(res, out)
    for k, v in verification.items():
        if not v["in_corpus"] or not v["verified"]:
            print(f"  {k}: verified={v['verified']} ({v['check']}, n={v['n_points']}, err={v['max_rel_err']}) "
                  f"in_corpus={v['in_corpus']} {v['note']}")
    return res


# ------------------------------------------- R curve catalogues (by package)
def curve_verify(row, evals, tol=1e-6, min_points=5):
    """Compare a transcribed formula with the R evaluations of the package's
    implementation (workflow/scripts/rcurves_eval.R): (verified, n points,
    max relative error). Unlike ``catalogue_verify`` the error is relative to
    |y| itself (not max(1, |y|)), so that forms with small values, e.g.
    Arrhenius factors exp(-E/kT), are compared meaningfully; points where R
    returns 0, NA or a non-finite value, or the formula is not real, are skipped."""
    import numpy as np
    expr, x, params = catalogue_expr(row)
    f = sp.lambdify([x] + list(params.values()), expr, modules=["numpy", "scipy"])
    errs = []
    for e in evals:
        yv = float(e["y"]) if e["y"] not in ("NA", "") else np.nan
        if not np.isfinite(yv) or yv == 0.0:
            continue
        with np.errstate(all="ignore"):
            try:
                yc = complex(f(float(e["x"]), *[float(v) for v in e["params"].split(",")]))
            except (ZeroDivisionError, OverflowError, ValueError):
                continue
        if not (np.isfinite(yc.real) and abs(yc.imag) <= 1e-12 * max(1.0, abs(yc.real))):
            continue
        errs.append(abs(yc.real - yv) / abs(yv))
    ok = len(errs) >= min_points and max(errs) < tol
    return bool(ok), len(errs), (float(max(errs)) if errs else None)


def curves_build(src, r_eval, out, label=""):
    """Corpus of an R curve catalogue (``hsr/data/sources/<name>_curves.csv``):
    rows are package functions (or branches of piecewise ones) transcribed as
    sympy formulas in x; a row enters the corpus only if it agrees with the R
    evaluation of the package's implementation (rows without ``r_call`` are
    kept with verified = false). Rows are grouped into models by the
    ``model`` column (the original publication, so that a model implemented in
    two packages counts once); each model contributes its distinct keys once."""
    import csv
    rows = catalogue_rows(src)
    with open(r_eval) as fh:
        ev = list(csv.DictReader(fh))
    with open(r_eval + ".versions") as fh:
        versions = fh.read().strip()
    by_func = {}
    for e in ev:
        by_func.setdefault(e["func"], []).append(e)
    by_model, verification = {}, {}
    for r in rows:
        if r["r_call"].strip():
            ok, n, err = curve_verify(r, by_func.get(r["id"], []))
        else:
            ok, n, err = False, 0, None
        key = catalogue_key(r)
        keep = (ok or not r["r_call"].strip()) and key is not None
        note = "" if key else "not expressible in the grammar"
        if r["r_call"].strip() and not ok:
            note = (note + "; " if note else "") + "failed numerical verification"
        verification[r["id"]] = {"verified": ok, "n_points": n, "max_rel_err": err, "key": key,
                                 "in_corpus": keep, "note": note, "formula": r["formula"],
                                 "package": r["package"], "function": r["function"], "model": r["model"]}
        if keep:
            by_model.setdefault(r["model"], set()).add(key)
    keys = Counter(k for ks in by_model.values() for k in ks)
    res = {"stats": {"rows": len(rows), "verified": sum(v["verified"] for v in verification.values()),
                     "expressible": sum(v["key"] is not None for v in verification.values()),
                     "rows_in_corpus": sum(v["in_corpus"] for v in verification.values()),
                     "models": len(by_model), "distinct_keys": len(keys),
                     "packages": sorted({r["package"] for r in rows})},
           "per_model": {m: sorted(ks) for m, ks in sorted(by_model.items())},
           "freq": dict(keys.most_common()), "verification": verification, "versions": versions,
           "source": label or f"R curve catalogue {os.path.basename(src)}; formulas transcribed and verified "
                              "numerically against the packages; one model per original publication"}
    _write(res, out)
    for k, v in verification.items():
        if not v["in_corpus"] or not v["verified"]:
            print(f"  {k}: verified={v['verified']} (n={v['n_points']}, err={v['max_rel_err']}) "
                  f"in_corpus={v['in_corpus']} {v['note']}")
    return res


# ------------------------------------------------------------------ ModelDB
MODELDB = "https://modeldb.science"


def modeldb_fetch(raw_dir, application="NEURON", limit=None, max_bytes=300_000_000, client=None):
    """Snapshot of the NMODL (.mod) files of every ModelDB model of a modelling
    application (default NEURON): one listing request
    (``/api/v1/models?modeling_application=...``), then one zip download per
    model (``/download/<id>``), at most 40 requests per minute. Only ``.mod``
    members are kept, under ``<raw_dir>/<id>/``; the manifest records each
    zip's URL, size and sha256."""
    import io
    import zipfile
    client = client or PoliteClient(min_interval=1.5, accept="application/json")
    arch = PoliteClient(min_interval=1.5, accept="application/zip,*/*")
    os.makedirs(raw_dir, exist_ok=True)
    listing = f"{MODELDB}/api/v1/models?" + urllib.parse.urlencode({"modeling_application": application})
    ids = sorted(int(i) for i in json.loads(client.get(listing)[0]))
    if limit:
        ids = ids[:int(limit)]
    entries = []
    for i, mid in enumerate(ids):
        rec = {"id": mid, "url": f"{MODELDB}/download/{mid}"}
        entries.append(rec)
        try:
            data, _ = arch.get(rec["url"])
        except Exception as e:
            rec["error"] = f"download: {e!r}"[:300]
            continue
        rec.update(sha256=sha256(data), bytes=len(data))
        if len(data) > max_bytes:
            rec["error"] = "archive too large"
            continue
        files = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for mem in zf.infolist():
                    parts = [p for p in mem.filename.replace("\\", "/").split("/") if p]
                    if (mem.is_dir() or not parts or any(p in ("..", ".") for p in parts)
                            or parts[0] == "__MACOSX" or not parts[-1].lower().endswith(".mod")):
                        continue
                    dest = os.path.join(raw_dir, str(mid), *parts)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as fh:
                        fh.write(zf.read(mem))
                    files.append("/".join(parts))
        except Exception as e:
            rec["error"] = f"unzip: {e!r}"[:300]
        rec["mod_files"] = sorted(files)
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(ids)}] {mid}: {len(files)} mod files", flush=True)
    manifest = {"source": "ModelDB (NMODL files)", "listing": listing, "date": _today(), "n_models": len(ids),
                "entries": entries, "sha256_all": sha256("".join(e.get("sha256", "") for e in entries).encode())}
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def _mod_file_equations(path, timeout=30):
    """{"status", "equations": [[rhs, [variables]]], "errors": n} of one NMODL file (hsr.nmodl)."""
    from .nmodl import Module
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8", errors="replace")
        eqs, errors = Module(text).equations()
        return {"status": "ok", "equations": sorted({(e, tuple(v)) for e, v in eqs}), "errors": len(errors)}
    except _Timeout:
        return {"status": "timeout", "equations": [], "errors": 0}
    except Exception as e:
        return {"status": f"failed: {type(e).__name__}", "equations": [], "errors": 0}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def modeldb_equations(raw_dir, out, processes=4):
    """Equations of every NMODL file of the ModelDB snapshot (``hsr.nmodl``);
    files with identical content are parsed once."""
    import multiprocessing as mp
    with open(os.path.join(raw_dir, "manifest.json")) as fh:
        man = json.load(fh)
    jobs, by_hash = [], {}
    for e in man["entries"]:
        for f in e.get("mod_files", []):
            path = os.path.join(raw_dir, str(e["id"]), *f.split("/"))
            with open(path, "rb") as fh:
                h = sha256(fh.read())
            jobs.append((e["id"], f, h))
            by_hash.setdefault(h, path)
    hashes = sorted(by_hash)
    with mp.get_context("fork").Pool(int(processes)) as pool:
        res = dict(zip(hashes, pool.map(_mod_file_equations, [by_hash[h] for h in hashes], chunksize=8)))
    models = {}
    for mid, f, h in jobs:
        models.setdefault(str(mid), []).append({"file": f, "sha256": h, **res[h]})
    E = {"date": man["date"], "listing": man["listing"], "sha256_all": man["sha256_all"],
         "n_files": len(jobs), "n_distinct_files": len(hashes), "models": models}
    with open(out, "w") as fh:
        json.dump(E, fh)
    st = Counter(r["status"] for r in res.values())
    print(f"{len(jobs)} files ({len(hashes)} distinct): {dict(st)}")
    return E


def modeldb_build(eq_json, out, processes=4):
    """Corpus from the ModelDB NMODL equations; one model per ModelDB entry;
    each variable of an equation (v, t, a STATE, a concentration read, an
    argument of the enclosing FUNCTION/PROCEDURE) in turn is x1."""
    with open(eq_json) as fh:
        E = json.load(fh)
    stats = Counter()
    rows = []
    for mid, files in E["models"].items():
        stats["models"] += 1
        stats["models_with_mod_files"] += bool(files)
        for f in files:
            stats["files"] += 1
            stats["files_" + f["status"].split(":")[0]] += 1
            stats["assignments_not_parsed"] += f["errors"]
            for rhs, var in f["equations"]:
                rows.append((f"ModelDB:{mid}", (rhs, tuple(var))))
    stats["equations"] = len(rows)
    law_cache = reduce_laws([ck for _, ck in rows], int(processes))
    by_model = {}
    for mid, ck in rows:
        ks = law_cache[ck]
        stats["equations_reduced"] += bool(ks)
        by_model.setdefault(mid, set()).update(ks)
    stats = dict(stats)
    stats["distinct_equations"] = len(law_cache)
    stats["distinct_equations_reduced"] = sum(1 for v in law_cache.values() if v)
    res = _corpus_dict(by_model, stats)
    res.update(source="ModelDB NEURON models: equations of the NMODL files (hsr.nmodl; locals substituted, the "
                      "file's FUNCTIONs inlined); one model per ModelDB entry; each variable in turn is x1; "
                      "literals generalised",
               snapshot={"date": E["date"], "listing": E["listing"], "sha256_all": E["sha256_all"]})
    _write(res, out)
    return res


# --------------------------------------------------------------- nlmixr2lib
CRAN = "https://cran.r-project.org/src/contrib"
_R_TOKEN = __import__("re").compile(r"(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|(?P<id>[A-Za-z.][A-Za-z0-9._]*)")
_R_FUNCS = {"exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh,
            "abs": sp.Abs, "expit": lambda a: 1 / (1 + sp.exp(-a)), "logit": lambda a: sp.log(a / (1 - a)),
            "log10": lambda a: sp.log(a) / sp.log(10)}


def cran_fetch(raw_dir, package, version, client=None):
    """Download one CRAN source tarball (current or archived) into ``raw_dir``
    with a manifest of its URL and sha256."""
    client = client or PoliteClient(min_interval=1.5, accept="application/gzip,*/*")
    os.makedirs(raw_dir, exist_ok=True)
    name = f"{package}_{version}.tar.gz"
    err = None
    for url in (f"{CRAN}/{name}", f"{CRAN}/Archive/{package}/{name}"):
        try:
            data, _ = client.get(url)
            break
        except RuntimeError as e:
            err = e
    else:
        raise RuntimeError(f"{name}: {err}")
    with open(os.path.join(raw_dir, name), "wb") as fh:
        fh.write(data)
    man = {"source": f"CRAN {package} {version}", "url": url, "date": _today(), "file": name,
           "sha256": sha256(data), "bytes": len(data)}
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return man


def r_infix(expr, env):
    """sympy expression of a deparsed R expression; identifiers become
    symbols ``r_<name>`` unless ``env`` gives their value. Raises ValueError
    for anything outside arithmetic and _R_FUNCS."""
    import re
    if re.search(r"[<>!&|%\[$]|==|\bif\b|\bifelse\b", expr):
        raise ValueError("not arithmetic")
    ns, out = {}, []
    for m in _R_TOKEN.finditer(expr):
        name = m.group("id")
        if name is None or name in _R_FUNCS:
            continue
        safe = "r_" + name.replace(".", "_dot_")
        ns[safe] = env.get(name, sp.pi if name == "pi" else sp.Symbol(safe, positive=True))
    ns.update(_R_FUNCS)
    txt = _R_TOKEN.sub(lambda m: m.group(0) if m.group("num") or m.group("id") in _R_FUNCS
                       else "r_" + m.group("id").replace(".", "_dot_"), expr).replace("^", "**")
    try:
        val = sp.sympify(txt, locals=ns)
    except Exception as e:
        raise ValueError(f"parse: {type(e).__name__}") from e
    if not isinstance(val, sp.Expr) or val.has(sp.Function("linCmt")) or any(
            isinstance(f, sp.core.function.AppliedUndef) for f in val.atoms(sp.Function)):
        raise ValueError("unknown function")
    return val


def nlmixr2lib_model_equations(rows):
    """[(equation string, [variable names])], n_failed of one nlmixr2lib model
    (its rows of nlmixr2lib_extract.R): assignments are substituted into later
    statements; variables are the states (``d/dt``), time (t, time) and the
    covariates (identifiers neither assigned in model() nor defined in ini());
    parameters, random effects and derived quantities are constants."""
    ini = {r["lhs"] for r in rows if r["kind"] == "ini"}
    stmts = [r for r in rows if r["kind"] in ("assign", "ddt")]
    assigned = {r["lhs"] for r in stmts}
    states = {r["lhs"] for r in stmts if r["kind"] == "ddt"}
    import re
    idents = {m.group("id") for r in stmts for m in _R_TOKEN.finditer(r["rhs"]) if m.group("id")}
    covariates = idents - assigned - ini - set(_R_FUNCS)
    variables = states | {"t", "time"} | covariates
    env, eqs, failed = {}, [], 0
    for r in stmts:
        try:
            val = r_infix(r["rhs"], env)
        except ValueError:
            failed += 1
            env.pop(r["lhs"], None)
            continue
        if r["kind"] == "assign" and r["lhs"] not in variables:
            env[r["lhs"]] = val
        present = sorted(n for n in variables if val.has(sp.Symbol("r_" + n.replace(".", "_dot_"), positive=True)))
        if present and sp.count_ops(val) <= 400:
            eqs.append((str(val), ["r_" + n.replace(".", "_dot_") for n in present]))
    return sorted(set((e, tuple(v)) for e, v in eqs)), failed


def nlmixr2lib_build(raw_dir, tsv, out, processes=4):
    """Corpus of the nlmixr2lib model library: one model per model file."""
    import csv
    with open(os.path.join(raw_dir, "manifest.json")) as fh:
        man = json.load(fh)
    with open(os.path.join(raw_dir, man["file"]), "rb") as fh:
        if sha256(fh.read()) != man["sha256"]:
            raise RuntimeError(f"checksum mismatch: {man['file']}")
    with open(tsv) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append({k: (v or "") for k, v in r.items()})
    stats = Counter(model_files=len(by_file))
    laws = []
    for f, rs in sorted(by_file.items()):
        if any(r["kind"] == "error" for r in rs):
            stats["files_unparsed"] += 1
            continue
        eqs, failed = nlmixr2lib_model_equations(rs)
        stats["statements"] += sum(r["kind"] in ("assign", "ddt") for r in rs)
        stats["statements_not_arithmetic"] += failed
        laws += [(f"nlmixr2lib:{f[:-2]}", law) for law in eqs]
    stats["equations"] = len(laws)
    law_cache = reduce_laws([law for _, law in laws], int(processes))
    by_model = {}
    for mid, law in laws:
        stats["equations_reduced"] += bool(law_cache[law])
        by_model.setdefault(mid, set()).update(law_cache[law])
    res = _corpus_dict(by_model, dict(stats))
    res.update(source="nlmixr2lib model library (pharmacometric PK/PD, disease and endogenous models; statements "
                      "of model() blocks, assignments substituted); one model per file; each state, time or "
                      "covariate in turn is x1; literals generalised",
               snapshot={"date": man["date"], "url": man["url"], "sha256": man["sha256"]})
    _write(res, out)
    return res


# -------------------------------------------------------------------- allodb
def github_fetch(raw_dir, repo, commit, client=None):
    """Download a public GitHub repository as a tarball at a fixed commit
    (``codeload.github.com``, no authentication) with a manifest of its URL and sha256."""
    client = client or PoliteClient(min_interval=1.5, accept="application/gzip,*/*")
    os.makedirs(raw_dir, exist_ok=True)
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    data, _ = client.get(url)
    name = f"{repo.replace('/', '_')}_{commit[:12]}.tar.gz"
    with open(os.path.join(raw_dir, name), "wb") as fh:
        fh.write(data)
    man = {"source": f"GitHub {repo}", "commit": commit, "url": url, "date": _today(), "file": name,
           "sha256": sha256(data), "bytes": len(data)}
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return man


def allodb_check(expr, grid, y, tol=1e-9):
    """Does the sympy transcription of an allodb equation agree with R's
    evaluation at the DBH grid? (relative error below ``tol`` at every point
    where R's value is finite and non-zero, and at least 3 such points)"""
    import math
    f = sp.lambdify([sp.Symbol("r_dbh", positive=True)], expr, modules="math")
    n = 0
    for d, yv in zip(grid, y):
        if not math.isfinite(yv) or yv == 0:
            continue
        try:
            yp = float(f(d))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return False
        if not abs(yp - yv) <= tol * abs(yv):
            return False
        n += 1
    return n >= 3


def allodb_build(raw_dir, tsv, out, processes=4):
    """Corpus of the allodb allometric equations with DBH as the only
    independent variable; one model per reference (publication), each
    contributing its distinct forms once. Every equation is checked against
    R's evaluation (rule allodb_equations) before it is used."""
    import csv
    import math
    with open(os.path.join(raw_dir, "manifest.json")) as fh:
        man = json.load(fh)
    with open(os.path.join(raw_dir, man["file"]), "rb") as fh:
        if sha256(fh.read()) != man["sha256"]:
            raise RuntimeError(f"checksum mismatch: {man['file']}")
    with open(tsv) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    stats = Counter(equations=len(rows))
    laws, failures = [], {}
    for r in rows:
        if r["independent_variable"] != "DBH":
            stats["not_univariate"] += 1
            continue
        try:
            expr = r_infix(r["equation_allometry"], {})
        except ValueError as e:
            failures[r["equation_id"]] = str(e)
            continue
        grid = [float(v) for v in r["dbh_grid"].split(";")]
        y = [float(v) if v.strip() not in ("NA", "") else math.nan for v in r["y"].split(";")]
        if not allodb_check(expr, grid, y):
            failures[r["equation_id"]] = "differs from R's evaluation"
            continue
        stats["verified"] += 1
        laws.append((f"allodb:{r['ref_id']}", (str(expr), ("r_dbh",))))
    law_cache = reduce_laws([law for _, law in laws], int(processes))
    by_model = {}
    for mid, law in laws:
        stats["equations_reduced"] += bool(law_cache[law])
        by_model.setdefault(mid, set()).update(law_cache[law])
    stats["failed"] = len(failures)
    res = _corpus_dict(by_model, dict(stats))
    res.update(source="allodb allometric equations for tree biomass and height as functions of DBH (equation "
                      "strings checked against R's evaluation); one model per reference; literals generalised",
               failures=failures, snapshot={k: man[k] for k in ("date", "url", "commit", "sha256")})
    _write(res, out)
    return res


# ----------------------------------------------------- overlap between sources
def source_forms(path, gen_map):
    """{model id: set of generalised forms} of a corpus file, as
    ``hsr.grammar_compare.source_items`` reads it (keys mapped through
    ``gen_map`` where known; unparseable ones dropped)."""
    with open(path) as fh:
        pm = json.load(fh)["per_model"]
    return {m: {gen_map.get(k, k) for k in ks} - {None} for m, ks in pm.items()}


def overlap_table(sources, gen_map, baseline):
    """Markdown summary of corpora ``sources`` {name: path}: models, model-form
    pairs, distinct forms, forms not in any other source, forms not in the
    union of the ``baseline`` sources, and the pairwise number of shared forms."""
    forms = {n: source_forms(p, gen_map) for n, p in sources.items()}
    distinct = {n: set().union(*f.values()) if f else set() for n, f in forms.items()}
    base = set().union(*(distinct[n] for n in baseline if n in distinct))
    names = list(sources)
    lines = ["| source | models | model-form pairs | distinct forms | not in any other source | "
             "not in baseline union |", "|---|---|---|---|---|---|"]
    for n in names:
        others = set().union(*(distinct[m] for m in names if m != n))
        lines.append(f"| {n} | {sum(1 for f in forms[n].values() if f)} | {sum(len(f) for f in forms[n].values())} | "
                     f"{len(distinct[n])} | {len(distinct[n] - others)} | "
                     f"{'-' if n in baseline else len(distinct[n] - base)} |")
    allf = set().union(*distinct.values())
    new = allf - base
    lines += ["", f"Baseline ({', '.join(baseline)}): {len(base)} distinct forms. All sources: {len(allf)} "
                  f"distinct forms ({len(new)} not in the baseline).", "",
              "Shared distinct forms (row and column source):", "",
              "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for a in names:
        lines.append(f"| {a} | " + " | ".join(str(len(distinct[a] & distinct[b])) for b in names) + " |")
    return "\n".join(lines) + "\n"


def overlap_report(gen_path, out, baseline, *named):
    """CLI: overlap_report <generalised.json> <out.md> <baseline names, comma-separated> name=path ..."""
    with open(gen_path) as fh:
        gen_map = json.load(fh)
    sources = dict(s.split("=", 1) for s in named)
    md = overlap_table(sources, gen_map, baseline.split(","))
    with open(out, "w") as fh:
        fh.write(md)
    print(md)


# ------------------------------------------------------- BioModels (by query)
BIOMODELS = "https://www.biomodels.org"
BIOMODELS_NONCURATED = 'curationstatus:"Non-curated" AND modelformat:"SBML"'


def biomodels_search_url(query, offset=0, n=100):
    return BIOMODELS + "/search?" + urllib.parse.urlencode(
        {"query": query, "numResults": n, "offset": offset, "sort": "id-asc", "format": "json"})


def biomodels_main_sbml(info):
    """(file name, server sha256) of a model's main SBML file from its
    ``/<id>?format=json`` record, or None if it has none."""
    for f in (info.get("files") or {}).get("main") or []:
        if f.get("name", "").lower().endswith((".xml", ".sbml")):
            return f["name"], f.get("sha256sum")
    return None


def biomodels_fetch(raw_dir, query=BIOMODELS_NONCURATED, limit=None, client=None):
    """Snapshot of the main SBML file of every BioModels entry matching
    ``query`` (default: all non-curated SBML models). One search request per
    100 models, then per model its record (file names and server checksums)
    and its main file, at most 40 requests per minute. Files are saved as
    ``<raw_dir>/<id>.xml``; a file already present whose sha256 matches the
    server's is not downloaded again. Downloads whose checksum differs from
    the server's are recorded as errors and not kept."""
    client = client or PoliteClient(min_interval=1.5, accept="application/json")
    dl = PoliteClient(min_interval=1.5, accept="application/xml,*/*")
    os.makedirs(raw_dir, exist_ok=True)
    ids, offset, total = [], 0, None
    while total is None or offset < total:
        data, _ = client.get(biomodels_search_url(query, offset))
        d = json.loads(data)
        total = int(d["matches"])
        page = [m["id"] for m in d["models"]]
        if not page:
            break
        ids += page
        offset += len(page)
    ids = sorted(set(ids))
    if limit:
        ids = ids[:int(limit)]
    entries = []
    for i, mid in enumerate(ids):
        rec = {"id": mid}
        entries.append(rec)
        try:
            info = json.loads(client.get(f"{BIOMODELS}/{mid}?format=json")[0])
        except Exception as e:
            rec["error"] = f"record: {e!r}"[:300]
            continue
        rec.update(name=" ".join((info.get("name") or "").split()), curation=info.get("curationStatus"))
        main = biomodels_main_sbml(info)
        if main is None:
            rec["error"] = "no main SBML file"
            continue
        fname, server_sha = main
        rec.update(filename=fname, sha256_server=server_sha,
                   url=f"{BIOMODELS}/model/download/{mid}?" + urllib.parse.urlencode({"filename": fname}))
        path = os.path.join(raw_dir, mid + ".xml")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data = fh.read()
            if not server_sha or sha256(data) == server_sha:
                rec.update(file=mid + ".xml", sha256=sha256(data), bytes=len(data), cached=True)
                continue
        try:
            data, _ = dl.get(rec["url"])
        except Exception as e:
            rec["error"] = f"download: {e!r}"[:300]
            continue
        if server_sha and sha256(data) != server_sha:
            rec["error"] = "checksum differs from the server's"
            continue
        with open(path, "wb") as fh:
            fh.write(data)
        rec.update(file=mid + ".xml", sha256=sha256(data), bytes=len(data))
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(ids)}] {mid}", flush=True)
    manifest = {"source": "BioModels", "query": query, "search": biomodels_search_url(query), "date": _today(),
                "matches": total, "entries": entries,
                "sha256_all": sha256("".join(e.get("sha256", "") for e in entries).encode())}
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


JWS = "https://jjj.bio.vu.nl"


def jws_fetch(raw_dir, limit=None, client=None):
    """Snapshot of the SBML export of every JWS Online model: one listing
    request (``/rest/models/``), then ``/models/<slug>/sbml/`` per model, at
    most 40 requests per minute. Files are saved as ``<raw_dir>/<slug>.xml``;
    the manifest (same layout as ``biomodels_fetch``) records URLs and sha256."""
    client = client or PoliteClient(min_interval=1.5, accept="application/json")
    dl = PoliteClient(min_interval=1.5, accept="application/xml,text/plain,*/*")
    os.makedirs(raw_dir, exist_ok=True)
    listing = f"{JWS}/rest/models/?format=json"
    models = json.loads(client.get(listing)[0])
    slugs = sorted(m["slug"] for m in models if not m.get("cbm"))
    if limit:
        slugs = slugs[:int(limit)]
    entries = []
    for i, slug in enumerate(slugs):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", slug):
            entries.append({"id": slug, "error": "unsafe slug"})
            continue
        rec = {"id": slug, "url": f"{JWS}/models/{urllib.parse.quote(slug)}/sbml/"}
        entries.append(rec)
        try:
            data, _ = dl.get(rec["url"])
        except Exception as e:
            rec["error"] = f"download: {e!r}"[:300]
            continue
        if b"<sbml" not in data[:2000]:
            rec["error"] = "not SBML"
            continue
        with open(os.path.join(raw_dir, slug + ".xml"), "wb") as fh:
            fh.write(data)
        rec.update(file=slug + ".xml", sha256=sha256(data), bytes=len(data))
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(slugs)}] {slug}", flush=True)
    manifest = {"source": "JWS Online", "listing": listing, "date": _today(), "matches": len(slugs),
                "entries": entries, "sha256_all": sha256("".join(e.get("sha256", "") for e in entries).encode())}
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def _model_keys_star(path):
    from .corpus import model_keys
    try:
        keys, n_laws, n_ok = model_keys(path)
        return sorted(keys), n_laws, n_ok, None
    except Exception as e:
        return [], 0, 0, repr(e)[:200]


def biomodels_build(raw_dir, out, processes=4, label="non-curated"):
    """Corpus of a BioModels snapshot (``biomodels_fetch``), reduced as
    ``hsr.corpus`` reduces the curated models (each species of each kinetic
    law in turn is x1), with keys generalised; one model per BioModels entry."""
    import multiprocessing as mp
    with open(os.path.join(raw_dir, "manifest.json")) as fh:
        man = json.load(fh)
    stats = Counter(entries=len(man["entries"]))
    todo = []
    for e in man["entries"]:
        if "file" not in e:
            stats["not_downloaded"] += 1
            continue
        path = os.path.join(raw_dir, e["file"])
        with open(path, "rb") as fh:
            if sha256(fh.read()) != e["sha256"]:
                raise RuntimeError(f"checksum mismatch: {path}")
        todo.append((e["id"], path))
    if int(processes) <= 1:
        res = [_model_keys_star(p) for _, p in todo]
    else:
        with mp.get_context("fork").Pool(int(processes)) as pool:
            res = pool.map(_model_keys_star, [p for _, p in todo], chunksize=4)
    from .grammar_compare import generalise
    gen_cache = {}

    def gen(k):
        if k not in gen_cache:
            try:
                gen_cache[k] = generalise(k)
            except Exception:
                gen_cache[k] = None
        return gen_cache[k]

    by_model = {}
    for (mid, _), (keys, n_laws, n_ok, err) in zip(todo, res):
        if err:
            stats["unreadable"] += 1
            continue
        stats["models_read"] += 1
        stats["laws"] += n_laws
        stats["laws_reduced"] += n_ok
        by_model[mid] = {g for g in map(gen, keys) if g}
    res = _corpus_dict(by_model, dict(stats))
    if man.get("source") == "BioModels":
        source = f"BioModels {label} SBML models"
    else:
        source = f"{man.get('source')} SBML models"
    res.update(source=source + " (kinetic laws, as hsr.corpus); one model per entry; each species in turn is x1; "
                               "literals generalised",
               snapshot={k: man[k] for k in ("date", "query", "listing", "matches", "sha256_all") if k in man})
    _write(res, out)
    return res

