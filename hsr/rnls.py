"""Hold-out source ``rnls``: nonlinear regression formulas that biologists and
ecologists fit, mined from the source of CRAN packages.

Pipeline (rules in workflow/rules/rnls.smk):

  rnls_index     CRAN's task-view index (src/contrib/Views.rds) and package
                 database (web/packages/packages.rds), with sha256 (``index_fetch``)
  rnls_views     packages of the chosen task views (workflow/scripts/rnls_views.R)
  rnls_packages  bio/eco filter -> hsr/data/sources/rnls_packages.tsv
                 (committed: package, pinned version, licence, views; ``select_packages``)
  rnls_fetch     the source tarball of every listed version (``fetch``)
  rnls_extract   nls/nlme/gnls/nlsLM/nls2 calls and selfStart definitions,
                 found with R's parser (workflow/scripts/rnls_extract.R)
  rnls_corpus    formulas reduced to generalised one-variable keys (``build``)

A "model" is one package: it contributes each distinct key once, so ``freq``
counts packages using a form.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter

import sympy as sp

from .corpus_sources import (PoliteClient, _corpus_dict, _today, _write, r_infix, reduce_laws, sha256)

CRAN = "https://cran.r-project.org/src/contrib"

# Title/Description keywords of a biological or ecological package (case-insensitive,
# word-start match). A package of the chosen task views is kept only if its title or
# description matches one of them; general statistical infrastructure (mixed models,
# optimisers, spatial or time-series tooling) is thereby dropped.
BIO_KEYWORDS = [
    "ecolog", "species", "population", "biodiversity", "communit", "abundance", "occupancy", "habitat",
    "fish", "fisher", "forest", "tree", "plant", "crop", "soil", "agricult", "agronom", "livestock",
    "animal", "wildlife", "bird", "insect", "pest", "vegetation", "phenolog", "photosynth", "leaf",
    "biomass", "allometr", "growth curve", "growth model", "mortality", "survival analysis of animal",
    "epidemi", "disease", "infect", "outbreak", "transmission", "vaccin", "pathogen", "virus", "viral",
    "malaria", "incidence", "pharmacokinet", "pharmacodynam", "pharmacometr", "drug", "dose", "toxic",
    "biolog", "physiolog", "organism", "microb", "bacteri", "enzym", "breeding", "genotyp", "marine",
    "phytoplankton", "plankton", "zooplankton", "coral", "predator", "prey", "capture-recapture",
    "mark-recapture", "life table", "demograph", "yield", "field trial", "seed", "germination",
    "herbivor", "pollinat", "biomed", "clinical", "patient", "health",
]


def _bio_regex():
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in BIO_KEYWORDS) + ")", re.I)


# ------------------------------------------------------------------ fetching
def index_fetch(raw_dir, client=None):
    """CRAN's task-view index (src/contrib/Views.rds) and package database
    (web/packages/packages.rds: DESCRIPTION fields of every current package)."""
    client = client or PoliteClient(min_interval=1.6, accept="*/*")
    os.makedirs(raw_dir, exist_ok=True)
    files = []
    for name, url in (("Views.rds", f"{CRAN}/Views.rds"),
                      ("packages.rds", "https://cran.r-project.org/web/packages/packages.rds")):
        data, _ = client.get(url)
        with open(os.path.join(raw_dir, name), "wb") as fh:
            fh.write(data)
        files.append({"file": name, "url": url, "sha256": sha256(data), "bytes": len(data)})
    man = {"source": "CRAN task-view and package indexes", "date": _today(), "files": files}
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return man


def _check(raw_dir, man_file="manifest.json"):
    with open(os.path.join(raw_dir, man_file)) as fh:
        man = json.load(fh)
    for f in man["files"]:
        with open(os.path.join(raw_dir, f["file"]), "rb") as fh:
            if sha256(fh.read()) != f["sha256"]:
                raise RuntimeError(f"checksum mismatch: {f['file']}")
    return man


def select_packages(index_dir, views_tsv, out_tsv, exclude=""):
    """Package list of the chosen task views kept by the bio/eco filter:
    package, version (current on the index date), licence, views, the
    matched keyword. ``views_tsv`` (rnls_views.R): view, package, version,
    license, title, description (empty version: not on CRAN).
    ``exclude``: comma-separated packages dropped by hand."""
    man = _check(index_dir)
    with open(views_tsv) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    views, pk = {}, {}
    for r in rows:
        views.setdefault(r["package"], []).append(r["view"])
        pk[r["package"]] = r
    rx = _bio_regex()
    excl = {p for p in exclude.split(",") if p}
    out, stats = [], Counter()
    for p in sorted(views):
        stats["in_views"] += 1
        if not pk[p]["version"]:
            stats["not_on_cran"] += 1
            continue
        m = rx.search(pk[p]["title"] + " " + pk[p]["description"])
        if not m or p in excl:
            stats["filtered_out"] += 1
            continue
        out.append({"package": p, "version": pk[p]["version"], "license": pk[p]["license"],
                    "views": ",".join(sorted(set(views[p]))), "keyword": m.group(1).lower()})
    with open(out_tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["package", "version", "license", "views", "keyword"], delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    stats["kept"] = len(out)
    print(dict(stats), "index date", man["date"])
    return out


def fetch(raw_dir, packages_tsv, client=None):
    """Source tarball of every package version in ``packages_tsv`` (current or
    CRAN archive); files already present with the recorded checksum are kept."""
    client = client or PoliteClient(min_interval=1.6, accept="application/gzip,*/*")
    os.makedirs(raw_dir, exist_ok=True)
    with open(packages_tsv) as fh:
        pkgs = list(csv.DictReader(fh, delimiter="\t"))
    old = {}
    mp = os.path.join(raw_dir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as fh:
            old = {f["file"]: f for f in json.load(fh)["files"]}
    files, failed = [], []
    for i, r in enumerate(pkgs):
        name = f"{r['package']}_{r['version']}.tar.gz"
        path = os.path.join(raw_dir, name)
        if name in old and os.path.exists(path):
            with open(path, "rb") as fh:
                if sha256(fh.read()) == old[name]["sha256"]:
                    files.append(old[name])
                    continue
        err = None
        for url in (f"{CRAN}/{name}", f"{CRAN}/Archive/{r['package']}/{name}"):
            try:
                data, _ = client.get(url)
                break
            except RuntimeError as e:
                err = str(e)
        else:
            failed.append({"package": r["package"], "error": err})
            continue
        with open(path, "wb") as fh:
            fh.write(data)
        files.append({"file": name, "package": r["package"], "url": url, "date": _today(),
                      "sha256": sha256(data), "bytes": len(data)})
        print(f"{i + 1}/{len(pkgs)} {name}", flush=True)
    man = {"source": "CRAN source tarballs (hsr/data/sources/rnls_packages.tsv)", "date": _today(),
           "files": files, "failed": failed}
    with open(mp, "w") as fh:
        json.dump(man, fh, indent=1)
    return man


# ------------------------------------------------------------------ reduction
_RESERVED = {"pi", "T", "F", "TRUE", "FALSE", "Inf", "NA"}


def formula_law(rhs, regressors):
    """(sympy-string law, regressor symbols) of a deparsed R right-hand side,
    or raise ValueError (non-arithmetic, unknown function)."""
    expr = r_infix(rhs, {})
    syms = tuple(sorted("r_" + v.replace(".", "_dot_") for v in regressors))
    present = tuple(s for s in syms if expr.has(sp.Symbol(s, positive=True)))
    if not present:
        raise ValueError("regressor not in the transcribed formula")
    return str(expr), present


def build(raw_dir, packages_tsv, formulas_tsv, out, processes=4):
    """Corpus: one model per package; each regressor of each formula in turn
    is x1 (hsr.corpus.project via corpus_sources.law_keys), literals generalised."""
    man = _check(raw_dir)
    with open(packages_tsv) as fh:
        pkgs = {r["package"]: r for r in csv.DictReader(fh, delimiter="\t")}
    with open(formulas_tsv) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    stats = Counter()
    drops = Counter()
    laws, seen = [], set()
    for r in rows:
        stats[f"found_{r['kind']}"] += 1
        if r["status"] != "ok":
            drops[r["status"]] += 1
            continue
        regs = [v for v in r["regressors"].split(",") if v]
        try:
            law = formula_law(r["rhs"], regs)
        except ValueError as e:
            drops[f"transcription: {e}"] += 1
            continue
        stats["parsed"] += 1
        stats["multi_regressor"] += len(regs) > 1
        mid = f"cran:{r['package']}"
        if (mid, law) in seen:
            stats["duplicates_within_package"] += 1
            continue
        seen.add((mid, law))
        laws.append((mid, law))
    stats["distinct_package_formulas"] = len(laws)
    law_cache = reduce_laws([law for _, law in laws], int(processes), stats=stats)
    by_model = {}
    for mid, law in laws:
        if law_cache[law]:
            stats["formulas_reduced"] += 1
        else:
            drops["no canonical key (reduction failed or timed out)"] += 1
        by_model.setdefault(mid, set()).update(law_cache[law])
    stats["packages_listed"] = len(pkgs)
    stats["packages_downloaded"] = len(man["files"])
    stats["packages_with_calls"] = len({r["package"] for r in rows})
    stats["found"] = len(rows)
    stats["dropped"] = sum(drops.values())
    res = _corpus_dict(by_model, dict(stats))
    res["stats"]["drop_reasons"] = dict(drops.most_common())
    res["licenses"] = {f"cran:{p}": r["license"] for p, r in sorted(pkgs.items())
                       if f"cran:{p}" in res["per_model"]}
    res.update(source="Formulas of nls/nlme/gnls/nlsLM/nls2 calls and selfStart definitions in the R code, "
                      "tests, vignettes and Rd examples of CRAN packages of the Environmetrics, Agriculture, "
                      "Epidemiology and Pharmacokinetics task views (bio/eco keyword filter); one model per "
                      "package; each regressor in turn is x1; literals generalised",
               snapshot={"date": man["date"], "packages": len(man["files"]),
                         "sha256_all": sha256("".join(f["sha256"] for f in man["files"]).encode())})
    _write(res, out)
    return res


if __name__ == "__main__":
    import sys
    globals()[sys.argv[1]](*sys.argv[2:])
