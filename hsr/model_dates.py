"""Publication years of the corpus models, for the temporal hold-out test
(train each prior on models published before a cutoff year, score the forms of
models published from that year on; rule ``temporal_folds``).

Sources and the year used (the year of the paper describing the model where
the source records one, otherwise the year the model was first submitted):

  biomodels, biomodels_nc  BioModels model record (``/<id>?format=json``):
                           ``publication.year``; otherwise the year of the
                           first revision in ``history``; otherwise
                           ``firstPublished``. EBI blocks GitHub-hosted
                           runners, so the fetch runs locally.
  modeldb                  ModelDB record (fields of ``/api/v1/models/<id>``,
                           taken from the whole-database field listings): the
                           earliest year among ``model_paper`` entries
                           ("Authors (2008)"); otherwise a year in the model
                           name ("... (Hemond et al. 2008)"); otherwise
                           ``created``.
  jws                      JWS Online manuscript record
                           (``/rest/models/<slug>/manuscript/``): ``year``
                           (null for some models: undated).
  sabiork                  the ``publication.year`` of the model's SABIO-RK
                           publication, read from the cached export pages
                           (no new requests).
  cellml                   a year written in the workspace name or, when the
                           workspace's CellML file names give one year, that
                           year (PMR's author_year naming convention, e.g.
                           gupta_2009.cellml); otherwise undated.
  allodb                   the year in the reference id (barney_1977_bdac).
  thermal, ecocurves       the earliest year of the references of the model's
                           catalogue rows (``reference`` column, "(2004)").
  biocat                   the year of the row's reference.
  curvelib, nlmixr2lib     undated sources (library functions / a model
                           library without per-model dates).

Output ``hsr/data/model_dates.json``: {"years": {source: {model id: year}},
"undated": {source: [model ids]}, "undated_sources": [...], "rules": {source:
how}, "snapshot": ...}.
"""
from __future__ import annotations

import csv
import datetime
import gzip
import json
import os
import re
import urllib.parse

YEAR_MIN, YEAR_MAX = 1800, datetime.date.today().year
_YEAR = re.compile(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)")

BIOMODELS_REC = "https://www.ebi.ac.uk/biomodels/{}?format=json"
MODELDB_REC = "https://modeldb.science/api/v1/models/{}"
JWS_REC = "https://jjj.bio.vu.nl/rest/models/{}/manuscript/?format=json"

# source -> (record URL template, corpus file name) for the fetched sources
FETCHED = {"biomodels": (BIOMODELS_REC, "corpus.json"),
           "biomodels_nc": (BIOMODELS_REC, "corpus_biomodels_nc.json"),
           "modeldb": (MODELDB_REC, "corpus_modeldb.json"),
           "jws": (JWS_REC, "corpus_jws.json")}
UNDATED_SOURCES = ("curvelib", "nlmixr2lib")


def _ok(y):
    return y if y is not None and YEAR_MIN <= y <= YEAR_MAX else None


def years_in(text):
    """Plausible years (1800..this year) written in a string, in order."""
    return [int(m) for m in _YEAR.findall(text or "") if _ok(int(m))]


def _epoch_year(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v > 1e11:               # milliseconds
        v /= 1000
    return _ok(datetime.datetime.fromtimestamp(v, datetime.timezone.utc).year)


# ------------------------------------------------------------ record parsers
def biomodels_year(rec):
    """(year, how) from a BioModels model record."""
    pub = rec.get("publication") or {}
    try:
        y = _ok(int(pub.get("year")))
    except (TypeError, ValueError):
        y = None
    if y:
        return y, "publication"
    revs = (rec.get("history") or {}).get("revisions") or []
    ys = [_epoch_year(r.get("submitted")) for r in revs]
    ys = [y for y in ys if y]
    if ys:
        return min(ys), "first submission"
    y = _epoch_year(rec.get("firstPublished"))
    return (y, "first published") if y else (None, "undated")


def modeldb_year(rec):
    """(year, how) from a ModelDB model record."""
    papers = (rec.get("model_paper") or {}).get("value") or []
    ys = [y for p in papers for y in years_in(p.get("object_name", ""))[-1:]]
    if ys:
        return min(ys), "paper"
    ys = years_in(rec.get("name", ""))
    if ys:
        return ys[-1], "name"
    ys = years_in(rec.get("created", "")[:4])
    return (ys[0], "created") if ys else (None, "undated")


def jws_year(rec):
    """(year, how) from a JWS Online manuscript record."""
    try:
        y = _ok(int(rec.get("year")))
    except (TypeError, ValueError):
        y = None
    return (y, "manuscript") if y else (None, "undated")


PARSERS = {"biomodels": biomodels_year, "biomodels_nc": biomodels_year, "modeldb": modeldb_year, "jws": jws_year}


def record_id(source, mid):
    """The id used in the source's record URL for a corpus model id."""
    if source == "modeldb":
        return mid.split(":", 1)[1]
    return mid


# ---------------------------------------------------------- offline sources
def sabiork_years(raw_dir):
    """{SABIORK_PUB<n>: year} from the cached SABIO-RK export pages."""
    with open(os.path.join(raw_dir, "manifest.json")) as fh:
        man = json.load(fh)
    out = {}
    for f in man["files"]:
        with gzip.open(os.path.join(raw_dir, f["file"]), "rt") as fh:
            data = json.load(fh).get("data") or []
        for e in data:
            pub = e.get("publication") or {}
            pid = pub.get("publication_id") or (e.get("general") or {}).get("publication_id")
            try:
                y = _ok(int(pub.get("year")))
            except (TypeError, ValueError):
                y = None
            if pid is not None and y:
                out[f"SABIORK_PUB{pid}"] = min(y, out.get(f"SABIORK_PUB{pid}", y))
    return out


def cellml_year(ws_dir, files):
    """Year of a PMR workspace from its name, or from its CellML file names
    when they give exactly one year; None otherwise."""
    ys = years_in(ws_dir.replace("_", " "))
    if ys:
        return ys[-1]
    ys = {y for f in files for y in years_in(os.path.basename(f).replace("_", " "))}
    return ys.pop() if len(ys) == 1 else None


def cellml_years(manifest_path):
    with open(manifest_path) as fh:
        man = json.load(fh)
    files = {}
    for e in man["entries"]:
        if e.get("dir"):
            files.setdefault(e["dir"], []).extend(e.get("cellml_files") or [])
    return {f"PMR:{d}": cellml_year(d, fs) for d, fs in files.items()}


def allodb_year(mid):
    ys = years_in(mid.split(":", 1)[-1].replace("_", " "))
    return ys[0] if ys else None


def catalogue_years(csv_path, id_col):
    """{model: earliest reference year} of a pipe-separated catalogue."""
    out = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh, delimiter="|"):
            ys = years_in(r.get("reference", ""))
            if ys:
                m = r[id_col]
                out[m] = min(ys[0], out.get(m, ys[0]))
    return out


# ------------------------------------------------------------------ fetching
def _corpus_ids(path):
    with open(path) as fh:
        return sorted(json.load(fh)["per_model"])


MODELDB_FIELDS = ("id", "name", "model_paper", "created")
MODELDB_BULK = "https://modeldb.science/api/v1/models/{}"
# sites: the sources fetched one after the other with one client (<= 40 requests per minute per site)
SITES = {"ebi": ("biomodels", "biomodels_nc"), "jws": ("jws",)}
# tries per request: JWS answers HTTP 500 (persistently) for some manuscript records
SITE_TRIES = {"ebi": 6, "jws": 1}


def modeldb_bulk(d, client):
    """ModelDB answers slowly (about 40 s per request), so its records are
    assembled from four whole-database field listings (``/api/v1/models/<field>``,
    one value per model in the same order as ``/api/v1/models/id``), written as
    ``<d>/<id>.json`` records with those fields. -> manifest entries of the listings."""
    from .corpus_sources import sha256
    os.makedirs(d, exist_ok=True)
    cols, bulk = {}, []
    for f in MODELDB_FIELDS:
        path = os.path.join(d, f"bulk_{f}.json")
        url = MODELDB_BULK.format(f)
        if not os.path.exists(path):
            data, _ = client.get(url)
            json.loads(data)
            with open(path, "wb") as fh:
                fh.write(data)
        with open(path, "rb") as fh:
            data = fh.read()
        cols[f] = json.loads(data)
        bulk.append({"url": url, "file": f"modeldb/bulk_{f}.json", "sha256": sha256(data)})
    n = {f: len(v) for f, v in cols.items()}
    if len(set(n.values())) != 1:
        raise RuntimeError(f"ModelDB field listings differ in length: {n}")
    for i, mid in enumerate(cols["id"]):
        with open(os.path.join(d, f"{int(mid)}.json"), "w") as fh:
            json.dump({f: cols[f][i] for f in MODELDB_FIELDS}, fh)
    return bulk


def fetch_records(raw_dir, data_dir, limit=None):
    """Download the record of every model of the fetched sources: one request
    per model for BioModels (curated and non-curated) and JWS Online, at most
    40 per minute per site (the two sites run in parallel; both BioModels
    sources share one client), and the whole-database field listings for
    ModelDB (``modeldb_bulk``). Records are cached as
    ``<raw_dir>/<source>/<id>.json`` (an existing file is not fetched again);
    ``manifest.json`` records URLs, date and sha256."""
    import threading
    from .corpus_sources import PoliteClient, sha256

    def run_source(source, client, out):
        url_t, corpus = FETCHED[source]
        d = os.path.join(raw_dir, source)
        os.makedirs(d, exist_ok=True)
        ids = _corpus_ids(os.path.join(data_dir, corpus))
        if limit:
            ids = ids[:int(limit)]
        entries = []
        for i, mid in enumerate(ids):
            rid = record_id(source, mid)
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", rid):
                entries.append({"id": mid, "error": "unsafe id"})
                continue
            url = url_t.format(urllib.parse.quote(rid))
            path = os.path.join(d, rid + ".json")
            rec = {"id": mid, "url": url, "file": f"{source}/{rid}.json"}
            if not os.path.exists(path):
                if source == "modeldb":
                    rec["error"] = "not in the ModelDB listings"
                    entries.append(rec)
                    continue
                try:
                    data, _ = client.get(url)
                    json.loads(data)
                except Exception as e:
                    rec["error"] = repr(e)[:300]
                    entries.append(rec)
                    continue
                with open(path, "wb") as fh:
                    fh.write(data)
            with open(path, "rb") as fh:
                rec["sha256"] = sha256(fh.read())
            entries.append(rec)
            if (i + 1) % 100 == 0:
                print(f"{source} [{i + 1}/{len(ids)}]", flush=True)
        out[source] = entries

    def run_site(site, out):
        client = PoliteClient(min_interval=1.5, tries=SITE_TRIES[site], accept="application/json")
        sources = SITES[site]
        for s in sources:
            run_source(s, client, out)

    os.makedirs(raw_dir, exist_ok=True)
    out = {}
    threads = [threading.Thread(target=run_site, args=(site, out)) for site in SITES]
    for t in threads:
        t.start()
    mdb = PoliteClient(min_interval=1.5, timeout=600, accept="application/json")
    bulk = modeldb_bulk(os.path.join(raw_dir, "modeldb"), mdb)
    run_source("modeldb", None, out)
    for t in threads:
        t.join()
    sources = list(FETCHED)
    man = {"what": "model records for publication years (hsr.model_dates)", "date": datetime.date.today().isoformat(),
           "modeldb_listings": bulk, "sources": {s: out.get(s, []) for s in sources}}
    man["sha256_all"] = sha256("".join([b["sha256"] for b in bulk] + [e.get("sha256", "") for s in sources
                                                                        for e in man["sources"][s]]).encode())
    with open(os.path.join(raw_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return man


# ------------------------------------------------------------------ building
def build(records_dir, sabiork_raw, cellml_manifest, data_dir, out):
    """Assemble ``model_dates.json`` from the fetched records, the cached
    SABIO-RK pages, the PMR manifest, the allodb ids and the catalogues."""
    with open(os.path.join(records_dir, "manifest.json")) as fh:
        man = json.load(fh)
    years, how, undated = {}, {}, {}
    for source, entries in man["sources"].items():
        ys, hw = {}, {}
        for e in entries:
            y = None
            if "error" not in e:
                with open(os.path.join(records_dir, e["file"])) as fh:
                    y, h = PARSERS[source](json.load(fh))
                hw[h] = hw.get(h, 0) + 1
            if y:
                ys[e["id"]] = y
        years[source], how[source] = ys, hw
    offline = {"sabiork": (sabiork_years(sabiork_raw), "corpus_sabiork.json"),
               "cellml": (cellml_years(cellml_manifest), "corpus_cellml.json"),
               "thermal": (catalogue_years(os.path.join(data_dir, "sources", "thermal_curves.csv"), "model"),
                           "corpus_thermal.json"),
               "ecocurves": (catalogue_years(os.path.join(data_dir, "sources", "ecocurves_curves.csv"), "model"),
                             "corpus_ecocurves.json"),
               "biocat": (catalogue_years(os.path.join(data_dir, "sources", "bio_catalogue.csv"), "id"),
                          "corpus_biocat.json")}
    for source, (ys, corpus) in offline.items():
        ids = _corpus_ids(os.path.join(data_dir, corpus))
        years[source] = {m: ys[m] for m in ids if ys.get(m)}
    years["allodb"] = {m: y for m in _corpus_ids(os.path.join(data_dir, "corpus_allodb.json"))
                       if (y := allodb_year(m))}
    corpora = {s: FETCHED[s][1] for s in FETCHED} | {s: c for s, (_, c) in offline.items()}
    corpora["allodb"] = "corpus_allodb.json"
    for s, c in corpora.items():
        undated[s] = [m for m in _corpus_ids(os.path.join(data_dir, c)) if m not in years[s]]
    res = {"years": {s: dict(sorted(v.items())) for s, v in sorted(years.items())},
           "undated": {s: v for s, v in sorted(undated.items())},
           "undated_sources": list(UNDATED_SOURCES),
           "how": how,
           "coverage": {s: {"dated": len(years[s]), "undated": len(undated[s])} for s in sorted(years)},
           "snapshot": {"records_date": man["date"], "records_sha256_all": man["sha256_all"]},
           "rules": __doc__.split("Sources and the year used")[1].split("Output ")[0].strip()}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    for s, c in res["coverage"].items():
        print(f"{s}: {c['dated']} dated, {c['undated']} undated", flush=True)
    return res


if __name__ == "__main__":
    import sys
    {"fetch": fetch_records, "build": build}[sys.argv[1]](*sys.argv[2:])
