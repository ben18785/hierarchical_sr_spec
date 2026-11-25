"""Temporal hold-out folds: train every prior on the models published before a
cutoff year and score the forms of the models published from that year on.

A corpus-learned prior could simply recapitulate the forms already in the
literature. Splitting the test forms into those already used before the
cutoff ("seen") and those first used at or after it ("novel": the form's
generalised key is not in the fold's training set) asks whether the prior
anticipates forms that had not been used yet, or penalises them.

Model years come from ``hsr/data/model_dates.json`` (``hsr.model_dates``).
Policy:
  * sources without per-model dates (``undated_sources``: the R curve
    libraries and nlmixr2lib) are always in the training set;
  * models of dated sources without a year are dropped (neither trained on nor
    tested), so that no model of unknown date can leak a later form into the
    training set;
  * test = every model-form pair of a dated model with year >= cutoff (each
    model's distinct forms once, as everywhere else).

The fold has the format of ``grammar_compare.make_loso`` (one fold, the same
common support: the candidates and every form of every source), so every
scoring rule applies, plus ``train_labels`` (the source of every training
pair, for the source-aware tree priors), ``test_sources``, ``test_years`` and
``novel`` (one entry per test pair).
"""
from __future__ import annotations

import json

from . import grammar_compare as gc


def model_items(path, gen_map) -> dict:
    """{model id: sorted distinct generalised forms} of a corpus (models
    without a form left out), as ``grammar_compare.source_items`` per model."""
    with open(path) as fh:
        pm = json.load(fh)["per_model"]
    out = {}
    for mid, ks in pm.items():
        forms = sorted({gen_map.get(k, k) for k in ks} - {None})
        if forms:
            out[mid] = forms
    return out


def weight_labelled(parts: dict, weighting: str):
    """``grammar_compare.weight_sources`` with the source label of every entry."""
    parts = {n: v for n, v in parts.items() if v}
    if weighting == "equal":
        big = max(len(v) for v in parts.values())
        reps = {n: max(1, round(big / len(v))) for n, v in parts.items()}
    elif weighting == "pooled":
        reps = {n: 1 for n in parts}
    else:
        raise ValueError(weighting)
    keys, labels = [], []
    for n, v in parts.items():
        keys += v * reps[n]
        labels += [n] * (len(v) * reps[n])
    return keys, labels


def split_models(per_model: dict, years: dict | None, cutoff: int):
    """(train ids, test ids, undated ids) of one source; years None = undated source (all train)."""
    if years is None:
        return list(per_model), [], []
    train = [m for m in per_model if m in years and years[m] < cutoff]
    test = [m for m in per_model if m in years and years[m] >= cutoff]
    undated = [m for m in per_model if m not in years]
    return train, test, undated


def make_temporal(gen_map: dict, dates: dict, cutoff: int, weighting="pooled", sources=None,
                  candidates=gc.CANDIDATES) -> dict:
    sources = sources or gc.SOURCES
    cutoff = int(cutoff)
    years = dates["years"]
    undated_sources = set(dates.get("undated_sources", []))
    missing = [n for n in sources if n not in years and n not in undated_sources]
    if missing:
        raise ValueError(f"no dates for sources {missing} (and not declared undated)")
    train_parts, test, test_models, test_sources, test_years = {}, [], [], [], []
    sizes, all_forms = {}, set()
    for name, path in sources.items():
        per = model_items(path, gen_map)
        all_forms |= {k for ks in per.values() for k in ks}
        ys = None if name in undated_sources else years[name]
        tr, te, und = split_models(per, ys, cutoff)
        train_parts[name] = [k for m in tr for k in per[m]]
        for m in te:
            test += per[m]
            test_models.append(f"{name}:{m}")
            test_sources += [name] * len(per[m])
            test_years += [ys[m]] * len(per[m])
        sizes[name] = {"train_models": len(tr), "test_models": len(te), "dropped_undated": len(und),
                       "train_pairs": len(train_parts[name]), "test_pairs": sum(len(per[m]) for m in te)}
    train, labels = weight_labelled(train_parts, weighting)
    assert train == gc.weight_sources(train_parts, weighting)
    seen = set(train)
    with open(candidates) as fh:
        cand = json.load(fh)["keys"]
    support = sorted({gen_map.get(c, c) for c in cand} - {None} | all_forms)
    fold = {"train": train, "train_labels": labels, "test": test, "test_models": test_models,
            "test_sources": test_sources, "test_years": test_years, "novel": [k not in seen for k in test],
            "test_source": "temporal"}
    return {"support": support, "folds": [fold], "k": 1, "seed": 0, "generalise_literals": True,
            "test_corpus": f"temporal (models dated >= {cutoff}; {weighting})", "cutoff": cutoff,
            "weighting": weighting, "train_sizes": {n: len(v) for n, v in train_parts.items() if v},
            "sizes": sizes, "undated_sources": sorted(undated_sources)}


def year_distribution(dates: dict, sources=None, gen_map=None, bins=(1990, 2000, 2005, 2010, 2015, 2020)) -> dict:
    """{source: {bin label: number of dated models with forms}}; models without forms are not counted."""
    sources = sources or gc.SOURCES
    edges = [None, *bins, None]
    out = {}
    for name, path in sources.items():
        if name not in dates["years"]:
            continue
        per = model_items(path, gen_map or {})
        ys = [dates["years"][name][m] for m in per if m in dates["years"][name]]
        row = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            lab = f"<{hi}" if lo is None else (f">={lo}" if hi is None else f"{lo}-{hi - 1}")
            row[lab] = sum(1 for y in ys if (lo is None or y >= lo) and (hi is None or y < hi))
        row["undated"] = sum(1 for m in per if m not in dates["years"][name])
        out[name] = row
    return out
