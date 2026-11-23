"""Subset of the FoRAGE functional-response database for hsr.realdata.

Source: Uiterwaal, DeLong et al. (2022) FoRAGE database, v2021.2 (KNB,
doi:10.5063/DB807S; CC-BY-NC-SA; cite Ecology 103:e3706). Download the two
CSVs from KNB, then
    python benchmarks/realdata/forage_prepare.py <curves.csv> <summary.csv>

Keeps experiments where eaten prey were replaced (so the Holling forms apply
directly; without replacement the expected form is Rogers' implicit random
predator equation), with >= 8 points and >= 5 distinct prey densities.
Writes hsr/data/real/forage_replaced.csv (group,x,y: x = prey density,
y = foraging rate, original units) and forage_replaced_meta.csv.
"""
import csv
import sys
from collections import defaultdict


def main(curves, summary, out="hsr/data/real/forage_replaced", min_n=8, min_distinct=5):
    S = list(csv.reader(open(summary, encoding="latin-1")))
    h = S[0]
    col = {c: h.index(c) for c in ["Source", "Prey replaced?", "Data type", "Dim", "Predator scientific name",
                                   "Prey scientific name", "Predation or Parasitism"]}
    meta = {r[0]: {k: r[i] for k, i in col.items()} for r in S[1:]}
    pts = defaultdict(list)
    for r in list(csv.reader(open(curves, encoding="latin-1")))[1:]:
        try:
            x, y = float(r[6]), float(r[8])
        except ValueError:
            continue
        if x >= 0 and y >= 0:
            pts[r[0]].append((x, y))
    keep = [i for i, p in pts.items()
            if meta.get(i, {}).get("Prey replaced?") == "Y" and len(p) >= min_n and len({x for x, _ in p}) >= min_distinct]
    keep.sort(key=int)
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "x", "y"])
        for i in keep:
            for x, y in sorted(pts[i]):
                w.writerow([i, x, y])
    with open(out + "_meta.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "n"] + list(col))
        for i in keep:
            w.writerow([i, len(pts[i])] + [meta[i][k] for k in col])
    print(f"{len(keep)} experiments, {sum(len(pts[i]) for i in keep)} points -> {out}.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
