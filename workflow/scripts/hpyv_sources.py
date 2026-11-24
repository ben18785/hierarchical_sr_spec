"""Model-form pairs of every corpus source (hsr.grammar_compare.source_items),
for the source-labelled tree priors (rule hpyv_scores).

    python hpyv_sources.py generalised.json out.json
"""
import json
import sys

from hsr import grammar_compare as gc

if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        gen = json.load(fh)
    with open(sys.argv[2], "w") as fh:
        json.dump({n: gc.source_items(p, gen) for n, p in gc.SOURCES.items()}, fh)
