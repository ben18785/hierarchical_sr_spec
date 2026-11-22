"""Build the grammar-only candidate set used by the benchmark.

    python benchmarks/build_candidates.py --enum-size 13 --samples 40000 --out hsr/data/candidates.json
"""
import argparse
import time

from hsr.candidates import build_candidates, default_n_jobs, merge_by_probes
from hsr.canonical import canonicalise
from hsr.grammar import Grammar
from hsr.simulate import BENCHMARK_FORMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum-size", type=int, default=13)
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--max-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=default_n_jobs())
    ap.add_argument("--out", default="hsr/data/candidates.json")
    ap.add_argument("--no-probe-merge", action="store_true")
    args = ap.parse_args()
    t = time.time()
    g = Grammar(max_size=args.max_size)
    c = build_candidates(g, enum_size=args.enum_size, n_samples=args.samples,
                         sample_temperature=args.temperature, max_size=args.max_size,
                         seed=args.seed, n_jobs=args.jobs, verbose=True)
    if not args.no_probe_merge:
        c = merge_by_probes(c, verbose=True)
    keys = set(c.keys)
    c.stats["benchmark_forms_in_C"] = {
        name: canonicalise(s).key in keys for name, (s, _) in BENCHMARK_FORMS.items()}
    c.stats["build_args"] = vars(args)
    print(c.stats["benchmark_forms_in_C"], f"{len(c)} candidates in {time.time() - t:.0f}s")
    c.to_json(args.out)


if __name__ == "__main__":
    main()
