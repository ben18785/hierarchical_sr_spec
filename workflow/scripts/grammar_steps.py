"""Snakemake script entry points for the grammar comparison (hsr environment)."""
import json
import sys

from hsr import grammar_compare as gc


def generalised(out):
    with open(out, "w") as fh:
        json.dump(gc.generalised_map(), fh)


def folds(gen_path, out, k, seed):
    with open(gen_path) as fh:
        gen = json.load(fh)
    with open(out, "w") as fh:
        json.dump(gc.make_folds(k=int(k), seed=int(seed), gen_map=gen), fh)


def transfer(gen_path, test_corpus, out):
    with open(gen_path) as fh:
        gen = json.load(fh)
    with open(out, "w") as fh:
        json.dump(gc.make_transfer(gen, test_corpus), fh)


def loso(gen_path, test, weighting, out):
    with open(gen_path) as fh:
        gen = json.load(fh)
    with open(out, "w") as fh:
        json.dump(gc.make_loso(gen, test, weighting), fh)


def ours(folds_path, out):
    with open(folds_path) as fh:
        F = json.load(fh)
    with open(out, "w") as fh:
        json.dump(gc.grammar_logpriors(F), fh)


def table(folds_path, out_json, out_md, *score_files):
    with open(folds_path) as fh:
        F = json.load(fh)
    lp = {}
    for p in score_files:
        with open(p) as fh:
            lp.update(json.load(fh)["logprior"])
    rows = gc.heldout_table(F, lp)
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=1)
    n_test = sum(len(f["test"]) for f in F["folds"])
    with open(out_md, "w") as fh:
        if F.get("test_corpus"):
            fh.write(f"Trained on all BioModels; tested on the {n_test} forms of {F['test_corpus']}; "
                     f"log probability renormalised over a common support of {len(F['support'])} forms.\n\n")
        else:
            fh.write(f"Held-out log probability per form, renormalised over a common support of "
                     f"{len(F['support'])} forms; {F['k']}-fold CV by BioModels model (seed {F['seed']}), "
                     f"{n_test} held-out model-form pairs.\n\n")
        fh.write("| prior | mean log P | spread over folds | zero-probability forms |\n|---|---|---|---|\n")
        for r in rows:
            fm = [v for v in r["fold_means"] if v == v]
            fh.write(f"| {r['prior']} | {r['heldout_logp']:.3f} | {min(fm):.2f} to {max(fm):.2f} | {r['n_zero']} |\n")


if __name__ == "__main__":
    {"generalised": generalised, "folds": folds, "transfer": transfer, "loso": loso, "ours": ours, "table": table}[sys.argv[1]](*sys.argv[2:])
