"""Forms of the mechanism classes (rule mechanism_forms_report): per class the
number of mechanisms, drops by reason, distinct forms, and every form with its
frequency, number of constants and whether it occurs in the training sources
(the training list of a mechanism hold-out fold, i.e. every source of
hsr.grammar_compare.SOURCES).

    python mechanism_report.py out.md folds.json corpus_mech_<class>.json ...
"""
import json
import sys

MAX_ROWS = 40


def main(out_md, folds_path, *corpora):
    with open(folds_path) as fh:
        train = set(json.load(fh)["folds"][0]["train"])
    L = ["# Mechanism-derived forms\n",
         "Forms of random mechanisms (hsr.mechanisms; generators in its module docstring), one corpus per class "
         "(hsr/data/corpus_mech_<class>.json). seen: the form occurs in the training sources. Scores of every "
         "prior on these forms: results/grammar/mechanisms.md.\n",
         "| class | x1 | mechanisms | kept | dropped (by reason) | distinct forms | seen distinct | "
         "mechanisms with a seen form |",
         "|---|---|---|---|---|---|---|---|"]
    docs = []
    for p in corpora:
        with open(p) as fh:
            d = json.load(fh)
        docs.append(d)
        freq = d["freq"]
        kept = sum(freq.values())
        seen_d = sum(k in train for k in freq)
        seen_m = sum(v for k, v in freq.items() if k in train)
        drops = ", ".join(f"{k} {v}" for k, v in sorted(d["dropped"].items(), key=lambda kv: -kv[1])) or "none"
        x = d["source"].split("x1 = ")[-1].rstrip(")")
        L.append(f"| {d['class']} | {x} | {d['n_mechanisms']} | {kept} | {drops} | {len(freq)} | {seen_d} | "
                 f"{seen_m} ({seen_m / max(1, kept):.0%}) |")
    L.append("")
    for d in docs:
        freq = d["freq"]
        kept = sum(freq.values())
        nc = {r["key"]: r["n_consts"] for r in d["records"]}
        L.append(f"## {d['class']}\n")
        L.append(f"seed {d['seed']}, at most {d['max_consts']} constants"
                 + (f", at most {d['max_terms']} exponential terms" if d["class"] == "temperature" else "") + ".\n")
        L.append("| form | mechanisms | share | constants | seen |")
        L.append("|---|---|---|---|---|")
        for k, v in list(freq.items())[:MAX_ROWS]:
            L.append(f"| `{k}` | {v} | {v / kept:.1%} | {nc[k]} | {'yes' if k in train else 'no'} |")
        rest = list(freq.items())[MAX_ROWS:]
        if rest:
            L.append(f"| ({len(rest)} further forms) | {sum(v for _, v in rest)} | "
                     f"{sum(v for _, v in rest) / kept:.1%} | | {sum(k in train for k, _ in rest)} seen |")
        L.append("")
    with open(out_md, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
