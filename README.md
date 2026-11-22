# hsr: hierarchical symbolic regression (prototype)

Partial pooling of *equation structure* across many small related datasets.
Each dataset gets its own closed-form skeleton (e.g. `C*exp(C*x)`), drawn
from a shared distribution over skeletons that is itself inferred from all
datasets: a Dirichlet process whose base measure is a probabilistic grammar
with Dirichlet-distributed rule probabilities. The method is specified in
[`SPEC.md`](SPEC.md). This package implements stage A, a finite candidate set
followed by a collapsed Gibbs sampler. [`REPORT.md`](REPORT.md) records the
inference plan, the deviations from the spec and what has been learned so far.

Status: research prototype. The full benchmark has not been run yet (see
"Benchmarks" below).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test,plots]"
```

Dependencies: numpy, scipy, sympy (matplotlib is needed only for plots).

## Quick start

```python
import numpy as np
from hsr.candidates import CandidateSet
from hsr.gibbs import GibbsConfig, run_gibbs
from hsr.likelihood import compute_likelihoods, scale_population

# xs, ys: lists of 1-D arrays, one pair per dataset
cands = CandidateSet.from_json("hsr/data/candidates.json")      # 3,368 canonical skeletons
X, Y, sx, sy = scale_population(xs, ys)                         # pooled RMS scaling
table = compute_likelihoods(cands.skeletons, np.array(X), np.array(Y))
res = run_gibbs(table.laplace, cands.counts, cands.grammar, GibbsConfig())

res.assign_probs          # (J, n_candidates) posterior P(z_j = e)
res.top                   # most probable skeleton per dataset
res.pop_weight            # posterior mean population weight of each skeleton
res.n_forms_distribution()
res.theta, res.gamma      # posterior draws of rule probabilities and DP concentration
```

`compute_likelihoods` stacks datasets into a `(J, n)` array, so every dataset
must have the same `n`; pass a 0/1 mask `M` to `fit_skeleton` for unequal
sizes. `benchmarks/quickstart.py` is a runnable version of this example on
synthetic data.

## Layout

| module | contents |
|---|---|
| `hsr/grammar.py` | PCFG, Dirichlet prior, sampling with a size cap, bottom-up enumeration, `log P_theta(e)` |
| `hsr/canonical.py` | parsing, canonicalisation (simplify, constant absorption, denominator normalisation, renaming), rule counts from sympy trees |
| `hsr/candidates.py` | candidate set: enumeration + sampling, probe-based family merging, optional data-driven one-step edits |
| `hsr/likelihood.py` | scaling, compiled skeletons, batched Levenberg–Marquardt MAP fits, Laplace and BIC, importance-sampling check, posterior predictive |
| `hsr/gibbs.py` | collapsed Gibbs sampler (CRP assignments, block relabel move, Dirichlet theta update, Escobar–West gamma) |
| `hsr/baselines.py` | no pooling, no pooling with averaging over skeletons, complete pooling, oracle list |
| `hsr/simulate.py` | synthetic populations (forms A, B, C; all-A; forms drawn from the grammar) |
| `hsr/metrics.py` | recovery, ARI, population-weight error, calibration |
| `hsr/experiment.py` | one benchmark replicate → JSON lines; CLI |
| `benchmarks/` | candidate builder, job matrix, summariser (tables + plots), quick start |

## Tests

```bash
.venv/bin/pytest -q                 # 95 tests, about 20 s
.venv/bin/pytest -q -m "not slow"   # skip the validation tests
```

Unit tests cover rule counts on hand-derived trees, `P_theta` on hand
examples, canonicalisation merges and non-merges, shipped-candidate
idempotence, gradients, masking, BIC against closed form, metrics and the
baselines. The validation tests check that:

- Laplace and importance sampling agree with 2-D quadrature;
- canonicalisation never loses a function (refitting the canonical form to the original curves);
- the Gibbs marginals equal the exact posterior obtained by enumerating every assignment, with and without the block move;
- the number of distinct forms under flat likelihoods follows the CRP;
- the Escobar–West step targets the exact gamma posterior;
- the theta update counts each distinct skeleton once;
- the sampler recovers a known assignment when L is made sharp;
- the block move frees a stuck cluster that single-site Gibbs cannot.

## Benchmarks

```bash
# grammar-only candidate set (data-independent, about 1 min; shipped in hsr/data)
.venv/bin/python benchmarks/build_candidates.py

# one cell of the grid: 20 replicate populations of J = 30 datasets
.venv/bin/python -m hsr.experiment --setting mixture --n 10 --noise 0.05 --reps 0:20 \
    --out results/raw/mixture_n10_s05.jsonl          # add --sensitivity / --tau 1 / --scaling per_dataset / --expand

.venv/bin/python benchmarks/summarise.py 'results/raw/*.jsonl' --out results/summary
```

The full grid (40 jobs: 3 settings × n_j ∈ {5, 10, 20, 50} × noise ∈ {5%,
15%}, plus 16 sensitivity jobs) is defined in `benchmarks/matrix.py`. It runs
as the `benchmark` GitHub Actions workflow (Actions tab → benchmark → Run
workflow). The `benchmark-summary` artifact holds `summary.md`,
`summary.csv`, one plot per metric, and the raw JSON lines. Expect roughly
2–4 minutes per replicate on a runner.
