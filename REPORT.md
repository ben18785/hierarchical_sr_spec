# Hierarchical symbolic regression, stage A: report

**Status (22 Nov 2026).** The package, the tests (95, all passing) and the
benchmark harness are done. **The benchmark grid in spec section 6 has not
been run yet**: it is set up to run on GitHub Actions, and this repository has
no remote yet. Sections 1–3 describe what was built and how it was checked.
Section 4 lists what development runs have already shown; these come from
single replicates, not from the benchmark. Section 5 says what the benchmark
will report.

## 1. Inference plan as implemented

**Model.** As in the spec: `theta_A ~ Dir(alpha_A)`, `G ~ DP(gamma, P_theta)`,
`e_j ~ G`, constants `phi_jk ~ N(0, 3²)`, `sigma_j ~ HalfNormal(1)`,
`y_ji ~ N(f_e(x_ji; phi_j), sigma_j²)` and `gamma ~ Gamma(1, 1)`.

**Step 1: candidates** (`hsr/candidates.py`, data-independent, built once):

1. Enumerate every derivation up to 13 rule applications (33,004 strings
   after pruning variable-free composites such as `C*C` and keeping ≤ 4
   constants).
2. Add 40,000 draws from a tempered PCFG (prior mean raised to 1/2),
   capped at 16 rules. This adds 1,233 larger derivations.
3. Canonicalise (section 2), which leaves 3,492 distinct skeletons.
4. Probe-merge: fit every skeleton to 6 generic noisy curves and merge
   skeletons that have the same number of constants and identical
   maximised log-likelihoods (4 decimals) on every probe. This merges 101
   groups and leaves 3,368 candidates. Every sampled group checked by hand
   was a genuine reparameterisation (sign flips through odd functions,
   `exp(c*log x) = x**c`, and so on). A failed optimisation can only split a
   group, never merge two.

All three benchmark forms are in the final set.

**Step 2: marginal likelihoods** (`hsr/likelihood.py`):

1. Each skeleton is compiled once with sympy: f, its gradient and its
   Hessian in the constants.
2. All J datasets × 10 starts are fitted as one batched Levenberg–Marquardt
   problem. Steps in phi alternate with the closed-form optimum of
   s = log sigma. With the HalfNormal prior and the log-Jacobian, that
   optimum solves u² + (n−1)u − S = 0 for u = sigma², where S is the
   residual sum of squares.
3. Laplace uses the exact Hessian of the negative log joint in (phi, s),
   including the second derivatives of f. A non-PD Hessian or a non-finite
   fit gives −inf.
4. BIC comes from a prior-free refit started at the MAP.
5. Cost: about 50 s per replicate for 3,368 skeletons × 30 datasets on
   one core.

**Step 3: Gibbs sampler** (`hsr/gibbs.py`). Each sweep runs:

1. Single-site CRP updates of z_j, in random order and in log space.
2. **A block relabel move (an addition):** each cluster resamples its
   skeleton while its members stay fixed. The exact conditional is
   `P(e) ∝ P_theta(e) · Π_{k=1}^{m−1}(k + gamma·P_theta(e)) · Π_{j∈cluster} exp L[j,e]`,
   over the skeletons that no other cluster uses.
3. `theta_A ~ Dir(alpha_A + counts of distinct skeletons in use)`.
4. The Escobar–West update of gamma.

The default run is 4 chains × 2,000 sweeps with 500 burn-in. Chain 0
starts at the argmax; the other chains start from random draws from the
no-pooling posterior. The spec has every chain start at the argmax, which
would make the chains share their starting point.

**Outputs.** Per-dataset posteriors over skeletons; the posterior of the
number of distinct forms; posterior mean population weights
`E[G({e})] = E[(n_e + gamma·P_theta(e)) / (gamma + J)]`; draws of theta and
gamma; and chain-agreement diagnostics (R-hat on K, and the largest total
variation distance between chains' per-dataset posteriors).

## 2. Deviations from the spec, and why

| spec | implemented | reason |
|---|---|---|
| Size cap 12 rule applications | Enumerate to 13, sample to 16 | `C*x/(C+x)` needs 13 rules and `C*x/(C+x²)` needs 15. With a cap of 12, forms B and C cannot be in C at all. |
| Sample derivations under the prior-mean theta | Exhaustive enumeration plus tempered sampling | At prior-mean theta, form C has probability of about 10⁻⁷ per draw, so sampling would never find it. |
| "Standardise x and y per dataset" | **Pooled** RMS scaling, the same for all datasets (`scaling="pooled"`). The literal per-dataset version is kept as a sensitivity option. | Centring changes the skeleton: `C*exp(C*x)` becomes `C*exp(C*x)+C`. Per-dataset rescaling leaks each dataset's amplitude, so constant-free skeletons match it for free. In one development replicate, `x**(1/4)` beat the true `C*x/(C+x)` by 3.8 nats this way. |
| Absorb constants (`C*C → C`, `f(C) → C`, …) | Absorb only when the merged piece contains a constant that occurs nowhere else; numeric literals stay literals | Otherwise absorption adds a degree of freedom. For example, `(x+c)²` expanded would turn 1 free constant into 2. |
| Canonicalisation steps 1–4 | Also: merge like terms (`c1*x − x → c*x`); normalise denominators, signs included (`x/(a*x+b)`, `x/(c−x)`); combine nested fractions; choose among the raw, simplified, expanded and combined forms by (constants, divisions, size); rename constants by the permutation with the smallest srepr; make keys survive `str → sympify`; probe-merge | Each one fixes a split seen in development (section 4a). |
| Rule counts | Conventions for cases the spec leaves open: a numeric literal counts as `T → C`; `1/x` counts as `C/x`; non-half powers count as `exp(p·log b)`; a lone sign counts as a constant factor | The grammar cannot produce these trees directly. |
| Baselines 1–3 | Same three, plus `no_pool_bma`: the independent per-dataset posterior, with averaging over skeletons | Separates the effect of pooling from the effect of averaging over skeletons in the held-out density comparison. |

Kept as in the spec, and still open: the base measure is not renormalised
over C, and the size-cap truncation constant of the PCFG is ignored in the
theta update.

## 3. Validation

**Marginal likelihood** (`tests/test_likelihood.py`). Laplace minus the
truth, where the truth is 2-D quadrature for `C*x` and importance sampling
(multivariate-t proposal, 20k draws, ESS > 1,000) for the others:

| skeleton | n=5 | n=10 | n=20 | n=40 |
|---|---|---|---|---|
| `C*x` (quadrature) | −0.22 | −0.10 | −0.05 | −0.02 |
| `C*exp(C*x)` | −0.73 | −0.28 | −0.12 | |
| `C*x/(C+x)` | −1.05 | −0.38 | −0.25 | |
| `C*x/(C+x²)` | −0.79 | −0.29 | −0.14 | |

Laplace runs low by roughly 0.1–0.35 nats per parameter (log sigma
included) at n = 5, and by less than 0.15 nats per parameter at n ≥ 10. The bias is similar for
skeletons with the same number of constants, so comparisons between them
suffer less than the raw numbers suggest. BIC differs from the true log
marginal likelihood by +8 to +10.5 nats in these settings. It is a different
penalty, not an approximation: ½·log n per constant instead of the
prior-volume term.

**Sampler** (`tests/test_gibbs.py`), all passing:

- marginals match the exact posterior from enumerating every assignment, with and without the block move, including a case with large gamma·P(e);
- the number of distinct forms matches the CRP's E[K] under flat likelihoods;
- the Escobar–West draws match the numerically integrated posterior mean of gamma;
- the theta update counts each distinct skeleton once;
- a sharp L gives back the known assignment;
- the block move frees a cluster that single-site Gibbs keeps stuck.

**Canonicalisation** (`tests/test_canonical.py`): 11 groups of forms that must
merge, 7 pairs that must stay apart, variable-free forms dropped, idempotence
and re-parse stability on a sample of the shipped candidates, and a check
that canonicalisation preserves the function family. For 25 random
derivations, the canonical skeleton is refitted to curves of the original
expression, with relative RMS error below 10⁻³.

## 4. What development runs have shown so far

These are single replicates (J = 30, noise 5%) used for debugging. They are
not benchmark results.

**(a) Missed equivalences matter, and so does the choice of parameterisation.**
Before denominator normalisation, all 12 datasets generated from `C*x/(C+x)`
went to `c0/(c1 + 1/x)`. That is the same family with the same maximum
likelihood on every dataset, but Laplace preferred it by 4.3 nats in total,
because the N(0, 3²) prior on constants is not invariant to
reparameterisation. Merging fixes the label, but the general point remains:
under this prior the marginal likelihood of a family depends on which
parameterisation is chosen as canonical.

**(b) Structure is pooled; constants are not.** At n = 10, over the 19
datasets from form A, summed BIC favours the true `C*exp(C*x)` over
`C*exp(-2*x)` by 53 nats, yet summed Laplace favours `C*exp(-2*x)` by 2.2
nats. Each dataset pays about 3 nats of Occam penalty for its own rate
constant under a prior with sd 3, while the true rates are tightly
clustered (−0.6 with sd about 0.2 in original units).
Pooling structure cannot remove that penalty, because each dataset pays it
separately. At small n_j this is probably the largest limitation of the stage-A
model. The natural fix is a hierarchical prior on constants within a skeleton
cluster, `phi_j | e ~ N(mu_e, Sigma_e)`, which also connects the model to
multi-view SR. The `--tau 1` sensitivity run in the benchmark tests a cheap
version: a narrower prior.

**(c) Mixing.** Without the block move, the two chains of the quick-start
example (J = 12, n = 20) settled in different modes. Every dataset got
p = 0.50, and the posterior over the number of forms was bimodal (2: 0.49,
5: 0.44). With the block move, the chains agree on the partition (2: 0.61,
3: 0.38). This is the rich-get-richer risk of spec section 8 in its mixing
form.

**(d) Constant-free look-alikes.** With 3,368 candidates, some constant-free
curves fit small datasets by chance; for example `sin(sqrt(x))` fits
saturating data at n = 10. Without pooling they win often. Pooling
penalises them through the γ·P(e) cost of opening a new cluster.

**(e) Indicative numbers.** In one replicate at n = 10 (code version before
the block move), recovery of the true form was 0.20 for the hierarchical
model, 0.00 for both baselines and 0.93 for the oracle under Laplace.
Under BIC it was 0.47, 0.03 and 0.93. The hierarchical model also had the
best held-out density beyond the observed range, among the non-oracle
methods. These numbers are for orientation only; the benchmark will replace
them.

## 5. The benchmark (to be run)

`benchmarks/matrix.py` defines 40 GitHub Actions jobs, each running 20
replicate populations of J = 30 datasets:

- **main grid:** settings {mixture of A/B/C at 0.6/0.3/0.1, all-A, 30 distinct
  forms from the grammar} × n_j ∈ {5, 10, 20, 50} × noise ∈ {5%, 15%} of each
  curve's range;
- **sensitivity runs** (mixture, 5% noise, every n_j):
  - alpha × 0.25 and × 4;
  - gamma prior Gamma(1, 0.1), Gamma(1, 10), and gamma fixed at 1;
  - per-dataset scaling;
  - tau = 1;
  - data-driven one-step edits added to C (a first version of the spec's
    "feed the posterior back into the search").

Every replicate fits both Laplace and BIC, so the Laplace-vs-BIC comparison
in spec section 7 comes free with every cell. The metrics are those in spec
section 6: recovery, held-out density inside the range and 20% beyond it,
ARI, population-weight error, and calibration. `benchmarks/summarise.py`
turns the raw JSON lines into `summary.md` / `summary.csv`, one plot per
metric (metric against n_j, by method, with standard-error bars) and a
calibration plot.

The hypothesis to test: at small n_j the hierarchical model beats no pooling
and complete pooling on recovery and held-out density, and it loses nothing
in the all-A and grammar settings. Point (b) above is a specific reason it
might fail under Laplace at small n_j.

## 6. Known limitations and next steps

- Hierarchical constants within a cluster (section 4b), then MCMC over trees
  instead of a fixed candidate set.
- Some families are still not merged by rule: `log(a*x + b)` against
  `C + log(x + C)`, and families equal only up to a boundary of parameter
  space (`c*x/(a + x)` against `c/(a + 1/x)`). Probe merging is
  deliberately conservative.
- The Laplace bias at n = 5 is about −0.1 to −0.35 per parameter.
  Importance sampling (`log_ml_importance`) could replace it for each
  dataset's top candidates.
- `compute_likelihoods` assumes equal n_j. `fit_skeleton` already accepts a
  mask for unequal sizes.
- The grammar setting draws its 30 forms from those in C, with N(0, 1.5²)
  constants and a check that each curve is non-degenerate, so recovery is
  possible in principle.
