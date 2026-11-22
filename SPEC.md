# Hierarchical symbolic regression: method spec

Purpose of this file: a self-contained description of a proposed method, written so that a coding
agent can implement a first prototype and test it on synthetic data. Nothing here has been
implemented or validated yet. Numbers marked "default" are starting points, not tuned values.

## 1. Problem

We have J related datasets, D_j = {(x_ji, y_ji)}, i = 1..n_j. Each n_j may be small (5 to 50).
We believe each dataset follows some closed-form equation, that a few functional forms recur across
datasets, and that we do not know which forms or how common each is.

Existing options are both extremes:

- Independent symbolic regression (SR) per dataset: no sharing, unreliable when n_j is small.
- Multi-view SR: one functional form forced on all datasets, only constants vary.

Goal: partial pooling of *equation structure*. Each dataset gets its own expression, drawn from a
shared distribution over expressions that is itself inferred from all datasets.

## 2. Generative model

```
theta_A        ~ Dirichlet(alpha_A)           for each grammar nonterminal A
G              ~ DP(gamma, P_theta)           random distribution over expressions
e_j | G        ~ G                            expression (skeleton) for dataset j
phi_j | e_j    ~ pi(phi | e_j)                constants of e_j for dataset j
y_ji           ~ Normal(f_{e_j}(x_ji; phi_j), sigma_j^2)
gamma          ~ Gamma(a, b)                  default a = 1, b = 1
```

- An expression e is a *skeleton*: a formula with placeholder constants C, e.g. `C*exp(C*x)`.
- P_theta is the distribution over skeletons defined by a probabilistic context-free grammar
  (section 3).
- G is discrete, so several datasets can share the same skeleton. gamma -> 0 gives one skeleton
  for all datasets; gamma -> infinity gives independent draws from the grammar.

With G integrated out (Chinese restaurant process), where n_e^{-j} is the number of datasets other
than j currently assigned skeleton e:

```
P(e_j = e | e_{-j}, y_j)  ∝  ( n_e^{-j} + gamma * P_theta(e) ) * p(y_j | e)
p(y_j | e) = ∫ p(y_j | e, phi, sigma) pi(phi) pi(sigma) dphi dsigma     (marginal likelihood)
```

## 3. Grammar and its prior

```
E -> E + F | E - F | F
F -> F * T | F / T | T
T -> ( E ) | R ( E ) | V | C
R -> exp | log | sin | sqrt
V -> x1 | ... | xd
C -> const
```

- The E/F/T layering encodes precedence, so each written expression has exactly one derivation.
- P_theta(e) = product over rules used in the derivation of e of that rule's probability.
- Rule probabilities for nonterminal A are a vector theta_A with a Dirichlet prior. Defaults:

| Nonterminal | Options (in order)      | alpha_A              |
|-------------|-------------------------|----------------------|
| E           | E+F, E-F, F             | (1, 1, 4)            |
| F           | F*T, F/T, T             | (1, 1, 4)            |
| T           | (E), R(E), V, C         | (0.5, 1, 3, 2)       |
| R           | exp, log, sin, sqrt     | (0.5, 0.5, 0.5, 0.5) |
| V           | x1..xd                  | (1, ..., 1)          |

- Conjugate update given the current assignments. Each *distinct* skeleton in use contributes its
  rule counts once (not once per dataset):
  `theta_A | assignments ~ Dirichlet(alpha_A + sum over distinct skeletons in use of counts_A(e))`
- Cap derivations at a maximum size (default: 12 rule applications from E/F/T, or depth 4) so that
  expression size stays finite whatever theta is drawn.

### Rule counts from an expression tree

To compute P_theta(e) for a skeleton held as a tree (e.g. sympy):

- A sum/difference of m terms uses `E -> E+F` or `E -> E-F` (m-1 times in total) and `E -> F` once.
- A product/quotient of k factors uses `F -> F*T` or `F -> F/T` (k-1 times) and `F -> T` once.
- Each atom uses one T rule. A sum appearing as a factor or function argument needs `T -> (E)`
  or sits directly inside `R(E)`.
- Integer power x^p counts as p-1 multiplications. Unary minus counts as multiplication by a constant.

Write unit tests for this on a few hand-derived examples.

## 4. Canonicalisation

Different trees can be the same function. Before counting or comparing skeletons:

1. Simplify with sympy.
2. Absorb constants: `C*C -> C`, `C+C -> C`, `C*(C*x) -> C*x`, `f(C) -> C`, and drop skeletons with
   no variable.
3. Rename constants in order of appearance and sort commutative arguments.
4. Use the resulting string as the skeleton's identity.

This is imperfect. Log how many candidates merge, and treat unmerged equivalents as a known
source of error.

## 5. Inference, stage A (implement this first)

Two stages: build a finite candidate set, then run a Gibbs sampler over assignments.

**Step 1. Candidates.** Build a set C of canonical skeletons. Default, dependency-free route:
enumerate or sample many derivations from the grammar under the prior mean of theta, up to the size
cap, canonicalise, and deduplicate (target a few hundred to a few thousand skeletons, at most 4
constants each). Optional alternative: run PySR on each dataset with the same operator set, keep
the top 10 per dataset, and take the union.

**Step 2. Marginal likelihoods.** For every skeleton e in C and every dataset j, compute
`L[j, e] ≈ log p(y_j | e)`:

- Standardise x and y per dataset first. Prior on constants: Normal(0, 3^2) each. Prior on sigma:
  HalfNormal(1).
- Find the MAP of (phi, log sigma) by multi-start optimisation (default 10 starts; the objective
  is non-convex).
- Laplace approximation with k = number of free parameters including log sigma:
  `log p(y|e) ≈ log p(y | MAP) + log prior(MAP) + (k/2) log(2*pi) - 0.5 * log det(H)`
  where H is the Hessian of the negative log joint at the MAP. If H is not positive definite or the
  fit produces NaN/inf (e.g. log of a negative), set L[j, e] = -inf.
- Also compute the BIC version, `max log-likelihood - (k/2) log n_j`, for comparison.

**Step 3. Gibbs sampler.** State: assignment z_j in C for each dataset, theta, gamma.

1. Initialise z_j = argmax_e (log P_theta(e) + L[j, e]).
2. For each j in random order, resample z_j from
   `P(z_j = e) ∝ (n_e^{-j} + gamma * P_theta(e)) * exp(L[j, e])`, computed in log space.
3. Resample each theta_A from its Dirichlet posterior (section 3), then recompute log P_theta(e)
   for all e in C.
4. Resample gamma with the Escobar and West (1995) auxiliary-variable step, using the current
   number of distinct skeletons in use.
5. Repeat for a few thousand sweeps; discard burn-in; run 4 chains from different seeds.

Outputs: for each dataset, posterior probabilities over skeletons; posterior over the number of
distinct forms; posterior population weight of each form; posterior of theta and gamma.

Truncating to C means a form absent from C can never be selected. Later stages (not for this
prototype) feed the posterior back into the search to add candidates, and replace the candidate
set with MCMC over trees.

## 6. Synthetic experiment

- Forms and population weights: A = `C*exp(C*x)` (0.6), B = `C*x/(C+x)` (0.3),
  C = `C*x/(C+x*x)` (0.1).
- J = 30 datasets. One input variable, x ~ Uniform(0.1, 5). Constants drawn per dataset from
  log-normal distributions chosen so curves are visibly non-linear on that range (negative rate
  for form A).
- Gaussian noise with sd = 5% and 15% of each curve's range.
- n_j in {5, 10, 20, 50}, same for all datasets within a run. 20 replicate populations per setting.
- Extra settings: all datasets from form A (pooling should help most) and 30 distinct forms
  sampled from the grammar (pooling should not hurt).

Baselines, all using the same C and the same L matrix:

1. No pooling: z_j = argmax_e (log P_theta0(e) + L[j, e]) with theta fixed at its prior mean.
2. Complete pooling: the single e maximising sum_j L[j, e].
3. Oracle list: the Gibbs sampler with C restricted to the three true forms.

Metrics:

- Fraction of datasets whose top-posterior skeleton equals the true form (after canonicalisation).
- Held-out log predictive density on 20 new points per dataset, inside and 20% beyond the x range.
- Adjusted Rand index between inferred and true grouping of datasets.
- Absolute error in estimated population weights of the true forms.
- Calibration: reliability curve for the posterior probability assigned to the true form.

Hypothesis to test: at small n_j the hierarchical model beats both baselines 1 and 2 on form
recovery and held-out density, and loses nothing at the two extreme settings. Report the result
either way.

## 7. Deliverables

- Python package: grammar (sampling, rule counts, log-probability), canonicalisation, marginal
  likelihood, Gibbs sampler, baselines, experiment runner. numpy/scipy/sympy only, plus optional PySR.
- Unit tests: rule counts, P_theta on hand examples, canonicalisation merges, sampler recovering a
  known assignment when L is made artificially sharp.
- A results table and plots of each metric against n_j, per method, with uncertainty over replicates.
- A short report: what worked, what failed, and how sensitive results are to the Laplace vs BIC
  choice, to alpha, and to the gamma prior.

## 8. Known risks to watch for

- Marginal likelihoods are least accurate at small n_j, where the method matters most. Compare
  Laplace against BIC and, for a few cases, against brute-force numerical integration.
- Rich-get-richer: the sampler can lock onto a wrong shared form early. Check agreement across
  chains and sensitivity to gamma.
- Missed equivalences split clusters and weaken pooling.
- exp and division produce overflow and poles; guard evaluations and treat failures as -inf.

## 9. Closest prior work

- Liang, Jordan, Klein (2010), "Learning Programs: A Hierarchical Bayesian Approach": same
  hierarchy for programs.
- Stephan et al. (2009), "Bayesian model selection for group studies": same hierarchy for a fixed,
  finite model list. This method replaces the list with a grammar.
- Brence, Todorovski, Džeroski (2021), "Probabilistic Grammars for Equation Discovery": PCFG priors
  over equations, not learned across datasets.
- Guimerà et al. (2020), "A Bayesian machine scientist": MCMC over expressions with a fixed
  corpus-based prior.
- Russeil et al. (2024), "Multi-View Symbolic Regression": one shared form, varying constants.
