# Look-alike discrimination test

When the data barely distinguish candidate forms, does the prior put its mass on the true one? True forms are drawn from the hold-out sources (seen and novel relative to the training sources), the mechanism classes (hsr.mechanisms) and the forms first used from the temporal cutoff on; datasets are simulated from each (hsr.simulate: constants N(0, 1.5^2) with rejection, x uniform on [0.1, 5], noise sd = noise x range of the curve); every candidate's Laplace log evidence is computed on every dataset (hsr.likelihood; GitHub Actions, .github/workflows/lookalike.yml). Design and scores: hsr.lookalike. Priors: trained on all training sources (runs mechholdout_<weighting>; for temporal true forms the priors trained on models dated before the cutoff, run temporal_<cutoff>_<weighting>), raw log P(form). The main tables use equal source weighting; pooled weighting is compared at the end of each design's section. Prior tempering: lookalike_tempering.md.

Look-alike set A_D: candidates within D nats of the best log evidence. rank: rank of the truth in A_D under the prior (1 = the prior's favourite, ties count half), given the truth is in A_D; top: fraction of those datasets where the truth is the prior's strict favourite in A_D; mass: prior probability of the truth renormalised within A_D (flat: 1/|A_D|); gain: median over datasets of log(mass x |A_D|), the log factor by which the prior beats the flat prior on the truth; P(true): posterior probability of the truth over the whole candidate set (log evidence + log prior); MAP: fraction of datasets where the truth is the posterior mode.

## Designs

| design | max constants | max nodes | candidates | true forms | datasets | true fit | evidence CPU h | Actions run |
|---|---|---|---|---|---|---|---|---|
| Design A: at most 5 constants and 15 nodes | 5 | 15 | 5140 | 49 | 588 | 93.9% | 3.3 | 45719012957 |
| Design B: at most 7 constants and 25 nodes | 7 | 25 | 5792 | 73 | 876 | 89.3% | 7.6 | 45902288688 |

## Design A: at most 5 constants and 15 nodes

Candidate set: the 5140 support forms of the mechanism hold-out run with at most 5 constants and 15 nodes (5116 of them in the temporal run's support, the candidate set for temporal true forms). 588 datasets: 49 true forms x 4 (n, noise) settings x 3 replicates. The true form's fit succeeded on 93.9% of datasets. Evidence: 3.3 CPU hours (GitHub Actions run 45719012957).

### True forms

| stratum | eligible distinct forms | chosen | seen | not simulable (skipped) |
|---|---|---|---|---|
| holdout_seen | 73 | 10 | 10 | 0 |
| holdout_novel | 25 | 10 | 0 | 0 |
| mech_enzyme | 6 | 6 | 6 | 0 |
| mech_compartment | 8 | 8 | 6 | 0 |
| mech_temperature | 3 | 3 | 2 | 0 |
| temporal_novel | 118 | 12 | 0 | 0 |

### How often is the truth a look-alike?

Fraction of datasets whose true form is in A_D, and the median size of A_D.

| group | datasets | truth in A_2 | median A_2 | truth in A_5 | median A_5 | truth in A_10 | median A_10 |
|---|---|---|---|---|---|---|---|
| all | 588 | 0.14 | 6 | 0.43 | 53 | 0.78 | 270 |
| holdout_seen | 120 | 0.17 | 6 | 0.62 | 54 | 0.81 | 258 |
| holdout_novel | 120 | 0.10 | 8 | 0.28 | 60 | 0.77 | 322 |
| mech_enzyme | 72 | 0.11 | 4 | 0.38 | 49 | 0.62 | 237 |
| mech_compartment | 96 | 0.12 | 6 | 0.44 | 50 | 0.92 | 256 |
| mech_temperature | 36 | 0.17 | 5 | 0.53 | 44 | 0.75 | 206 |
| temporal_novel | 144 | 0.16 | 6 | 0.40 | 45 | 0.78 | 231 |
| seen | 288 | 0.14 | 5 | 0.49 | 54 | 0.77 | 254 |
| novel | 300 | 0.14 | 6 | 0.37 | 53 | 0.80 | 284 |
| n=10, noise=0.02 | 147 | 0.12 | 6 | 0.38 | 50 | 0.76 | 283 |
| n=10, noise=0.1 | 147 | 0.02 | 8 | 0.36 | 102 | 0.82 | 597 |
| n=40, noise=0.02 | 147 | 0.36 | 4 | 0.60 | 30 | 0.83 | 135 |
| n=40, noise=0.1 | 147 | 0.06 | 5 | 0.39 | 54 | 0.73 | 284 |

### All datasets, D = 2

Truth in A_2 on 82 of 588 datasets.

| prior | family | mean rank | rank percentile | top | mass | gain | P(true) | MAP |
|---|---|---|---|---|---|---|---|---|
| flat | - | 4.34 | 0.42 | 0.16 | 0.319 | 0.00 | 0.045 | 0.06 |
| hpy3_depth | tree | 2.35 | 0.26 | 0.55 | 0.531 | 0.40 | 0.080 | 0.10 |
| fnprior_kn_tuned | tree | 2.36 | 0.24 | 0.54 | 0.521 | 0.00 | 0.080 | 0.09 |
| fnprior_kn | tree | 2.36 | 0.24 | 0.54 | 0.520 | 0.00 | 0.080 | 0.09 |
| hpy2 | tree | 2.36 | 0.24 | 0.54 | 0.521 | 0.00 | 0.080 | 0.09 |
| hpy | tree | 2.36 | 0.24 | 0.54 | 0.520 | 0.00 | 0.080 | 0.09 |
| sym0 | tree | 2.35 | 0.23 | 0.59 | 0.552 | 0.00 | 0.079 | 0.09 |
| form_ad | tree | 2.63 | 0.32 | 0.51 | 0.484 | 0.00 | 0.074 | 0.09 |
| form_ad3_depth | tree | 2.70 | 0.33 | 0.51 | 0.483 | 0.00 | 0.074 | 0.09 |
| tree_ad3 | tree | 2.74 | 0.33 | 0.51 | 0.482 | 0.00 | 0.074 | 0.09 |
| hpy2_src | tree | 2.52 | 0.26 | 0.52 | 0.503 | 0.00 | 0.071 | 0.08 |
| uniform symbols | tree | 3.38 | 0.34 | 0.37 | 0.423 | 0.00 | 0.043 | 0.05 |
| cf | pcfg | 2.41 | 0.25 | 0.56 | 0.550 | 0.00 | 0.077 | 0.08 |
| position_backoff20 | pcfg | 2.34 | 0.24 | 0.57 | 0.546 | 0.00 | 0.072 | 0.07 |
| position | pcfg | 2.37 | 0.24 | 0.57 | 0.546 | 0.00 | 0.072 | 0.07 |
| parent | pcfg | 2.46 | 0.25 | 0.56 | 0.533 | 0.00 | 0.072 | 0.07 |
| parent_backoff20 | pcfg | 2.44 | 0.25 | 0.56 | 0.533 | 0.00 | 0.072 | 0.07 |
| term_root_backoff20 | pcfg | 2.35 | 0.24 | 0.56 | 0.536 | 0.00 | 0.068 | 0.07 |
| term_root | pcfg | 2.35 | 0.24 | 0.56 | 0.536 | 0.00 | 0.068 | 0.07 |
| term_backoff20 | pcfg | 2.43 | 0.25 | 0.55 | 0.535 | 0.00 | 0.067 | 0.07 |
| term | pcfg | 2.40 | 0.25 | 0.55 | 0.533 | 0.00 | 0.067 | 0.07 |
| default_prior | pcfg | 3.20 | 0.33 | 0.43 | 0.441 | 0.00 | 0.055 | 0.06 |

### All datasets, D = 5

Truth in A_5 on 254 of 588 datasets.

| prior | family | mean rank | rank percentile | top | mass | gain | P(true) | MAP |
|---|---|---|---|---|---|---|---|---|
| flat | - | 48.74 | 0.49 | 0.01 | 0.053 | 0.00 | 0.045 | 0.06 |
| hpy3_depth | tree | 16.68 | 0.25 | 0.15 | 0.140 | 0.15 | 0.080 | 0.10 |
| fnprior_kn_tuned | tree | 17.47 | 0.25 | 0.13 | 0.140 | -0.18 | 0.080 | 0.09 |
| fnprior_kn | tree | 17.54 | 0.25 | 0.13 | 0.140 | -0.18 | 0.080 | 0.09 |
| hpy2 | tree | 17.51 | 0.25 | 0.13 | 0.140 | -0.18 | 0.080 | 0.09 |
| hpy | tree | 17.52 | 0.25 | 0.13 | 0.139 | -0.18 | 0.080 | 0.09 |
| sym0 | tree | 23.66 | 0.27 | 0.19 | 0.180 | -2.00 | 0.079 | 0.09 |
| form_ad | tree | 20.30 | 0.29 | 0.16 | 0.136 | 0.02 | 0.074 | 0.09 |
| form_ad3_depth | tree | 19.37 | 0.29 | 0.16 | 0.136 | 0.02 | 0.074 | 0.09 |
| tree_ad3 | tree | 19.35 | 0.29 | 0.16 | 0.136 | 0.02 | 0.074 | 0.09 |
| hpy2_src | tree | 19.68 | 0.27 | 0.13 | 0.134 | -1.11 | 0.071 | 0.08 |
| uniform symbols | tree | 49.48 | 0.46 | 0.09 | 0.111 | -5.95 | 0.043 | 0.05 |
| cf | pcfg | 24.04 | 0.29 | 0.16 | 0.174 | -1.57 | 0.077 | 0.08 |
| position_backoff20 | pcfg | 22.46 | 0.27 | 0.15 | 0.147 | -1.46 | 0.072 | 0.07 |
| position | pcfg | 22.64 | 0.27 | 0.14 | 0.147 | -1.46 | 0.072 | 0.07 |
| parent | pcfg | 23.32 | 0.30 | 0.15 | 0.156 | -1.83 | 0.072 | 0.07 |
| parent_backoff20 | pcfg | 23.20 | 0.29 | 0.15 | 0.156 | -1.83 | 0.072 | 0.07 |
| term_root_backoff20 | pcfg | 22.23 | 0.29 | 0.14 | 0.136 | -1.51 | 0.068 | 0.07 |
| term_root | pcfg | 22.59 | 0.29 | 0.14 | 0.136 | -1.51 | 0.068 | 0.07 |
| term_backoff20 | pcfg | 22.19 | 0.29 | 0.12 | 0.133 | -1.55 | 0.067 | 0.07 |
| term | pcfg | 22.67 | 0.29 | 0.12 | 0.133 | -1.55 | 0.067 | 0.07 |
| default_prior | pcfg | 45.35 | 0.42 | 0.11 | 0.124 | -3.61 | 0.055 | 0.06 |

### All datasets, D = 10

Truth in A_10 on 461 of 588 datasets.

| prior | family | mean rank | rank percentile | top | mass | gain | P(true) | MAP |
|---|---|---|---|---|---|---|---|---|
| flat | - | 244.63 | 0.50 | 0.00 | 0.010 | 0.00 | 0.045 | 0.06 |
| hpy3_depth | tree | 98.17 | 0.28 | 0.03 | 0.023 | -1.24 | 0.080 | 0.10 |
| fnprior_kn_tuned | tree | 101.89 | 0.28 | 0.03 | 0.031 | -2.20 | 0.080 | 0.09 |
| fnprior_kn | tree | 103.10 | 0.28 | 0.03 | 0.031 | -2.20 | 0.080 | 0.09 |
| hpy2 | tree | 102.25 | 0.28 | 0.03 | 0.031 | -2.20 | 0.080 | 0.09 |
| hpy | tree | 102.23 | 0.28 | 0.03 | 0.031 | -2.19 | 0.080 | 0.09 |
| sym0 | tree | 163.38 | 0.33 | 0.05 | 0.053 | -5.20 | 0.079 | 0.09 |
| form_ad | tree | 122.45 | 0.34 | 0.03 | 0.021 | -2.33 | 0.074 | 0.09 |
| form_ad3_depth | tree | 117.46 | 0.33 | 0.03 | 0.021 | -2.33 | 0.074 | 0.09 |
| tree_ad3 | tree | 121.13 | 0.34 | 0.03 | 0.021 | -2.33 | 0.074 | 0.09 |
| hpy2_src | tree | 119.12 | 0.32 | 0.03 | 0.028 | -3.08 | 0.071 | 0.08 |
| uniform symbols | tree | 310.61 | 0.55 | 0.02 | 0.030 | -10.71 | 0.043 | 0.05 |
| cf | pcfg | 176.38 | 0.36 | 0.05 | 0.050 | -5.16 | 0.077 | 0.08 |
| position_backoff20 | pcfg | 147.65 | 0.33 | 0.03 | 0.036 | -4.48 | 0.072 | 0.07 |
| position | pcfg | 149.67 | 0.33 | 0.03 | 0.036 | -4.47 | 0.072 | 0.07 |
| parent | pcfg | 161.41 | 0.35 | 0.03 | 0.041 | -4.81 | 0.072 | 0.07 |
| parent_backoff20 | pcfg | 160.46 | 0.35 | 0.03 | 0.041 | -4.81 | 0.072 | 0.07 |
| term_root_backoff20 | pcfg | 144.32 | 0.34 | 0.03 | 0.030 | -4.52 | 0.068 | 0.07 |
| term_root | pcfg | 148.42 | 0.35 | 0.03 | 0.030 | -4.52 | 0.068 | 0.07 |
| term_backoff20 | pcfg | 144.55 | 0.33 | 0.03 | 0.030 | -4.71 | 0.067 | 0.07 |
| term | pcfg | 149.30 | 0.34 | 0.03 | 0.030 | -4.71 | 0.067 | 0.07 |
| default_prior | pcfg | 293.57 | 0.52 | 0.03 | 0.036 | -7.90 | 0.055 | 0.06 |

### Renormalised prior mass on the truth within A_5, by stratum

| prior | holdout_seen | holdout_novel | mech_enzyme | mech_compartment | mech_temperature | temporal_novel | seen | novel |
|---|---|---|---|---|---|---|---|---|
| datasets | 120 | 120 | 72 | 96 | 36 | 144 | 288 | 300 |
| flat | 0.023 | 0.063 | 0.042 | 0.061 | 0.025 | 0.093 | 0.028 | 0.085 |
| hpy3_depth | 0.224 | 0.032 | 0.259 | 0.152 | 0.075 | 0.055 | 0.208 | 0.055 |
| fnprior_kn_tuned | 0.241 | 0.036 | 0.240 | 0.156 | 0.015 | 0.052 | 0.206 | 0.055 |
| fnprior_kn | 0.241 | 0.036 | 0.240 | 0.156 | 0.015 | 0.052 | 0.206 | 0.055 |
| hpy2 | 0.241 | 0.036 | 0.240 | 0.156 | 0.015 | 0.052 | 0.206 | 0.055 |
| hpy | 0.241 | 0.036 | 0.240 | 0.156 | 0.015 | 0.052 | 0.206 | 0.055 |
| sym0 | 0.292 | 0.055 | 0.181 | 0.194 | 0.232 | 0.084 | 0.258 | 0.082 |
| form_ad | 0.204 | 0.029 | 0.284 | 0.157 | 0.045 | 0.058 | 0.200 | 0.055 |
| form_ad3_depth | 0.204 | 0.029 | 0.284 | 0.157 | 0.045 | 0.058 | 0.200 | 0.055 |
| tree_ad3 | 0.204 | 0.029 | 0.284 | 0.157 | 0.045 | 0.058 | 0.200 | 0.055 |
| hpy2_src | 0.223 | 0.039 | 0.330 | 0.105 | 0.011 | 0.046 | 0.196 | 0.055 |
| uniform symbols | 0.204 | 0.067 | 0.072 | 0.093 | 0.102 | 0.051 | 0.149 | 0.062 |
| cf | 0.307 | 0.057 | 0.071 | 0.216 | 0.264 | 0.062 | 0.240 | 0.091 |
| position_backoff20 | 0.226 | 0.055 | 0.120 | 0.196 | 0.199 | 0.059 | 0.204 | 0.074 |
| position | 0.226 | 0.055 | 0.120 | 0.196 | 0.199 | 0.059 | 0.204 | 0.074 |
| parent | 0.267 | 0.059 | 0.066 | 0.200 | 0.222 | 0.059 | 0.216 | 0.079 |
| parent_backoff20 | 0.267 | 0.059 | 0.066 | 0.200 | 0.222 | 0.059 | 0.216 | 0.079 |
| term_root_backoff20 | 0.213 | 0.051 | 0.162 | 0.148 | 0.172 | 0.057 | 0.195 | 0.062 |
| term_root | 0.213 | 0.051 | 0.162 | 0.148 | 0.172 | 0.057 | 0.195 | 0.062 |
| term_backoff20 | 0.216 | 0.052 | 0.108 | 0.147 | 0.180 | 0.060 | 0.188 | 0.063 |
| term | 0.216 | 0.052 | 0.107 | 0.147 | 0.180 | 0.060 | 0.188 | 0.063 |
| default_prior | 0.239 | 0.032 | 0.036 | 0.138 | 0.156 | 0.053 | 0.174 | 0.062 |

### Mean rank of the truth within A_5, by stratum

| prior | holdout_seen | holdout_novel | mech_enzyme | mech_compartment | mech_temperature | temporal_novel | seen | novel |
|---|---|---|---|---|---|---|---|---|
| datasets | 120 | 120 | 72 | 96 | 36 | 144 | 288 | 300 |
| flat | 50.80 | 68.44 | 53.93 | 52.46 | 46.97 | 30.02 | 53.15 | 43.13 |
| hpy3_depth | 5.55 | 48.21 | 8.35 | 7.21 | 13.00 | 24.34 | 6.44 | 29.67 |
| fnprior_kn_tuned | 5.97 | 43.21 | 9.24 | 9.19 | 21.42 | 25.59 | 7.67 | 29.89 |
| fnprior_kn | 5.97 | 43.32 | 9.24 | 9.19 | 21.47 | 25.79 | 7.68 | 30.04 |
| hpy2 | 5.97 | 43.38 | 9.24 | 9.19 | 21.74 | 25.55 | 7.67 | 29.98 |
| hpy | 5.97 | 43.29 | 9.24 | 9.19 | 21.42 | 25.78 | 7.67 | 30.02 |
| sym0 | 10.67 | 61.09 | 39.00 | 9.00 | 12.26 | 25.51 | 15.50 | 34.00 |
| form_ad | 5.99 | 59.59 | 7.39 | 11.26 | 21.58 | 27.67 | 6.55 | 37.74 |
| form_ad3_depth | 5.97 | 55.47 | 7.39 | 11.21 | 19.47 | 26.76 | 6.53 | 35.65 |
| tree_ad3 | 6.01 | 53.15 | 7.39 | 11.24 | 20.05 | 27.76 | 6.56 | 35.56 |
| hpy2_src | 6.32 | 45.97 | 7.06 | 15.26 | 26.95 | 28.02 | 8.73 | 33.56 |
| uniform symbols | 23.73 | 111.07 | 68.41 | 45.94 | 32.45 | 45.57 | 37.82 | 64.27 |
| cf | 11.66 | 59.34 | 28.00 | 10.88 | 10.95 | 31.13 | 14.25 | 36.45 |
| position_backoff20 | 10.82 | 61.03 | 19.41 | 9.70 | 12.00 | 28.81 | 11.82 | 35.96 |
| position | 11.16 | 61.47 | 19.19 | 9.68 | 12.05 | 28.98 | 11.96 | 36.18 |
| parent | 11.62 | 57.71 | 26.31 | 9.46 | 12.42 | 30.29 | 13.62 | 35.61 |
| parent_backoff20 | 11.32 | 57.21 | 26.72 | 9.46 | 12.21 | 30.33 | 13.52 | 35.47 |
| term_root_backoff20 | 10.04 | 57.56 | 16.93 | 12.01 | 12.74 | 30.03 | 11.27 | 36.12 |
| term_root | 10.09 | 58.85 | 16.89 | 12.04 | 12.68 | 30.83 | 11.28 | 36.93 |
| term_backoff20 | 10.15 | 56.24 | 19.33 | 10.56 | 13.95 | 30.03 | 11.58 | 35.63 |
| term | 10.34 | 57.35 | 19.44 | 10.51 | 14.26 | 31.12 | 11.74 | 36.52 |
| default_prior | 20.93 | 109.40 | 62.13 | 37.76 | 22.74 | 44.08 | 32.62 | 61.50 |

### MAP accuracy (posterior mode = truth), by stratum

| prior | holdout_seen | holdout_novel | mech_enzyme | mech_compartment | mech_temperature | temporal_novel | seen | novel |
|---|---|---|---|---|---|---|---|---|
| datasets | 120 | 120 | 72 | 96 | 36 | 144 | 288 | 300 |
| flat | 0.04 | 0.06 | 0.06 | 0.06 | 0.03 | 0.07 | 0.04 | 0.07 |
| hpy3_depth | 0.23 | 0.00 | 0.15 | 0.11 | 0.08 | 0.03 | 0.17 | 0.03 |
| fnprior_kn_tuned | 0.22 | 0.01 | 0.11 | 0.10 | 0.06 | 0.02 | 0.15 | 0.03 |
| fnprior_kn | 0.22 | 0.01 | 0.11 | 0.10 | 0.06 | 0.02 | 0.15 | 0.03 |
| hpy2 | 0.22 | 0.01 | 0.11 | 0.10 | 0.06 | 0.02 | 0.15 | 0.03 |
| hpy | 0.22 | 0.01 | 0.11 | 0.10 | 0.06 | 0.02 | 0.15 | 0.03 |
| sym0 | 0.18 | 0.01 | 0.07 | 0.12 | 0.11 | 0.04 | 0.14 | 0.04 |
| form_ad | 0.21 | 0.00 | 0.14 | 0.10 | 0.06 | 0.02 | 0.15 | 0.02 |
| form_ad3_depth | 0.21 | 0.00 | 0.14 | 0.10 | 0.06 | 0.02 | 0.15 | 0.02 |
| tree_ad3 | 0.21 | 0.00 | 0.14 | 0.10 | 0.06 | 0.02 | 0.15 | 0.02 |
| hpy2_src | 0.18 | 0.01 | 0.15 | 0.09 | 0.06 | 0.01 | 0.14 | 0.02 |
| uniform symbols | 0.12 | 0.00 | 0.03 | 0.04 | 0.06 | 0.03 | 0.07 | 0.02 |
| cf | 0.20 | 0.01 | 0.04 | 0.11 | 0.14 | 0.03 | 0.13 | 0.04 |
| position_backoff20 | 0.15 | 0.01 | 0.06 | 0.12 | 0.11 | 0.03 | 0.11 | 0.03 |
| position | 0.15 | 0.01 | 0.06 | 0.12 | 0.11 | 0.03 | 0.11 | 0.03 |
| parent | 0.15 | 0.02 | 0.04 | 0.12 | 0.11 | 0.03 | 0.11 | 0.04 |
| parent_backoff20 | 0.15 | 0.02 | 0.04 | 0.12 | 0.11 | 0.03 | 0.11 | 0.04 |
| term_root_backoff20 | 0.13 | 0.02 | 0.07 | 0.12 | 0.11 | 0.03 | 0.11 | 0.04 |
| term_root | 0.13 | 0.02 | 0.07 | 0.12 | 0.11 | 0.03 | 0.11 | 0.04 |
| term_backoff20 | 0.14 | 0.01 | 0.06 | 0.12 | 0.11 | 0.03 | 0.11 | 0.03 |
| term | 0.14 | 0.01 | 0.06 | 0.12 | 0.11 | 0.03 | 0.11 | 0.03 |
| default_prior | 0.15 | 0.00 | 0.03 | 0.10 | 0.11 | 0.02 | 0.10 | 0.03 |

### Posterior probability of the truth, by stratum

| prior | holdout_seen | holdout_novel | mech_enzyme | mech_compartment | mech_temperature | temporal_novel | seen | novel |
|---|---|---|---|---|---|---|---|---|
| datasets | 120 | 120 | 72 | 96 | 36 | 144 | 288 | 300 |
| flat | 0.032 | 0.039 | 0.035 | 0.062 | 0.033 | 0.057 | 0.031 | 0.058 |
| hpy3_depth | 0.187 | 0.004 | 0.104 | 0.105 | 0.082 | 0.025 | 0.134 | 0.029 |
| fnprior_kn_tuned | 0.189 | 0.009 | 0.099 | 0.109 | 0.048 | 0.025 | 0.131 | 0.030 |
| fnprior_kn | 0.189 | 0.009 | 0.099 | 0.109 | 0.048 | 0.025 | 0.131 | 0.030 |
| hpy2 | 0.189 | 0.009 | 0.099 | 0.109 | 0.048 | 0.025 | 0.131 | 0.030 |
| hpy | 0.189 | 0.009 | 0.099 | 0.109 | 0.048 | 0.025 | 0.131 | 0.030 |
| sym0 | 0.172 | 0.007 | 0.069 | 0.106 | 0.126 | 0.035 | 0.127 | 0.032 |
| form_ad | 0.173 | 0.000 | 0.107 | 0.105 | 0.052 | 0.022 | 0.127 | 0.024 |
| form_ad3_depth | 0.173 | 0.000 | 0.107 | 0.105 | 0.052 | 0.022 | 0.127 | 0.024 |
| tree_ad3 | 0.173 | 0.000 | 0.107 | 0.105 | 0.052 | 0.022 | 0.127 | 0.024 |
| hpy2_src | 0.164 | 0.010 | 0.121 | 0.084 | 0.035 | 0.019 | 0.117 | 0.026 |
| uniform symbols | 0.111 | 0.004 | 0.032 | 0.049 | 0.050 | 0.020 | 0.069 | 0.018 |
| cf | 0.180 | 0.010 | 0.043 | 0.106 | 0.136 | 0.028 | 0.120 | 0.035 |
| position_backoff20 | 0.149 | 0.012 | 0.058 | 0.113 | 0.116 | 0.027 | 0.112 | 0.034 |
| position | 0.149 | 0.012 | 0.057 | 0.113 | 0.116 | 0.027 | 0.111 | 0.034 |
| parent | 0.158 | 0.012 | 0.036 | 0.111 | 0.125 | 0.027 | 0.110 | 0.035 |
| parent_backoff20 | 0.158 | 0.012 | 0.036 | 0.111 | 0.125 | 0.027 | 0.110 | 0.035 |
| term_root_backoff20 | 0.136 | 0.012 | 0.069 | 0.103 | 0.106 | 0.025 | 0.105 | 0.032 |
| term_root | 0.136 | 0.012 | 0.069 | 0.103 | 0.106 | 0.025 | 0.105 | 0.032 |
| term_backoff20 | 0.141 | 0.011 | 0.053 | 0.101 | 0.109 | 0.027 | 0.104 | 0.032 |
| term | 0.141 | 0.011 | 0.053 | 0.101 | 0.109 | 0.027 | 0.104 | 0.032 |
| default_prior | 0.141 | 0.002 | 0.030 | 0.074 | 0.088 | 0.020 | 0.089 | 0.023 |

### Renormalised prior mass on the truth within A_5, by (n, noise)

| prior | n=10, noise=0.02 | n=10, noise=0.1 | n=40, noise=0.02 | n=40, noise=0.1 |
|---|---|---|---|---|
| datasets | 147 | 147 | 147 | 147 |
| flat | 0.034 | 0.014 | 0.097 | 0.039 |
| hpy3_depth | 0.145 | 0.094 | 0.170 | 0.134 |
| fnprior_kn_tuned | 0.150 | 0.099 | 0.160 | 0.134 |
| fnprior_kn | 0.150 | 0.099 | 0.160 | 0.134 |
| hpy2 | 0.150 | 0.099 | 0.160 | 0.134 |
| hpy | 0.150 | 0.099 | 0.160 | 0.134 |
| sym0 | 0.174 | 0.124 | 0.239 | 0.148 |
| form_ad | 0.154 | 0.102 | 0.154 | 0.123 |
| form_ad3_depth | 0.154 | 0.102 | 0.154 | 0.123 |
| tree_ad3 | 0.154 | 0.102 | 0.154 | 0.123 |
| hpy2_src | 0.131 | 0.103 | 0.152 | 0.137 |
| uniform symbols | 0.111 | 0.057 | 0.174 | 0.062 |
| cf | 0.180 | 0.103 | 0.233 | 0.144 |
| position_backoff20 | 0.160 | 0.089 | 0.201 | 0.103 |
| position | 0.160 | 0.089 | 0.201 | 0.103 |
| parent | 0.162 | 0.093 | 0.210 | 0.124 |
| parent_backoff20 | 0.162 | 0.093 | 0.210 | 0.124 |
| term_root_backoff20 | 0.147 | 0.090 | 0.186 | 0.093 |
| term_root | 0.147 | 0.090 | 0.186 | 0.093 |
| term_backoff20 | 0.143 | 0.084 | 0.184 | 0.090 |
| term | 0.143 | 0.084 | 0.184 | 0.090 |
| default_prior | 0.132 | 0.081 | 0.173 | 0.083 |

### MAP accuracy, by (n, noise)

| prior | n=10, noise=0.02 | n=10, noise=0.1 | n=40, noise=0.02 | n=40, noise=0.1 |
|---|---|---|---|---|
| datasets | 147 | 147 | 147 | 147 |
| flat | 0.04 | 0.00 | 0.16 | 0.03 |
| hpy3_depth | 0.09 | 0.04 | 0.18 | 0.07 |
| fnprior_kn_tuned | 0.08 | 0.03 | 0.17 | 0.05 |
| fnprior_kn | 0.08 | 0.03 | 0.17 | 0.05 |
| hpy2 | 0.08 | 0.03 | 0.17 | 0.05 |
| hpy | 0.08 | 0.03 | 0.17 | 0.05 |
| sym0 | 0.07 | 0.05 | 0.18 | 0.04 |
| form_ad | 0.08 | 0.04 | 0.16 | 0.05 |
| form_ad3_depth | 0.08 | 0.04 | 0.16 | 0.05 |
| tree_ad3 | 0.08 | 0.04 | 0.16 | 0.05 |
| hpy2_src | 0.07 | 0.05 | 0.16 | 0.04 |
| uniform symbols | 0.03 | 0.02 | 0.12 | 0.01 |
| cf | 0.07 | 0.03 | 0.18 | 0.05 |
| position_backoff20 | 0.06 | 0.02 | 0.17 | 0.04 |
| position | 0.06 | 0.02 | 0.17 | 0.04 |
| parent | 0.06 | 0.02 | 0.17 | 0.04 |
| parent_backoff20 | 0.06 | 0.02 | 0.17 | 0.04 |
| term_root_backoff20 | 0.06 | 0.02 | 0.17 | 0.04 |
| term_root | 0.06 | 0.02 | 0.17 | 0.04 |
| term_backoff20 | 0.05 | 0.02 | 0.17 | 0.04 |
| term | 0.05 | 0.02 | 0.17 | 0.04 |
| default_prior | 0.05 | 0.03 | 0.14 | 0.03 |

### True forms used

| stratum | form | seen | constants | nodes |
|---|---|---|---|---|
| holdout_seen | `c0*exp(c1*(c2 + x1)**2)` | yes | 3 | 10 |
| holdout_seen | `c0/x1` | yes | 1 | 3 |
| holdout_seen | `c0*x1**2 + c1*x1 + c2` | yes | 3 | 11 |
| holdout_seen | `c0*(c1*x1 + c2)**c3` | yes | 4 | 9 |
| holdout_seen | `c0*x1**3 + c1` | yes | 2 | 7 |
| holdout_seen | `c0 + x1` | yes | 1 | 3 |
| holdout_seen | `c0*x1**c1/(c2 + x1**c1) + c3` | yes | 4 | 13 |
| holdout_seen | `c0*x1**2 + c1` | yes | 2 | 7 |
| holdout_seen | `c0/(c1 + x1) + c2` | yes | 3 | 7 |
| holdout_seen | `c0*(c2 + x1)**2/(c1 + x1)` | yes | 3 | 12 |
| holdout_novel | `c0*exp(-c1*exp(c2*x1) + c1 + c3*x1)` | no | 4 | 15 |
| holdout_novel | `c0 - (c1*x1 + c2)**c3` | no | 4 | 9 |
| holdout_novel | `(c0*log(x1) + c1)**2` | no | 2 | 8 |
| holdout_novel | `c0*(c1 - exp(c2*x1))*exp(c3*x1)` | no | 4 | 13 |
| holdout_novel | `c0/(c1*log(x1)**3 + c2 + log(x1))` | no | 3 | 13 |
| holdout_novel | `c0*(c1 + x1)**c2*exp(c3*x1)` | no | 4 | 12 |
| holdout_novel | `c0*exp(c1*x1**2 + c2*x1)` | no | 3 | 12 |
| holdout_novel | `log(c0*exp(c1*x1) + c2*exp(c3*x1))` | no | 4 | 14 |
| holdout_novel | `c0*(c1*x1 + c2)**c3/x1` | no | 4 | 12 |
| holdout_novel | `c0*x1 + c1*exp(c2*x1**c3)` | no | 4 | 12 |
| mech_enzyme | `c0/(c1 + x1)` | yes | 2 | 5 |
| mech_enzyme | `c0*x1**2/(c1*x1 + c2 + x1**2)` | yes | 3 | 15 |
| mech_enzyme | `c0*x1/(c1 + x1)` | yes | 2 | 7 |
| mech_enzyme | `(c0 + c1*x1)/(c2 + x1)` | yes | 3 | 10 |
| mech_enzyme | `c0*x1/(c1*x1 + c2 + x1**2)` | yes | 3 | 13 |
| mech_enzyme | `c0/(c1*x1 + c2 + x1**2)` | yes | 3 | 11 |
| mech_compartment | `c0*exp(c1*x1)` | yes | 2 | 6 |
| mech_compartment | `c0*exp(c1*x1) + c2*exp(c3*x1) + c4` | yes | 5 | 15 |
| mech_compartment | `c0*exp(c1*x1) + c2` | yes | 3 | 8 |
| mech_compartment | `c0*exp(c1*x1) + c2*exp(c3*x1)` | yes | 4 | 13 |
| mech_compartment | `c0*x1**2*exp(c1*x1)` | no | 2 | 10 |
| mech_compartment | `c0*x1 + c1*exp(c2*x1) + c3` | yes | 4 | 12 |
| mech_compartment | `c0 + (c1*x1 + c2)*exp(c3*x1)` | no | 4 | 12 |
| mech_compartment | `c0*x1*exp(c1*x1)` | yes | 2 | 8 |
| mech_temperature | `c0*exp(c1/x1)` | yes | 2 | 6 |
| mech_temperature | `c0*exp(c3/x1)/(c1 + exp(c2/x1))` | yes | 4 | 14 |
| mech_temperature | `c0*x1*exp(c1/x1)` | no | 2 | 8 |
| temporal_novel | `c0*(c1 - x1)*(c1 + c2*x1)/x1` | no | 3 | 14 |
| temporal_novel | `c0 + x1**c1/(c2 + x1**c1)` | no | 3 | 11 |
| temporal_novel | `(c0*(c1 - x1)**2 + c2)*exp(c3*x1)` | no | 4 | 14 |
| temporal_novel | `-c0*exp(c1*exp(c2*x1)) + c0` | no | 3 | 11 |
| temporal_novel | `c0 + (c1*x1 + c2)*exp(c3*(c4 + x1))` | no | 5 | 14 |
| temporal_novel | `c0*sin(c1*x1 + c2) + c0` | no | 3 | 10 |
| temporal_novel | `(c0 + x1)/(c1 + exp(c2*(c0 + x1)))` | no | 3 | 13 |
| temporal_novel | `c0/(c0 + exp(c1/(c2*x1**c3 + c4)))` | no | 5 | 14 |
| temporal_novel | `c0/(c0 + (1/(c1 + x1))**c2)` | no | 3 | 10 |
| temporal_novel | `c0*exp(c1*x1 - c1)` | no | 2 | 8 |
| temporal_novel | `(c0 + c1*(c2 - x1)**2)/(c3 - x1)` | no | 4 | 14 |
| temporal_novel | `c0*exp(c1*(c2 + x1)**c3)` | no | 4 | 10 |

### Source weighting: equal vs pooled

P(true): mean posterior probability of the truth; log P: mean log P(true) over datasets whose true-form fit succeeded; mass: renormalised prior mass on the truth within A_5.

| prior | weighting | P(true) all | P(true) seen | P(true) novel | log P seen | log P novel | MAP seen | MAP novel | mass seen | mass novel |
|---|---|---|---|---|---|---|---|---|---|---|
| flat | - | 0.045 | 0.031 | 0.058 | -21.91 | -10.06 | 0.04 | 0.07 | 0.028 | 0.085 |
| hpy3_depth | equal | 0.080 | 0.134 | 0.029 | -20.41 | -16.56 | 0.17 | 0.03 | 0.208 | 0.055 |
| hpy3_depth | pooled | 0.072 | 0.121 | 0.024 | -21.18 | -16.33 | 0.14 | 0.03 | 0.203 | 0.049 |
| fnprior_kn_tuned | equal | 0.080 | 0.131 | 0.030 | -21.29 | -15.38 | 0.15 | 0.03 | 0.206 | 0.055 |
| fnprior_kn_tuned | pooled | 0.077 | 0.128 | 0.028 | -21.64 | -15.43 | 0.14 | 0.03 | 0.208 | 0.052 |
| fnprior_kn | equal | 0.080 | 0.131 | 0.030 | -21.30 | -15.27 | 0.15 | 0.03 | 0.206 | 0.055 |
| fnprior_kn | pooled | 0.077 | 0.128 | 0.028 | -21.64 | -15.40 | 0.14 | 0.03 | 0.208 | 0.052 |
| hpy2 | equal | 0.080 | 0.131 | 0.030 | -21.29 | -15.20 | 0.15 | 0.03 | 0.206 | 0.055 |
| hpy2 | pooled | 0.077 | 0.128 | 0.028 | -21.64 | -15.41 | 0.14 | 0.03 | 0.208 | 0.052 |
| hpy | equal | 0.080 | 0.131 | 0.030 | -21.29 | -15.18 | 0.15 | 0.03 | 0.206 | 0.055 |
| hpy | pooled | 0.077 | 0.128 | 0.027 | -21.64 | -15.33 | 0.14 | 0.03 | 0.208 | 0.052 |
| sym0 | equal | 0.079 | 0.127 | 0.032 | -24.54 | -17.57 | 0.14 | 0.04 | 0.258 | 0.082 |
| sym0 | pooled | 0.079 | 0.132 | 0.028 | -24.67 | -18.11 | 0.14 | 0.03 | 0.264 | 0.083 |
| form_ad | equal | 0.074 | 0.127 | 0.024 | -20.24 | -19.75 | 0.15 | 0.02 | 0.200 | 0.055 |
| form_ad | pooled | 0.068 | 0.118 | 0.021 | -20.85 | -17.92 | 0.14 | 0.03 | 0.193 | 0.051 |
| form_ad3_depth | equal | 0.074 | 0.127 | 0.024 | -20.24 | -19.41 | 0.15 | 0.02 | 0.200 | 0.055 |
| form_ad3_depth | pooled | 0.068 | 0.118 | 0.021 | -20.84 | -17.64 | 0.14 | 0.03 | 0.193 | 0.051 |
| tree_ad3 | equal | 0.074 | 0.127 | 0.024 | -20.24 | -19.89 | 0.15 | 0.02 | 0.200 | 0.055 |
| tree_ad3 | pooled | 0.068 | 0.118 | 0.021 | -20.83 | -18.07 | 0.14 | 0.03 | 0.193 | 0.051 |
| hpy2_src | equal | 0.071 | 0.117 | 0.026 | -22.21 | -16.42 | 0.14 | 0.02 | 0.196 | 0.055 |
| hpy2_src | pooled | 0.073 | 0.116 | 0.031 | -22.20 | -15.05 | 0.14 | 0.03 | 0.189 | 0.056 |
| uniform symbols | equal | 0.043 | 0.069 | 0.018 | -30.69 | -24.11 | 0.07 | 0.02 | 0.149 | 0.062 |
| uniform symbols | pooled | 0.043 | 0.069 | 0.018 | -30.69 | -24.11 | 0.07 | 0.02 | 0.149 | 0.062 |
| cf | equal | 0.077 | 0.120 | 0.035 | -24.60 | -19.83 | 0.13 | 0.04 | 0.240 | 0.091 |
| cf | pooled | 0.083 | 0.131 | 0.036 | -24.38 | -20.15 | 0.14 | 0.04 | 0.270 | 0.101 |
| position_backoff20 | equal | 0.072 | 0.112 | 0.034 | -23.85 | -19.35 | 0.11 | 0.03 | 0.204 | 0.074 |
| position_backoff20 | pooled | 0.078 | 0.123 | 0.035 | -23.63 | -19.73 | 0.13 | 0.03 | 0.233 | 0.084 |
| position | equal | 0.072 | 0.111 | 0.034 | -23.86 | -19.36 | 0.11 | 0.03 | 0.204 | 0.074 |
| position | pooled | 0.078 | 0.123 | 0.035 | -23.64 | -19.75 | 0.13 | 0.03 | 0.233 | 0.084 |
| parent | equal | 0.072 | 0.110 | 0.035 | -24.52 | -19.54 | 0.11 | 0.04 | 0.216 | 0.079 |
| parent | pooled | 0.077 | 0.120 | 0.036 | -24.31 | -19.90 | 0.13 | 0.03 | 0.241 | 0.088 |
| parent_backoff20 | equal | 0.072 | 0.110 | 0.035 | -24.52 | -19.55 | 0.11 | 0.04 | 0.216 | 0.079 |
| parent_backoff20 | pooled | 0.077 | 0.120 | 0.036 | -24.31 | -19.90 | 0.13 | 0.03 | 0.241 | 0.088 |
| term_root_backoff20 | equal | 0.068 | 0.105 | 0.032 | -23.53 | -19.03 | 0.11 | 0.04 | 0.195 | 0.062 |
| term_root_backoff20 | pooled | 0.075 | 0.119 | 0.032 | -23.46 | -19.57 | 0.12 | 0.03 | 0.227 | 0.078 |
| term_root | equal | 0.068 | 0.105 | 0.032 | -23.54 | -19.04 | 0.11 | 0.04 | 0.195 | 0.062 |
| term_root | pooled | 0.074 | 0.118 | 0.032 | -23.48 | -19.60 | 0.12 | 0.03 | 0.226 | 0.077 |
| term_backoff20 | equal | 0.067 | 0.104 | 0.032 | -23.78 | -19.22 | 0.11 | 0.03 | 0.188 | 0.063 |
| term_backoff20 | pooled | 0.071 | 0.111 | 0.033 | -23.48 | -19.69 | 0.11 | 0.03 | 0.209 | 0.077 |
| term | equal | 0.067 | 0.104 | 0.032 | -23.79 | -19.22 | 0.11 | 0.03 | 0.188 | 0.063 |
| term | pooled | 0.071 | 0.111 | 0.033 | -23.50 | -19.71 | 0.11 | 0.03 | 0.208 | 0.077 |
| default_prior | equal | 0.055 | 0.089 | 0.023 | -27.51 | -22.31 | 0.10 | 0.03 | 0.174 | 0.062 |
| default_prior | pooled | 0.055 | 0.089 | 0.023 | -27.51 | -22.31 | 0.10 | 0.03 | 0.174 | 0.062 |

## Design B: at most 7 constants and 25 nodes

Candidate set: the 5792 support forms of the mechanism hold-out run with at most 7 constants and 25 nodes (5733 of them in the temporal run's support, the candidate set for temporal true forms). 876 datasets: 73 true forms x 4 (n, noise) settings x 3 replicates. The true form's fit succeeded on 89.3% of datasets. Evidence: 7.6 CPU hours (GitHub Actions run 45902288688).

### True forms

| stratum | eligible distinct forms | chosen | seen | not simulable (skipped) |
|---|---|---|---|---|
| holdout_seen | 88 | 10 | 10 | 0 |
| holdout_novel | 38 | 10 | 0 | 1 |
| mech_enzyme | 15 | 15 | 8 | 0 |
| mech_compartment | 20 | 20 | 6 | 0 |
| mech_temperature | 6 | 6 | 2 | 0 |
| temporal_novel | 246 | 12 | 0 | 0 |

### How often is the truth a look-alike?

Fraction of datasets whose true form is in A_D, and the median size of A_D.

| group | datasets | truth in A_2 | median A_2 | truth in A_5 | median A_5 | truth in A_10 | median A_10 |
|---|---|---|---|---|---|---|---|
| all | 876 | 0.11 | 6 | 0.28 | 58 | 0.60 | 354 |
| holdout_seen | 120 | 0.25 | 6 | 0.49 | 62 | 0.77 | 352 |
| holdout_novel | 120 | 0.12 | 6 | 0.31 | 66 | 0.69 | 370 |
| mech_enzyme | 180 | 0.09 | 6 | 0.29 | 54 | 0.57 | 364 |
| mech_compartment | 240 | 0.08 | 6 | 0.23 | 54 | 0.65 | 334 |
| mech_temperature | 72 | 0.10 | 7 | 0.31 | 63 | 0.56 | 410 |
| temporal_novel | 144 | 0.03 | 7 | 0.16 | 59 | 0.35 | 341 |
| seen | 312 | 0.18 | 6 | 0.43 | 60 | 0.73 | 364 |
| novel | 564 | 0.06 | 6 | 0.20 | 58 | 0.52 | 344 |
| n=10, noise=0.02 | 219 | 0.10 | 6 | 0.25 | 59 | 0.65 | 387 |
| n=10, noise=0.1 | 219 | 0.03 | 9 | 0.18 | 112 | 0.51 | 785 |
| n=40, noise=0.02 | 219 | 0.25 | 5 | 0.47 | 39 | 0.71 | 198 |
| n=40, noise=0.1 | 219 | 0.05 | 6 | 0.24 | 57 | 0.53 | 382 |

### All datasets, D = 2

Truth in A_2 on 92 of 876 datasets.

| prior | family | mean rank | rank percentile | top | mass | gain | P(true) | MAP |
|---|---|---|---|---|---|---|---|---|
| flat | - | 4.36 | 0.46 | 0.08 | 0.261 | 0.00 | 0.025 | 0.04 |
| hpy3_depth | tree | 2.34 | 0.27 | 0.55 | 0.548 | 0.67 | 0.064 | 0.07 |
| tree_ad3 | tree | 2.71 | 0.31 | 0.57 | 0.525 | 0.66 | 0.063 | 0.08 |
| form_ad3_depth | tree | 2.73 | 0.31 | 0.57 | 0.525 | 0.66 | 0.063 | 0.08 |
| form_ad | tree | 2.74 | 0.32 | 0.57 | 0.525 | 0.66 | 0.063 | 0.08 |
| hpy2_src | tree | 2.28 | 0.26 | 0.58 | 0.553 | 0.36 | 0.062 | 0.07 |
| fnprior_kn_tuned | tree | 2.24 | 0.26 | 0.58 | 0.547 | 0.51 | 0.061 | 0.07 |
| fnprior_kn | tree | 2.24 | 0.26 | 0.58 | 0.547 | 0.51 | 0.061 | 0.07 |
| hpy | tree | 2.24 | 0.26 | 0.58 | 0.547 | 0.51 | 0.061 | 0.07 |
| hpy2 | tree | 2.24 | 0.26 | 0.58 | 0.547 | 0.51 | 0.061 | 0.07 |
| sym0 | tree | 2.41 | 0.30 | 0.53 | 0.525 | 0.45 | 0.057 | 0.06 |
| uniform symbols | tree | 2.89 | 0.33 | 0.35 | 0.432 | 0.00 | 0.035 | 0.04 |
| position | pcfg | 2.22 | 0.24 | 0.50 | 0.536 | 0.69 | 0.058 | 0.07 |
| position_backoff20 | pcfg | 2.25 | 0.25 | 0.50 | 0.536 | 0.69 | 0.058 | 0.06 |
| cf | pcfg | 2.31 | 0.27 | 0.46 | 0.520 | 0.28 | 0.057 | 0.06 |
| term_root | pcfg | 2.24 | 0.24 | 0.49 | 0.536 | 0.69 | 0.057 | 0.07 |
| term_root_backoff20 | pcfg | 2.28 | 0.26 | 0.49 | 0.535 | 0.69 | 0.057 | 0.06 |
| term | pcfg | 2.25 | 0.23 | 0.57 | 0.517 | 0.69 | 0.053 | 0.06 |
| term_backoff20 | pcfg | 2.27 | 0.24 | 0.48 | 0.516 | 0.69 | 0.052 | 0.06 |
| parent_backoff20 | pcfg | 2.33 | 0.26 | 0.45 | 0.510 | 0.11 | 0.052 | 0.06 |
| parent | pcfg | 2.33 | 0.26 | 0.45 | 0.510 | 0.11 | 0.052 | 0.06 |
| default_prior | pcfg | 2.38 | 0.29 | 0.45 | 0.490 | 0.34 | 0.044 | 0.05 |

### All datasets, D = 5

Truth in A_5 on 248 of 876 datasets.

| prior | family | mean rank | rank percentile | top | mass | gain | P(true) | MAP |
|---|---|---|---|---|---|---|---|---|
| flat | - | 52.12 | 0.50 | 0.00 | 0.041 | 0.00 | 0.025 | 0.04 |
| hpy3_depth | tree | 18.22 | 0.26 | 0.19 | 0.152 | -0.22 | 0.064 | 0.07 |
| tree_ad3 | tree | 24.42 | 0.34 | 0.19 | 0.144 | 0.12 | 0.063 | 0.08 |
| form_ad3_depth | tree | 24.32 | 0.34 | 0.19 | 0.144 | 0.12 | 0.063 | 0.08 |
| form_ad | tree | 25.20 | 0.34 | 0.19 | 0.144 | 0.12 | 0.063 | 0.08 |
| hpy2_src | tree | 22.68 | 0.29 | 0.16 | 0.157 | -1.55 | 0.062 | 0.07 |
| fnprior_kn_tuned | tree | 20.54 | 0.27 | 0.17 | 0.146 | -1.29 | 0.061 | 0.07 |
| fnprior_kn | tree | 20.73 | 0.27 | 0.17 | 0.146 | -1.29 | 0.061 | 0.07 |
| hpy | tree | 20.71 | 0.27 | 0.17 | 0.146 | -1.29 | 0.061 | 0.07 |
| hpy2 | tree | 20.77 | 0.27 | 0.17 | 0.146 | -1.29 | 0.061 | 0.07 |
| sym0 | tree | 40.19 | 0.35 | 0.19 | 0.178 | -3.33 | 0.057 | 0.06 |
| uniform symbols | tree | 57.73 | 0.46 | 0.06 | 0.101 | -5.93 | 0.035 | 0.04 |
| position | pcfg | 37.38 | 0.33 | 0.16 | 0.154 | -2.04 | 0.058 | 0.07 |
| position_backoff20 | pcfg | 37.24 | 0.33 | 0.16 | 0.154 | -2.04 | 0.058 | 0.06 |
| cf | pcfg | 40.17 | 0.33 | 0.14 | 0.164 | -2.56 | 0.057 | 0.06 |
| term_root | pcfg | 37.15 | 0.34 | 0.14 | 0.146 | -1.95 | 0.057 | 0.07 |
| term_root_backoff20 | pcfg | 36.66 | 0.33 | 0.14 | 0.146 | -1.95 | 0.057 | 0.06 |
| term | pcfg | 37.51 | 0.33 | 0.15 | 0.132 | -2.08 | 0.053 | 0.06 |
| term_backoff20 | pcfg | 37.01 | 0.33 | 0.13 | 0.132 | -2.08 | 0.052 | 0.06 |
| parent_backoff20 | pcfg | 39.30 | 0.34 | 0.12 | 0.142 | -2.72 | 0.052 | 0.06 |
| parent | pcfg | 39.40 | 0.34 | 0.12 | 0.142 | -2.72 | 0.052 | 0.06 |
| default_prior | pcfg | 52.08 | 0.40 | 0.11 | 0.123 | -2.64 | 0.044 | 0.05 |

### All datasets, D = 10

Truth in A_10 on 525 of 876 datasets.

| prior | family | mean rank | rank percentile | top | mass | gain | P(true) | MAP |
|---|---|---|---|---|---|---|---|---|
| flat | - | 274.54 | 0.50 | 0.00 | 0.007 | 0.00 | 0.025 | 0.04 |
| hpy3_depth | tree | 132.71 | 0.31 | 0.04 | 0.025 | -2.23 | 0.064 | 0.07 |
| tree_ad3 | tree | 176.03 | 0.41 | 0.06 | 0.022 | -4.87 | 0.063 | 0.08 |
| form_ad3_depth | tree | 178.71 | 0.42 | 0.06 | 0.022 | -4.85 | 0.063 | 0.08 |
| form_ad | tree | 190.78 | 0.43 | 0.06 | 0.022 | -4.87 | 0.063 | 0.08 |
| hpy2_src | tree | 167.89 | 0.36 | 0.04 | 0.037 | -4.89 | 0.062 | 0.07 |
| fnprior_kn_tuned | tree | 141.32 | 0.31 | 0.04 | 0.032 | -3.69 | 0.061 | 0.07 |
| fnprior_kn | tree | 143.35 | 0.31 | 0.04 | 0.032 | -3.69 | 0.061 | 0.07 |
| hpy | tree | 143.38 | 0.31 | 0.04 | 0.032 | -3.70 | 0.061 | 0.07 |
| hpy2 | tree | 143.07 | 0.31 | 0.04 | 0.032 | -3.69 | 0.061 | 0.07 |
| sym0 | tree | 276.76 | 0.45 | 0.04 | 0.047 | -9.09 | 0.057 | 0.06 |
| uniform symbols | tree | 360.27 | 0.55 | 0.02 | 0.028 | -16.56 | 0.035 | 0.04 |
| position | pcfg | 256.65 | 0.41 | 0.05 | 0.040 | -7.18 | 0.058 | 0.07 |
| position_backoff20 | pcfg | 251.34 | 0.40 | 0.05 | 0.040 | -7.18 | 0.058 | 0.06 |
| cf | pcfg | 275.82 | 0.42 | 0.04 | 0.046 | -7.66 | 0.057 | 0.06 |
| term_root | pcfg | 252.87 | 0.42 | 0.03 | 0.035 | -6.62 | 0.057 | 0.07 |
| term_root_backoff20 | pcfg | 243.25 | 0.41 | 0.03 | 0.035 | -6.62 | 0.057 | 0.06 |
| term | pcfg | 254.28 | 0.41 | 0.03 | 0.030 | -7.05 | 0.053 | 0.06 |
| term_backoff20 | pcfg | 245.16 | 0.40 | 0.03 | 0.030 | -7.05 | 0.052 | 0.06 |
| parent_backoff20 | pcfg | 265.07 | 0.42 | 0.03 | 0.037 | -7.74 | 0.052 | 0.06 |
| parent | pcfg | 267.60 | 0.42 | 0.03 | 0.037 | -7.74 | 0.052 | 0.06 |
| default_prior | pcfg | 336.58 | 0.50 | 0.04 | 0.036 | -10.61 | 0.044 | 0.05 |

### Renormalised prior mass on the truth within A_5, by stratum

| prior | holdout_seen | holdout_novel | mech_enzyme | mech_compartment | mech_temperature | temporal_novel | seen | novel |
|---|---|---|---|---|---|---|---|---|
| datasets | 120 | 120 | 180 | 240 | 72 | 144 | 312 | 564 |
| flat | 0.043 | 0.053 | 0.044 | 0.039 | 0.028 | 0.030 | 0.035 | 0.049 |
| hpy3_depth | 0.337 | 0.000 | 0.133 | 0.152 | 0.098 | 0.013 | 0.254 | 0.030 |
| tree_ad3 | 0.357 | 0.000 | 0.130 | 0.118 | 0.040 | 0.022 | 0.243 | 0.025 |
| form_ad3_depth | 0.357 | 0.000 | 0.130 | 0.118 | 0.040 | 0.022 | 0.243 | 0.025 |
| form_ad | 0.357 | 0.000 | 0.130 | 0.118 | 0.039 | 0.022 | 0.243 | 0.025 |
| hpy2_src | 0.393 | 0.013 | 0.167 | 0.104 | 0.007 | 0.024 | 0.270 | 0.021 |
| fnprior_kn_tuned | 0.351 | 0.006 | 0.117 | 0.161 | 0.015 | 0.005 | 0.253 | 0.019 |
| fnprior_kn | 0.351 | 0.006 | 0.117 | 0.161 | 0.015 | 0.005 | 0.253 | 0.019 |
| hpy | 0.351 | 0.006 | 0.117 | 0.161 | 0.015 | 0.005 | 0.253 | 0.019 |
| hpy2 | 0.351 | 0.006 | 0.117 | 0.161 | 0.015 | 0.005 | 0.253 | 0.019 |
| sym0 | 0.376 | 0.007 | 0.091 | 0.236 | 0.184 | 0.000 | 0.313 | 0.016 |
| uniform symbols | 0.279 | 0.049 | 0.039 | 0.068 | 0.046 | 0.000 | 0.166 | 0.023 |
| position | 0.309 | 0.057 | 0.073 | 0.214 | 0.107 | 0.000 | 0.259 | 0.028 |
| position_backoff20 | 0.309 | 0.057 | 0.073 | 0.214 | 0.107 | 0.000 | 0.258 | 0.028 |
| cf | 0.348 | 0.059 | 0.051 | 0.217 | 0.168 | 0.000 | 0.274 | 0.034 |
| term_root | 0.307 | 0.057 | 0.083 | 0.181 | 0.082 | 0.000 | 0.246 | 0.025 |
| term_root_backoff20 | 0.307 | 0.057 | 0.082 | 0.181 | 0.082 | 0.000 | 0.246 | 0.025 |
| term | 0.264 | 0.055 | 0.055 | 0.191 | 0.090 | 0.000 | 0.223 | 0.024 |
| term_backoff20 | 0.264 | 0.055 | 0.054 | 0.191 | 0.090 | 0.000 | 0.223 | 0.024 |
| parent_backoff20 | 0.280 | 0.057 | 0.037 | 0.217 | 0.139 | 0.000 | 0.238 | 0.028 |
| parent | 0.280 | 0.057 | 0.037 | 0.217 | 0.139 | 0.000 | 0.238 | 0.028 |
| default_prior | 0.287 | 0.120 | 0.024 | 0.104 | 0.096 | 0.000 | 0.186 | 0.047 |

### Mean rank of the truth within A_5, by stratum

| prior | holdout_seen | holdout_novel | mech_enzyme | mech_compartment | mech_temperature | temporal_novel | seen | novel |
|---|---|---|---|---|---|---|---|---|
| datasets | 120 | 120 | 180 | 240 | 72 | 144 | 312 | 564 |
| flat | 47.98 | 54.65 | 57.48 | 35.26 | 51.23 | 86.72 | 54.31 | 49.49 |
| hpy3_depth | 2.73 | 35.05 | 18.00 | 16.94 | 11.20 | 41.15 | 6.79 | 31.88 |
| tree_ad3 | 2.49 | 57.81 | 21.09 | 22.42 | 17.16 | 46.24 | 6.49 | 45.83 |
| form_ad3_depth | 2.49 | 56.97 | 21.53 | 22.51 | 16.70 | 45.80 | 6.49 | 45.64 |
| form_ad | 2.49 | 61.51 | 22.51 | 21.79 | 18.39 | 45.76 | 6.49 | 47.55 |
| hpy2_src | 3.03 | 40.43 | 19.32 | 23.06 | 21.48 | 52.50 | 9.14 | 38.85 |
| fnprior_kn_tuned | 2.85 | 37.11 | 20.77 | 17.18 | 16.34 | 50.67 | 8.36 | 35.10 |
| fnprior_kn | 2.85 | 37.97 | 20.77 | 17.21 | 16.57 | 50.98 | 8.40 | 35.46 |
| hpy | 2.85 | 37.86 | 20.83 | 17.21 | 16.52 | 50.85 | 8.39 | 35.42 |
| hpy2 | 2.85 | 38.14 | 20.81 | 17.23 | 16.52 | 51.11 | 8.39 | 35.57 |
| sym0 | 3.24 | 64.49 | 61.34 | 19.08 | 10.89 | 124.76 | 21.75 | 62.22 |
| uniform symbols | 11.11 | 83.84 | 82.10 | 31.14 | 34.14 | 164.20 | 38.23 | 81.04 |
| position | 2.97 | 59.03 | 34.38 | 18.05 | 12.48 | 166.93 | 11.56 | 68.22 |
| position_backoff20 | 3.07 | 58.43 | 34.50 | 17.95 | 12.20 | 166.37 | 11.65 | 67.82 |
| cf | 3.49 | 60.53 | 46.23 | 18.29 | 10.89 | 166.91 | 16.61 | 68.31 |
| term_root | 2.88 | 57.46 | 31.32 | 20.91 | 12.66 | 167.35 | 10.56 | 68.92 |
| term_root_backoff20 | 2.98 | 55.57 | 31.26 | 20.88 | 12.52 | 165.17 | 10.62 | 67.76 |
| term | 3.03 | 58.30 | 33.45 | 19.32 | 13.89 | 167.13 | 11.41 | 68.69 |
| term_backoff20 | 3.14 | 56.19 | 33.25 | 19.19 | 13.25 | 166.35 | 11.30 | 67.74 |
| parent_backoff20 | 4.07 | 57.89 | 43.68 | 17.42 | 11.66 | 167.54 | 15.47 | 67.78 |
| parent | 4.07 | 58.41 | 43.39 | 17.47 | 12.20 | 167.72 | 15.44 | 68.01 |
| default_prior | 5.47 | 76.73 | 77.08 | 26.19 | 18.23 | 167.54 | 31.84 | 76.27 |

### MAP accuracy (posterior mode = truth), by stratum

| prior | holdout_seen | holdout_novel | mech_enzyme | mech_compartment | mech_temperature | temporal_novel | seen | novel |
|---|---|---|---|---|---|---|---|---|
| datasets | 120 | 120 | 180 | 240 | 72 | 144 | 312 | 564 |
| flat | 0.07 | 0.03 | 0.04 | 0.03 | 0.03 | 0.00 | 0.06 | 0.02 |
| hpy3_depth | 0.30 | 0.00 | 0.04 | 0.07 | 0.04 | 0.00 | 0.19 | 0.01 |
| tree_ad3 | 0.32 | 0.00 | 0.05 | 0.07 | 0.04 | 0.00 | 0.20 | 0.01 |
| form_ad3_depth | 0.32 | 0.00 | 0.05 | 0.07 | 0.04 | 0.00 | 0.20 | 0.01 |
| form_ad | 0.32 | 0.00 | 0.05 | 0.07 | 0.04 | 0.00 | 0.20 | 0.01 |
| hpy2_src | 0.32 | 0.00 | 0.06 | 0.06 | 0.03 | 0.00 | 0.19 | 0.01 |
| fnprior_kn_tuned | 0.32 | 0.00 | 0.05 | 0.07 | 0.03 | 0.00 | 0.20 | 0.01 |
| fnprior_kn | 0.32 | 0.00 | 0.05 | 0.07 | 0.03 | 0.00 | 0.20 | 0.01 |
| hpy | 0.32 | 0.00 | 0.05 | 0.07 | 0.03 | 0.00 | 0.20 | 0.01 |
| hpy2 | 0.32 | 0.00 | 0.05 | 0.07 | 0.03 | 0.00 | 0.20 | 0.01 |
| sym0 | 0.25 | 0.00 | 0.03 | 0.06 | 0.06 | 0.00 | 0.16 | 0.01 |
| uniform symbols | 0.19 | 0.01 | 0.02 | 0.03 | 0.01 | 0.00 | 0.10 | 0.01 |
| position | 0.25 | 0.02 | 0.03 | 0.07 | 0.04 | 0.00 | 0.17 | 0.01 |
| position_backoff20 | 0.22 | 0.02 | 0.03 | 0.07 | 0.04 | 0.00 | 0.15 | 0.01 |
| cf | 0.20 | 0.02 | 0.02 | 0.07 | 0.04 | 0.00 | 0.14 | 0.01 |
| term_root | 0.23 | 0.02 | 0.04 | 0.07 | 0.03 | 0.00 | 0.16 | 0.01 |
| term_root_backoff20 | 0.20 | 0.02 | 0.04 | 0.07 | 0.03 | 0.00 | 0.15 | 0.01 |
| term | 0.23 | 0.02 | 0.03 | 0.07 | 0.04 | 0.00 | 0.16 | 0.01 |
| term_backoff20 | 0.23 | 0.02 | 0.03 | 0.07 | 0.04 | 0.00 | 0.16 | 0.01 |
| parent_backoff20 | 0.23 | 0.02 | 0.02 | 0.07 | 0.04 | 0.00 | 0.14 | 0.01 |
| parent | 0.23 | 0.02 | 0.02 | 0.07 | 0.04 | 0.00 | 0.14 | 0.01 |
| default_prior | 0.17 | 0.05 | 0.02 | 0.04 | 0.03 | 0.00 | 0.10 | 0.02 |

### Posterior probability of the truth, by stratum

| prior | holdout_seen | holdout_novel | mech_enzyme | mech_compartment | mech_temperature | temporal_novel | seen | novel |
|---|---|---|---|---|---|---|---|---|
| datasets | 120 | 120 | 180 | 240 | 72 | 144 | 312 | 564 |
| flat | 0.060 | 0.023 | 0.028 | 0.023 | 0.018 | 0.004 | 0.041 | 0.017 |
| hpy3_depth | 0.259 | 0.000 | 0.047 | 0.056 | 0.040 | 0.002 | 0.167 | 0.007 |
| tree_ad3 | 0.268 | 0.000 | 0.046 | 0.050 | 0.031 | 0.004 | 0.161 | 0.009 |
| form_ad3_depth | 0.268 | 0.000 | 0.046 | 0.050 | 0.031 | 0.004 | 0.161 | 0.009 |
| form_ad | 0.268 | 0.000 | 0.046 | 0.050 | 0.031 | 0.004 | 0.161 | 0.009 |
| hpy2_src | 0.268 | 0.005 | 0.052 | 0.045 | 0.016 | 0.004 | 0.162 | 0.007 |
| fnprior_kn_tuned | 0.250 | 0.002 | 0.042 | 0.058 | 0.020 | 0.001 | 0.160 | 0.007 |
| fnprior_kn | 0.250 | 0.002 | 0.042 | 0.058 | 0.020 | 0.001 | 0.160 | 0.007 |
| hpy | 0.250 | 0.002 | 0.042 | 0.058 | 0.020 | 0.001 | 0.160 | 0.007 |
| hpy2 | 0.250 | 0.002 | 0.042 | 0.058 | 0.020 | 0.001 | 0.160 | 0.007 |
| sym0 | 0.224 | 0.002 | 0.027 | 0.061 | 0.052 | 0.000 | 0.151 | 0.006 |
| uniform symbols | 0.171 | 0.014 | 0.013 | 0.021 | 0.016 | 0.000 | 0.088 | 0.006 |
| position | 0.212 | 0.018 | 0.028 | 0.062 | 0.041 | 0.000 | 0.144 | 0.010 |
| position_backoff20 | 0.212 | 0.018 | 0.028 | 0.062 | 0.041 | 0.000 | 0.144 | 0.010 |
| cf | 0.210 | 0.018 | 0.018 | 0.064 | 0.050 | 0.000 | 0.139 | 0.011 |
| term_root | 0.211 | 0.018 | 0.034 | 0.055 | 0.039 | 0.000 | 0.142 | 0.010 |
| term_root_backoff20 | 0.211 | 0.018 | 0.034 | 0.055 | 0.039 | 0.000 | 0.142 | 0.010 |
| term | 0.192 | 0.018 | 0.023 | 0.058 | 0.039 | 0.000 | 0.131 | 0.009 |
| term_backoff20 | 0.192 | 0.018 | 0.023 | 0.058 | 0.039 | 0.000 | 0.130 | 0.009 |
| parent_backoff20 | 0.185 | 0.018 | 0.015 | 0.064 | 0.046 | 0.000 | 0.128 | 0.010 |
| parent | 0.185 | 0.018 | 0.015 | 0.064 | 0.046 | 0.000 | 0.128 | 0.010 |
| default_prior | 0.179 | 0.038 | 0.014 | 0.033 | 0.029 | 0.000 | 0.100 | 0.013 |

### Renormalised prior mass on the truth within A_5, by (n, noise)

| prior | n=10, noise=0.02 | n=10, noise=0.1 | n=40, noise=0.02 | n=40, noise=0.1 |
|---|---|---|---|---|
| datasets | 219 | 219 | 219 | 219 |
| flat | 0.033 | 0.012 | 0.060 | 0.034 |
| hpy3_depth | 0.180 | 0.123 | 0.158 | 0.130 |
| tree_ad3 | 0.185 | 0.127 | 0.137 | 0.129 |
| form_ad3_depth | 0.184 | 0.127 | 0.136 | 0.129 |
| form_ad | 0.184 | 0.127 | 0.136 | 0.129 |
| hpy2_src | 0.186 | 0.149 | 0.143 | 0.159 |
| fnprior_kn_tuned | 0.189 | 0.120 | 0.148 | 0.117 |
| fnprior_kn | 0.189 | 0.120 | 0.148 | 0.117 |
| hpy | 0.189 | 0.120 | 0.148 | 0.117 |
| hpy2 | 0.189 | 0.120 | 0.148 | 0.117 |
| sym0 | 0.238 | 0.093 | 0.193 | 0.148 |
| uniform symbols | 0.135 | 0.023 | 0.131 | 0.063 |
| position | 0.183 | 0.070 | 0.181 | 0.131 |
| position_backoff20 | 0.183 | 0.070 | 0.181 | 0.131 |
| cf | 0.209 | 0.077 | 0.188 | 0.137 |
| term_root | 0.166 | 0.073 | 0.169 | 0.133 |
| term_root_backoff20 | 0.166 | 0.073 | 0.169 | 0.133 |
| term | 0.157 | 0.053 | 0.158 | 0.114 |
| term_backoff20 | 0.157 | 0.053 | 0.157 | 0.115 |
| parent_backoff20 | 0.181 | 0.055 | 0.168 | 0.117 |
| parent | 0.181 | 0.055 | 0.168 | 0.117 |
| default_prior | 0.138 | 0.042 | 0.165 | 0.083 |

### MAP accuracy, by (n, noise)

| prior | n=10, noise=0.02 | n=10, noise=0.1 | n=40, noise=0.02 | n=40, noise=0.1 |
|---|---|---|---|---|
| datasets | 219 | 219 | 219 | 219 |
| flat | 0.02 | 0.01 | 0.11 | 0.00 |
| hpy3_depth | 0.07 | 0.04 | 0.14 | 0.05 |
| tree_ad3 | 0.08 | 0.04 | 0.13 | 0.06 |
| form_ad3_depth | 0.08 | 0.04 | 0.13 | 0.06 |
| form_ad | 0.08 | 0.04 | 0.13 | 0.06 |
| hpy2_src | 0.07 | 0.04 | 0.13 | 0.06 |
| fnprior_kn_tuned | 0.08 | 0.04 | 0.13 | 0.05 |
| fnprior_kn | 0.08 | 0.04 | 0.13 | 0.05 |
| hpy | 0.08 | 0.04 | 0.13 | 0.05 |
| hpy2 | 0.08 | 0.04 | 0.13 | 0.05 |
| sym0 | 0.08 | 0.01 | 0.11 | 0.04 |
| uniform symbols | 0.05 | 0.00 | 0.09 | 0.02 |
| position | 0.08 | 0.02 | 0.13 | 0.04 |
| position_backoff20 | 0.07 | 0.02 | 0.12 | 0.04 |
| cf | 0.06 | 0.02 | 0.12 | 0.03 |
| term_root | 0.07 | 0.02 | 0.13 | 0.04 |
| term_root_backoff20 | 0.06 | 0.02 | 0.13 | 0.04 |
| term | 0.07 | 0.02 | 0.13 | 0.03 |
| term_backoff20 | 0.07 | 0.02 | 0.13 | 0.03 |
| parent_backoff20 | 0.07 | 0.01 | 0.11 | 0.04 |
| parent | 0.07 | 0.01 | 0.11 | 0.04 |
| default_prior | 0.04 | 0.00 | 0.11 | 0.03 |

### True forms used

| stratum | form | seen | constants | nodes |
|---|---|---|---|---|
| holdout_seen | `c0*x1` | yes | 1 | 3 |
| holdout_seen | `c0*x1**2` | yes | 1 | 5 |
| holdout_seen | `c0*x1/(c1 + x1)` | yes | 2 | 7 |
| holdout_seen | `c0*(c1*x1 + c2)**c3` | yes | 4 | 9 |
| holdout_seen | `x1/(c0 + x1)` | yes | 1 | 5 |
| holdout_seen | `-c0*x1 + c0` | yes | 1 | 5 |
| holdout_seen | `c0*x1**2/(c1*x1**2 + c2 + x1)` | yes | 3 | 15 |
| holdout_seen | `c0*x1/(c1*x1**2 + c2 + x1)` | yes | 3 | 13 |
| holdout_seen | `c0*exp(c1*x1**c2) + c3` | yes | 4 | 10 |
| holdout_seen | `c0*(c1 - x1)*(c2 + x1)**c3` | yes | 4 | 11 |
| holdout_novel | `-c0*sin(c1*x1 + c2) + c0` | no | 3 | 10 |
| holdout_novel | `log(c0*exp(c1*x1) + c2*exp(c3*x1))` | no | 4 | 14 |
| holdout_novel | `c0/sqrt(c1 + x1) + c2*sqrt(c3 + x1) + c4` | no | 5 | 16 |
| holdout_novel | `c0*exp(c3*c4/x1 + c4)/(c1 - exp(c2)*exp(c2*c3/x1))` | no | 5 | 23 |
| holdout_novel | `c0*exp(c1*x1**2 + c2*x1)` | no | 3 | 12 |
| holdout_novel | `c0 - (c1 - exp(c2*x1))**c3` | no | 4 | 10 |
| holdout_novel | `c0*tanh(c1*x1) + c2` | no | 3 | 8 |
| holdout_novel | `c0*(c1 + c2*x1)*(c3 + x1)/(c4 + x1)` | no | 5 | 16 |
| holdout_novel | `(c0**x1*c1 + c2)/(c0**x1*c3 + c4 + c1/c0**x1)` | no | 5 | 24 |
| holdout_novel | `c0*x1*(c1 + x1**c2)**c3` | no | 4 | 11 |
| mech_enzyme | `(c0*x1**3 + c1)/(c2*x1**2 + c3*x1 + c4 + x1**3)` | no | 5 | 24 |
| mech_enzyme | `c0*x1**3/(c1*x1**2 + c2*x1 + c3 + x1**3)` | no | 4 | 21 |
| mech_enzyme | `c0*x1**2/(c1*x1**2 + c2*x1 + c3 + x1**3)` | no | 4 | 21 |
| mech_enzyme | `(c0*x1**2 + c1)/(c2*x1 + c3 + x1**2)` | yes | 4 | 18 |
| mech_enzyme | `c0*x1**2/(c1*x1 + c2 + x1**2)` | yes | 3 | 15 |
| mech_enzyme | `c0*x1/(c1 + x1)` | yes | 2 | 7 |
| mech_enzyme | `(c0*x1 + c1)/(c2*x1 + c3 + x1**2)` | yes | 4 | 16 |
| mech_enzyme | `c0*x1/(c1*x1**2 + c2*x1 + c3 + x1**3)` | no | 4 | 19 |
| mech_enzyme | `(c0*x1 + c1)/(c2*x1**2 + c3*x1 + c4 + x1**3)` | no | 5 | 22 |
| mech_enzyme | `c0/(c1 + x1)` | yes | 2 | 5 |
| mech_enzyme | `(c0*x1**2 + c1)/(c2*x1**2 + c3*x1 + c4 + x1**3)` | no | 5 | 24 |
| mech_enzyme | `c0*x1/(c1*x1 + c2 + x1**2)` | yes | 3 | 13 |
| mech_enzyme | `c0/(c1*x1**2 + c2*x1 + c3 + x1**3)` | no | 4 | 17 |
| mech_enzyme | `c0/(c1*x1 + c2 + x1**2)` | yes | 3 | 11 |
| mech_enzyme | `(c0 + c1*x1)/(c2 + x1)` | yes | 3 | 10 |
| mech_compartment | `c0 + c1*exp(c2*x1) + c3*exp(c4*x1) + c5*exp(c6*x1)` | no | 7 | 22 |
| mech_compartment | `c0*exp(c1*x1) + c5*exp(c6*x1) + (c2 + c3*x1)*exp(c4*x1)` | no | 7 | 24 |
| mech_compartment | `c0 + c1*x1 + (c2 + c3*x1 + c4*x1**2)*exp(c5*x1)` | no | 6 | 22 |
| mech_compartment | `c0 + (c1*x1 + c2)*exp(c3*x1)` | no | 4 | 12 |
| mech_compartment | `c0*exp(c1*x1) + c2*exp(c3*x1) + c4*exp(c5*x1)` | no | 6 | 20 |
| mech_compartment | `c0*x1 + c1 + (c2*x1 + c3)*exp(c4*x1)` | no | 5 | 16 |
| mech_compartment | `c0*exp(c1*x1) + c2` | yes | 3 | 8 |
| mech_compartment | `c0*x1 + c1*exp(c2*x1) + c3` | yes | 4 | 12 |
| mech_compartment | `c0*exp(c1*x1)` | yes | 2 | 6 |
| mech_compartment | `c0*x1*exp(c1*x1)` | yes | 2 | 8 |
| mech_compartment | `c0*x1**2*exp(c1*x1)` | no | 2 | 10 |
| mech_compartment | `c0*exp(c1*x1) + c2*exp(c3*x1)` | yes | 4 | 13 |
| mech_compartment | `c0 + (c1*x1**2 + c2*x1 + c3)*exp(c4*x1)` | no | 5 | 18 |
| mech_compartment | `c0*exp(c1*x1) + (c2*x1 + c3)*exp(c4*x1)` | no | 5 | 17 |
| mech_compartment | `c0*exp(c1*x1) + c2*exp(c3*x1) + c4` | yes | 5 | 15 |
| mech_compartment | `c0*exp(c1*x1) + (c2 + c3*x1 + c4*x1**2)*exp(c5*x1)` | no | 6 | 23 |
| mech_compartment | `c3*exp(c4*x1) + c5*exp(c6*x1) + (c0 + c1*x1)*exp(c2*x1)` | no | 7 | 24 |
| mech_compartment | `c0 + c1*x1 + c2*exp(c3*x1) + c4*exp(c5*x1)` | no | 6 | 19 |
| mech_compartment | `c0 + c4*exp(c5*x1) + (c1 + c2*x1)*exp(c3*x1)` | no | 6 | 19 |
| mech_compartment | `c0 + c1*exp(c2*x1) + (c3 + c4*x1)*exp(c5*x1)` | no | 6 | 19 |
| mech_temperature | `c2*x1*exp(c0/x1)/(c1*exp(c5/x1) + c3 + exp(c4/x1))` | no | 6 | 23 |
| mech_temperature | `c0*exp(c1/x1)` | yes | 2 | 6 |
| mech_temperature | `c0*exp(c3/x1)/(c1 + exp(c2/x1))` | yes | 4 | 14 |
| mech_temperature | `c2*exp(c0/x1)/(c1*exp(c5/x1) + c3 + exp(c4/x1))` | no | 6 | 21 |
| mech_temperature | `c0*x1*exp(c1/x1)` | no | 2 | 8 |
| mech_temperature | `c0*x1*exp(c3/x1)/(c1 + exp(c2/x1))` | no | 4 | 16 |
| temporal_novel | `(c0 + c1*x1)*(c2 + x1)/(c3*(c2 + x1) + x1)` | no | 4 | 18 |
| temporal_novel | `c2/(c4 + exp(c0*x1)) + c3/(c5 + exp(c1*x1))` | no | 6 | 17 |
| temporal_novel | `c0*(c1*(c2 - x1)**2 + c3)/(c0 + c2 - x1)` | no | 4 | 18 |
| temporal_novel | `c0/(c1 + exp(c2*exp(c4*x1) + c3*exp(c5*x1)))` | no | 6 | 18 |
| temporal_novel | `(c0 + c1*x1)*(1/(c3 + x1))**c4/(c2 + x1)` | no | 5 | 17 |
| temporal_novel | `c1/(c2 + c3*exp(c5*x1) + c4*exp(c6*x1) + exp(c0*x1))` | no | 7 | 22 |
| temporal_novel | `c0*(c1 - x1)/(c2 + exp(c3*x1)) + c4*x1` | no | 5 | 17 |
| temporal_novel | `(c0 + c1*x1**8)/(c2 + x1**8)` | no | 3 | 14 |
| temporal_novel | `c0*((c1 - x1)/(c2 - x1))**c3/(c4 + ((c1 - x1)/(c2 - x1))**c3)` | no | 5 | 25 |
| temporal_novel | `c0/(c1 + exp(c5*x1)) + c2*exp(c3*x1) + c4` | no | 6 | 17 |
| temporal_novel | `c0*exp(c2/(c1 + x1))/(c1 + x1)**(3/2)` | no | 3 | 14 |
| temporal_novel | `c2/(c3 + ((c1*x1 + c5)/(c0*x1 + c4))**c6)` | no | 7 | 18 |

### Source weighting: equal vs pooled

P(true): mean posterior probability of the truth; log P: mean log P(true) over datasets whose true-form fit succeeded; mass: renormalised prior mass on the truth within A_5.

| prior | weighting | P(true) all | P(true) seen | P(true) novel | log P seen | log P novel | MAP seen | MAP novel | mass seen | mass novel |
|---|---|---|---|---|---|---|---|---|---|---|
| flat | - | 0.025 | 0.041 | 0.017 | -14.01 | -17.78 | 0.06 | 0.02 | 0.035 | 0.049 |
| hpy3_depth | equal | 0.064 | 0.167 | 0.007 | -12.30 | -26.96 | 0.19 | 0.01 | 0.254 | 0.030 |
| hpy3_depth | pooled | 0.058 | 0.156 | 0.004 | -13.22 | -27.38 | 0.19 | 0.00 | 0.263 | 0.009 |
| tree_ad3 | equal | 0.063 | 0.161 | 0.009 | -12.32 | -29.77 | 0.20 | 0.01 | 0.243 | 0.025 |
| tree_ad3 | pooled | 0.058 | 0.155 | 0.005 | -13.02 | -27.81 | 0.19 | 0.01 | 0.252 | 0.018 |
| form_ad3_depth | equal | 0.063 | 0.161 | 0.009 | -12.32 | -30.44 | 0.20 | 0.01 | 0.243 | 0.025 |
| form_ad3_depth | pooled | 0.058 | 0.155 | 0.005 | -13.03 | -28.46 | 0.19 | 0.00 | 0.252 | 0.018 |
| form_ad | equal | 0.063 | 0.161 | 0.009 | -12.33 | -31.59 | 0.20 | 0.01 | 0.243 | 0.025 |
| form_ad | pooled | 0.058 | 0.155 | 0.004 | -13.05 | -29.62 | 0.19 | 0.00 | 0.252 | 0.015 |
| hpy2_src | equal | 0.062 | 0.162 | 0.007 | -14.05 | -27.89 | 0.19 | 0.01 | 0.270 | 0.021 |
| hpy2_src | pooled | 0.063 | 0.165 | 0.007 | -13.89 | -26.99 | 0.20 | 0.01 | 0.269 | 0.020 |
| fnprior_kn_tuned | equal | 0.061 | 0.160 | 0.007 | -13.35 | -26.62 | 0.20 | 0.01 | 0.253 | 0.019 |
| fnprior_kn_tuned | pooled | 0.058 | 0.155 | 0.005 | -13.76 | -26.88 | 0.18 | 0.00 | 0.270 | 0.013 |
| fnprior_kn | equal | 0.061 | 0.160 | 0.007 | -13.35 | -26.56 | 0.20 | 0.01 | 0.253 | 0.019 |
| fnprior_kn | pooled | 0.058 | 0.155 | 0.005 | -13.76 | -26.84 | 0.18 | 0.00 | 0.270 | 0.013 |
| hpy | equal | 0.061 | 0.160 | 0.007 | -13.35 | -26.53 | 0.20 | 0.01 | 0.253 | 0.019 |
| hpy | pooled | 0.058 | 0.155 | 0.005 | -13.75 | -26.82 | 0.18 | 0.00 | 0.270 | 0.013 |
| hpy2 | equal | 0.061 | 0.160 | 0.007 | -13.35 | -26.52 | 0.20 | 0.01 | 0.253 | 0.019 |
| hpy2 | pooled | 0.058 | 0.155 | 0.005 | -13.75 | -26.83 | 0.18 | 0.00 | 0.270 | 0.013 |
| sym0 | equal | 0.057 | 0.151 | 0.006 | -17.22 | -33.58 | 0.16 | 0.01 | 0.313 | 0.016 |
| sym0 | pooled | 0.059 | 0.156 | 0.005 | -17.41 | -33.73 | 0.16 | 0.01 | 0.329 | 0.016 |
| uniform symbols | equal | 0.035 | 0.088 | 0.006 | -22.87 | -45.98 | 0.10 | 0.01 | 0.166 | 0.023 |
| uniform symbols | pooled | 0.035 | 0.088 | 0.006 | -22.87 | -45.98 | 0.10 | 0.01 | 0.166 | 0.023 |
| position | equal | 0.058 | 0.144 | 0.010 | -15.48 | -33.99 | 0.17 | 0.01 | 0.259 | 0.028 |
| position | pooled | 0.063 | 0.157 | 0.010 | -15.43 | -33.93 | 0.17 | 0.01 | 0.307 | 0.032 |
| position_backoff20 | equal | 0.058 | 0.144 | 0.010 | -15.48 | -34.00 | 0.15 | 0.01 | 0.258 | 0.028 |
| position_backoff20 | pooled | 0.063 | 0.157 | 0.010 | -15.41 | -33.93 | 0.18 | 0.01 | 0.307 | 0.032 |
| cf | equal | 0.057 | 0.139 | 0.011 | -16.28 | -35.27 | 0.14 | 0.01 | 0.274 | 0.034 |
| cf | pooled | 0.061 | 0.149 | 0.012 | -16.11 | -34.97 | 0.16 | 0.01 | 0.315 | 0.038 |
| term_root | equal | 0.057 | 0.142 | 0.010 | -15.07 | -33.85 | 0.16 | 0.01 | 0.246 | 0.025 |
| term_root | pooled | 0.063 | 0.158 | 0.010 | -15.15 | -34.40 | 0.18 | 0.01 | 0.309 | 0.032 |
| term_root_backoff20 | equal | 0.057 | 0.142 | 0.010 | -15.06 | -33.86 | 0.15 | 0.01 | 0.246 | 0.025 |
| term_root_backoff20 | pooled | 0.063 | 0.159 | 0.010 | -15.13 | -34.40 | 0.18 | 0.01 | 0.309 | 0.032 |
| term | equal | 0.053 | 0.131 | 0.009 | -15.44 | -34.03 | 0.16 | 0.01 | 0.223 | 0.024 |
| term | pooled | 0.060 | 0.150 | 0.010 | -15.30 | -34.01 | 0.16 | 0.01 | 0.288 | 0.029 |
| term_backoff20 | equal | 0.052 | 0.130 | 0.009 | -15.43 | -34.04 | 0.16 | 0.01 | 0.223 | 0.024 |
| term_backoff20 | pooled | 0.060 | 0.150 | 0.010 | -15.29 | -34.01 | 0.16 | 0.01 | 0.288 | 0.029 |
| parent_backoff20 | equal | 0.052 | 0.128 | 0.010 | -16.30 | -34.99 | 0.14 | 0.01 | 0.238 | 0.028 |
| parent_backoff20 | pooled | 0.057 | 0.140 | 0.011 | -16.25 | -35.02 | 0.16 | 0.01 | 0.279 | 0.032 |
| parent | equal | 0.052 | 0.128 | 0.010 | -16.30 | -34.99 | 0.14 | 0.01 | 0.238 | 0.028 |
| parent | pooled | 0.057 | 0.140 | 0.011 | -16.25 | -35.03 | 0.16 | 0.01 | 0.279 | 0.032 |
| default_prior | equal | 0.044 | 0.100 | 0.013 | -18.74 | -40.14 | 0.10 | 0.02 | 0.186 | 0.047 |
| default_prior | pooled | 0.044 | 0.100 | 0.013 | -18.74 | -40.14 | 0.10 | 0.02 | 0.186 | 0.047 |
