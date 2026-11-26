# Look-alike discrimination test

When the data barely distinguish candidate forms, does the prior put its mass on the true one? True forms are drawn from the hold-out sources (seen and novel relative to the training sources), the mechanism classes (hsr.mechanisms) and the forms first used from the temporal cutoff 2015 on; datasets are simulated from each (hsr.simulate: constants N(0, 1.5^2) with rejection, x uniform on [0.1, 5], noise sd = noise x range of the curve); every candidate's Laplace log evidence is computed on every dataset (hsr.likelihood; GitHub Actions, .github/workflows/lookalike.yml). Design and scores: hsr.lookalike. Priors: trained on all training sources with equal source weighting (runs mechholdout_equal; for temporal true forms the priors trained on models dated before 2015, run temporal_2015_equal), raw log P(form).

Candidate set: the 5140 support forms of the mechanism hold-out run with at most 5 constants and 15 nodes (5116 of them in the temporal run's support, the candidate set for temporal true forms). 588 datasets: 49 true forms x 4 (n, noise) settings x 3 replicates. The true form's fit succeeded on 93.9% of datasets. Evidence: 3.3 CPU hours.

Look-alike set A_D: candidates within D nats of the best log evidence. rank: rank of the truth in A_D under the prior (1 = the prior's favourite, ties count half), given the truth is in A_D; top: fraction of those datasets where the truth is the prior's strict favourite in A_D; mass: prior probability of the truth renormalised within A_D (flat: 1/|A_D|); gain: median over datasets of log(mass x |A_D|), the log factor by which the prior beats the flat prior on the truth; P(true): posterior probability of the truth over the whole candidate set (log evidence + log prior); MAP: fraction of datasets where the truth is the posterior mode.

## True forms

| stratum | eligible distinct forms | chosen | seen | not simulable (skipped) |
|---|---|---|---|---|
| holdout_seen | 73 | 10 | 10 | 0 |
| holdout_novel | 25 | 10 | 0 | 0 |
| mech_enzyme | 6 | 6 | 6 | 0 |
| mech_compartment | 8 | 8 | 6 | 0 |
| mech_temperature | 3 | 3 | 2 | 0 |
| temporal_novel | 118 | 12 | 0 | 0 |

## How often is the truth a look-alike?

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

## All datasets, D = 2

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

## All datasets, D = 5

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

## All datasets, D = 10

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

## Renormalised prior mass on the truth within A_5, by stratum

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

## Mean rank of the truth within A_5, by stratum

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

## MAP accuracy (posterior mode = truth), by stratum

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

## Posterior probability of the truth, by stratum

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

## Renormalised prior mass on the truth within A_5, by (n, noise)

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

## MAP accuracy, by (n, noise)

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

## True forms used

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
