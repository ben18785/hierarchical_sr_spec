# Temporal hold-out: do the priors anticipate forms not yet used?

Each prior is trained only on the models published before a cutoff year Y and scored on the model-form pairs of the models published from Y on (raw log P of the form under the prior, mean per pair; no renormalisation). Test pairs are split into forms already used by some model before Y ("seen": the generalised key is in the training set) and forms first used at or after Y ("novel"). Years: the publication year of the paper describing the model where the source records it, otherwise the first submission (hsr/data/model_dates.json; hsr.model_dates). Sources without per-model dates (curvelib, nlmixr2lib) are always in the training set; undated models of dated sources are dropped. Tree priors (fnprior, hpy, hpyv variants) share one space, so their raw scores are comparable; the PCFG grammars generate a different space. Source-aware variants score test forms as a new, unseen source. Per-run tables, including the scores renormalised over the common support: results/grammar/temporal_<Y>_<weighting>/table_v.md.

## Date coverage

| source | dated models | undated models | year from |
|---|---|---|---|
| allodb | 53 | 1 |  |
| biocat | 54 | 0 |  |
| biomodels | 1075 | 0 | publication 1075 |
| biomodels_nc | 531 | 0 | publication 495, first submission 36 |
| cellml | 328 | 19 |  |
| ecocurves | 83 | 0 |  |
| jws | 522 | 151 | undated 121, manuscript 522 |
| modeldb | 778 | 0 | paper 743, name 26, created 9 |
| sabiork | 5057 | 0 |  |
| thermal | 61 | 1 |  |
| curvelib | - | all (always training) | |
| nlmixr2lib | - | all (always training) | |

Coverage counts every model of the corpus file; the year distribution below counts models with at least one form.

## Year distribution (models with forms)

| source | <1990 | 1990-1999 | 2000-2004 | 2005-2009 | 2010-2014 | 2015-2019 | >=2020 | undated |
|---|---|---|---|---|---|---|---|---|
| biomodels | 10 | 57 | 130 | 179 | 286 | 215 | 70 | 0 |
| sabiork | 1040 | 1046 | 1171 | 996 | 358 | 368 | 78 | 0 |
| cellml | 22 | 69 | 93 | 111 | 24 | 6 | 3 | 19 |
| biocat | 47 | 1 | 0 | 0 | 4 | 2 | 0 | 0 |
| biomodels_nc | 5 | 20 | 33 | 73 | 143 | 150 | 107 | 0 |
| thermal | 25 | 8 | 4 | 7 | 6 | 9 | 2 | 1 |
| ecocurves | 24 | 7 | 12 | 15 | 3 | 1 | 21 | 0 |
| modeldb | 12 | 64 | 65 | 119 | 151 | 211 | 156 | 0 |
| allodb | 19 | 12 | 2 | 4 | 6 | 10 | 0 | 1 |
| jws | 10 | 45 | 97 | 117 | 217 | 30 | 6 | 151 |
| all | 1214 | 1329 | 1607 | 1621 | 1198 | 1002 | 443 | 172 |

## Test sets

| cutoff Y | training pairs (pooled) | test models | test pairs | novel pairs | fraction novel | distinct novel forms | test pairs by source |
|---|---|---|---|---|---|---|---|
| 2010 | 17900 | 2643 | 23279 | 2515 | 0.108 | 944 | biomodels 2273, sabiork 1331, cellml 909, biocat 6, biomodels_nc 2003, thermal 24, ecocurves 25, modeldb 15219, allodb 25, jws 1464 |
| 2015 | 26129 | 1445 | 15050 | 1170 | 0.078 | 541 | biomodels 985, sabiork 733, cellml 299, biocat 2, biomodels_nc 1380, thermal 17, ecocurves 22, modeldb 11438, allodb 17, jws 157 |
| 2020 | 34664 | 443 | 6515 | 309 | 0.047 | 246 | biomodels 171, sabiork 119, cellml 118, biomodels_nc 794, thermal 3, ecocurves 21, modeldb 5258, jws 31 |

## Raw log P per test pair (equal source weighting)

all / seen / novel test pairs; rows ordered by the mean over cutoffs of the novel-form score (tree priors first). Zero-probability pairs are left out of the means (counted in the last column).

| prior | family | 2010 all | 2010 seen | 2010 novel | 2015 all | 2015 seen | 2015 novel | 2020 all | 2020 seen | 2020 novel | zero-probability pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tree_ad3 | tree | -8.05 | -5.40 | -29.97 | -7.82 | -5.73 | -32.69 | -7.33 | -6.19 | -30.30 | 0/0/0 |
| form_ad3_depth | tree | -8.27 | -5.39 | -32.07 | -7.94 | -5.73 | -34.22 | -7.51 | -6.19 | -34.17 | 0/0/0 |
| hpy3_depth | tree | -12.27 | -9.52 | -34.93 | -12.99 | -10.89 | -37.93 | -13.54 | -12.32 | -38.04 | 0/0/0 |
| form_ad | tree | -8.64 | -5.39 | -35.40 | -8.27 | -5.73 | -38.44 | -7.72 | -6.19 | -38.39 | 0/0/0 |
| hpy | tree | -14.65 | -11.83 | -37.87 | -15.73 | -13.53 | -41.83 | -16.60 | -15.37 | -41.32 | 0/0/0 |
| hpy2 | tree | -14.65 | -11.83 | -37.87 | -15.74 | -13.53 | -41.90 | -16.60 | -15.37 | -41.33 | 0/0/0 |
| fnprior_kn | tree | -14.65 | -11.84 | -37.91 | -15.74 | -13.53 | -41.91 | -16.60 | -15.37 | -41.31 | 0/0/0 |
| fnprior_kn_tuned | tree | -14.66 | -11.83 | -37.97 | -15.75 | -13.53 | -42.03 | -16.61 | -15.37 | -41.46 | 0/0/0 |
| hpy2_src | tree | -15.94 | -13.19 | -38.66 | -17.52 | -15.35 | -43.23 | -18.42 | -17.04 | -46.06 | 0/0/0 |
| sym0 | tree | -24.19 | -20.54 | -54.29 | -25.78 | -22.98 | -58.97 | -27.30 | -25.65 | -60.51 | 0/0/0 |
| term_root_backoff20 | pcfg | -24.20 | -19.92 | -59.49 | -25.83 | -22.58 | -64.39 | -27.45 | -25.64 | -63.82 | 0/0/0 |
| term_root | pcfg | -24.20 | -19.92 | -59.50 | -25.83 | -22.57 | -64.41 | -27.45 | -25.64 | -63.87 | 0/0/0 |
| term_backoff20 | pcfg | -24.89 | -20.62 | -60.18 | -26.59 | -23.34 | -65.08 | -28.21 | -26.39 | -64.89 | 0/0/0 |
| term | pcfg | -24.90 | -20.62 | -60.19 | -26.59 | -23.34 | -65.10 | -28.22 | -26.39 | -64.94 | 0/0/0 |
| position_backoff20 | pcfg | -25.24 | -20.94 | -60.72 | -26.97 | -23.71 | -65.62 | -28.62 | -26.79 | -65.55 | 0/0/0 |
| position | pcfg | -25.24 | -20.94 | -60.73 | -26.97 | -23.71 | -65.63 | -28.62 | -26.78 | -65.56 | 0/0/0 |
| parent_backoff20 | pcfg | -26.64 | -22.28 | -62.70 | -28.39 | -25.07 | -67.87 | -30.17 | -28.31 | -67.43 | 0/0/0 |
| parent | pcfg | -26.65 | -22.28 | -62.71 | -28.40 | -25.07 | -67.89 | -30.17 | -28.31 | -67.45 | 0/0/0 |
| cf | pcfg | -27.22 | -22.86 | -63.19 | -29.12 | -25.83 | -68.12 | -30.92 | -28.97 | -70.11 | 0/0/0 |
| default_prior | pcfg | -32.03 | -26.91 | -74.34 | -34.41 | -30.60 | -79.61 | -36.54 | -34.21 | -83.19 | 0/0/0 |

Rank of each tree prior on novel forms (equal; 1 = highest log P):

| prior | 2010 seen | 2010 novel | 2015 seen | 2015 novel | 2020 seen | 2020 novel |
|---|---|---|---|---|---|---|
| tree_ad3 | 3 | 1 | 3 | 1 | 3 | 1 |
| form_ad3_depth | 1 | 2 | 1 | 2 | 1 | 2 |
| hpy3_depth | 4 | 3 | 4 | 3 | 4 | 3 |
| form_ad | 2 | 4 | 2 | 4 | 2 | 4 |
| hpy | 7 | 5 | 6 | 5 | 7 | 6 |
| hpy2 | 6 | 6 | 8 | 6 | 6 | 7 |
| fnprior_kn | 8 | 7 | 7 | 7 | 8 | 5 |
| fnprior_kn_tuned | 5 | 8 | 5 | 8 | 5 | 8 |
| hpy2_src | 9 | 9 | 9 | 9 | 9 | 9 |
| sym0 | 10 | 10 | 10 | 10 | 10 | 10 |

## Raw log P per test pair (pooled source weighting)

all / seen / novel test pairs; rows ordered by the mean over cutoffs of the novel-form score (tree priors first). Zero-probability pairs are left out of the means (counted in the last column).

| prior | family | 2010 all | 2010 seen | 2010 novel | 2015 all | 2015 seen | 2015 novel | 2020 all | 2020 seen | 2020 novel | zero-probability pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tree_ad3 | tree | -7.42 | -4.90 | -28.22 | -7.17 | -5.16 | -31.01 | -6.61 | -5.52 | -28.58 | 0/0/0 |
| form_ad3_depth | tree | -7.65 | -4.90 | -30.34 | -7.31 | -5.16 | -32.71 | -6.79 | -5.52 | -32.31 | 0/0/0 |
| hpy3_depth | tree | -10.98 | -8.47 | -31.70 | -11.67 | -9.72 | -34.74 | -12.10 | -10.97 | -34.79 | 0/0/0 |
| form_ad | tree | -8.01 | -4.90 | -33.65 | -7.63 | -5.17 | -36.89 | -7.00 | -5.53 | -36.58 | 0/0/0 |
| hpy2 | tree | -13.91 | -11.25 | -35.89 | -14.87 | -12.81 | -39.33 | -15.45 | -14.29 | -38.74 | 0/0/0 |
| hpy | tree | -13.91 | -11.25 | -35.89 | -14.87 | -12.81 | -39.35 | -15.45 | -14.29 | -38.73 | 0/0/0 |
| fnprior_kn | tree | -13.91 | -11.25 | -35.92 | -14.87 | -12.81 | -39.38 | -15.44 | -14.29 | -38.71 | 0/0/0 |
| fnprior_kn_tuned | tree | -13.91 | -11.24 | -35.92 | -14.87 | -12.81 | -39.39 | -15.44 | -14.28 | -38.76 | 0/0/0 |
| hpy2_src | tree | -15.08 | -12.55 | -35.98 | -16.60 | -14.65 | -39.70 | -17.15 | -16.02 | -39.84 | 0/0/0 |
| sym0 | tree | -24.04 | -20.34 | -54.54 | -25.45 | -22.66 | -58.53 | -26.70 | -25.08 | -59.23 | 0/0/0 |
| term_root | pcfg | -23.67 | -19.41 | -58.81 | -25.19 | -21.99 | -63.13 | -26.51 | -24.69 | -62.95 | 0/0/0 |
| term_root_backoff20 | pcfg | -23.67 | -19.41 | -58.83 | -25.19 | -21.99 | -63.16 | -26.52 | -24.70 | -62.95 | 0/0/0 |
| term | pcfg | -24.55 | -20.26 | -60.01 | -26.11 | -22.89 | -64.33 | -27.47 | -25.68 | -63.49 | 0/0/0 |
| term_backoff20 | pcfg | -24.56 | -20.26 | -60.03 | -26.12 | -22.89 | -64.35 | -27.48 | -25.69 | -63.48 | 0/0/0 |
| position | pcfg | -25.06 | -20.71 | -60.96 | -26.68 | -23.42 | -65.34 | -28.06 | -26.25 | -64.31 | 0/0/0 |
| position_backoff20 | pcfg | -25.06 | -20.71 | -60.98 | -26.68 | -23.42 | -65.38 | -28.07 | -26.26 | -64.33 | 0/0/0 |
| parent_backoff20 | pcfg | -26.43 | -22.02 | -62.85 | -28.08 | -24.76 | -67.52 | -29.60 | -27.76 | -66.37 | 0/0/0 |
| parent | pcfg | -26.43 | -22.02 | -62.88 | -28.08 | -24.76 | -67.53 | -29.60 | -27.77 | -66.37 | 0/0/0 |
| cf | pcfg | -27.09 | -22.70 | -63.32 | -28.93 | -25.62 | -68.20 | -30.56 | -28.64 | -69.13 | 0/0/0 |
| default_prior | pcfg | -32.03 | -26.91 | -74.34 | -34.41 | -30.60 | -79.61 | -36.54 | -34.21 | -83.19 | 0/0/0 |

Rank of each tree prior on novel forms (pooled; 1 = highest log P):

| prior | 2010 seen | 2010 novel | 2015 seen | 2015 novel | 2020 seen | 2020 novel |
|---|---|---|---|---|---|---|
| tree_ad3 | 1 | 1 | 1 | 1 | 1 | 1 |
| form_ad3_depth | 2 | 2 | 2 | 2 | 2 | 2 |
| hpy3_depth | 4 | 3 | 4 | 3 | 4 | 3 |
| form_ad | 3 | 4 | 3 | 4 | 3 | 4 |
| hpy2 | 7 | 6 | 8 | 5 | 7 | 7 |
| hpy | 8 | 5 | 7 | 6 | 8 | 6 |
| fnprior_kn | 6 | 7 | 6 | 7 | 6 | 5 |
| fnprior_kn_tuned | 5 | 8 | 5 | 8 | 5 | 8 |
| hpy2_src | 9 | 9 | 9 | 9 | 9 | 9 |
| sym0 | 10 | 10 | 10 | 10 | 10 | 10 |

## Novel forms

Forms first used by a model dated at or after the cutoff (generalised keys; the ten used by the most test models).

### Y = 2010: 944 novel forms in 2515 test pairs

| form | test models | sources | first year |
|---|---|---|---|
| `c0*(c11 + c12*exp(c13*x1))*(c14 - exp(c15*x1))/((c1 - exp(c2*x1))*(c3 + c4*x1) + (c5 + c6*exp(c7*x1))*(c8 + c9*x1)*exp(c10*x1))` | 53 | modeldb | 2012 |
| `c2*(c6 - exp(c1*(c0 + x1)))*(c7 - exp(c1*(c5 - x1)))*exp(c1*x1)/(c0*exp(c1*c5) + x1*exp(c1*c5) - (c0 - c3*c4*c5*exp(c1*(c0 + x1)) + c3*c4*c5 + c3*c4*x1*exp(c1*(c0 + x1)) - c3*c4*x1 + x1)*exp(c1*x1))` | 52 | modeldb | 2015 |
| `c0*(c1 + x1)*(c7 + exp(c2*(c3 - x1)))/(c0*(c1 + x1)*(c5 + exp(c2*(c3 - x1))) + c4*(c3 - x1)*(c6 + exp(c2*(c1 + x1))))` | 51 | modeldb | 2015 |
| `c0*(c1 + x1) + c2*(c1 + x1)/(c3 + exp(c4*x1))` | 35 | modeldb | 2010 |
| `c0 + (c1*x1 + c2)*exp(-(c3*x1 + c4)**2)` | 30 | modeldb | 2011 |
| `c5*(c0 + c7*exp(c8*x1))**2/(c2 + c3*exp(c1*x1) + c4*x1*exp(c6*x1) + x1)` | 30 | modeldb | 2011 |
| `(c0*x1 + c1)/(c2*c3 + c2*c4*x1 + c3*x1 + c4*x1**2)` | 24 | biomodels, jws | 2013 |
| `(c2 + c3*x1)*(1/(c1*x1 + c4))**c5/(c0*c7*x1*(1/(c1*x1 + c4))**c5 + c0*c8*(1/(c1*x1 + c4))**c5 + c6*c7*x1 + c6*c8)` | 24 | biomodels, jws | 2013 |
| `c2*(c8 + x1)*(1/(c1*x1 + c3))**c4/((c0*(1/(c1*x1 + c3))**c4 + c5)*(c6*x1 + c7))` | 24 | biomodels, jws | 2013 |
| `c2*(c7 + exp(c4*(c3 - x1)))*exp(c1*(c0 - x1))/(c2*(c6 + exp(c4*(c3 - x1)))*exp(c1*(c0 - x1)) + c5)` | 22 | modeldb | 2010 |

### Y = 2015: 541 novel forms in 1170 test pairs

| form | test models | sources | first year |
|---|---|---|---|
| `c2*(c6 - exp(c1*(c0 + x1)))*(c7 - exp(c1*(c5 - x1)))*exp(c1*x1)/(c0*exp(c1*c5) + x1*exp(c1*c5) - (c0 - c3*c4*c5*exp(c1*(c0 + x1)) + c3*c4*c5 + c3*c4*x1*exp(c1*(c0 + x1)) - c3*c4*x1 + x1)*exp(c1*x1))` | 52 | modeldb | 2015 |
| `c0*(c1 + x1)*(c7 + exp(c2*(c3 - x1)))/(c0*(c1 + x1)*(c5 + exp(c2*(c3 - x1))) + c4*(c3 - x1)*(c6 + exp(c2*(c1 + x1))))` | 51 | modeldb | 2015 |
| `(c0 + c1*log(c2*x1)**2)/(c3 + log(c2*x1)**2)` | 15 | biomodels_nc | 2015 |
| `(c0 + c1*log(c2/x1)**2)/(c3 + log(c2/x1)**2)` | 15 | biomodels_nc | 2015 |
| `(c1 + c2*x1)/(c0*c4*x1*((c3*x1 + c7)/(c0*x1 + c5*x1 + c6))**c8 + c0*x1 + c4*c5*x1*((c3*x1 + c7)/(c0*x1 + c5*x1 + c6))**c8 + c4*c6*((c3*x1 + c7)/(c0*x1 + c5*x1 + c6))**c8 + c5*x1 + c6)` | 15 | biomodels_nc | 2015 |
| `c0*log(c2*x1)**4/(c1 + log(c2*x1)**4)` | 15 | biomodels_nc | 2015 |
| `c0*log(c2/x1)**4/(c1 + log(c2/x1)**4)` | 15 | biomodels_nc | 2015 |
| `c0/(c1 + ((c2*x1 + c6)/(c3*x1 + c5))**c4)` | 15 | biomodels_nc | 2015 |
| `c0/(c1 + (c2 + log(x1))**2)` | 15 | biomodels_nc | 2015 |
| `c0/(c1 + (c2 - log(x1))**2)` | 15 | biomodels_nc | 2015 |

### Y = 2020: 246 novel forms in 309 test pairs

| form | test models | sources | first year |
|---|---|---|---|
| `(c0 + 1/(c1 + c2*exp(c3*x1)))*(c4*exp(c5*x1) + c6*exp(c7*x1))` | 4 | modeldb | 2021 |
| `c3*(c0 + 1/(c2*exp(c6*x1) + c5))/(c1 + exp(c4*x1))` | 4 | modeldb | 2021 |
| `c3*(c0 + 1/(c8 + exp(c1*(c4 - x1))))/(c2 + exp(c5*c6)*exp(c6*x1)*exp(c7*x1**2)*exp(c5*c7*x1))` | 4 | modeldb | 2021 |
| `(c2 + c3*x1)/(c0*c4*x1**2 + c0*c5*x1 + c1*c4*x1 + c1*c5)` | 4 | jws | 2023 |
| `c0*exp(c3*c4 - c4*x1)/(c1 + exp(c2*c3 - c2*x1))` | 3 | modeldb | 2022 |
| `(c1 + c2*exp(c7*x1))/(c4 + c5*x1 + c6*exp(c0*x1) + exp(c3*x1))` | 3 | modeldb | 2020 |
| `(c3 + c4*exp(c5*x1))*(c0*(c7 + c9*exp(c10*x1)) + c6)/((c1 + c8*exp(c11*x1))*(c12 + c13*exp(c2*x1)))` | 3 | modeldb | 2021 |
| `c0*exp((c1 + c2*x1)*(c3 + x1)) + c4` | 3 | modeldb | 2021 |
| `c0*(c2 + x1)**2/(c1 + (c2 + x1)**2) + c3` | 2 | biomodels_nc | 2021 |
| `c0*exp(c2/(c1 + x1))/(c1 + x1)**(3/2)` | 2 | modeldb | 2021 |

