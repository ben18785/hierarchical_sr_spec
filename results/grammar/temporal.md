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

## Excluding modeldb from the test set

modeldb supplies most test pairs and most novel pairs (many of them large, multi-gate channel kinetics), so the summary is repeated with every modeldb test pair dropped. The priors are the same (trained before Y, modeldb included in training); only the test set changes.

| cutoff Y | test pairs | without modeldb | novel pairs | novel without modeldb |
|---|---|---|---|---|
| 2010 | 23279 | 8060 | 2515 | 1024 |
| 2015 | 15050 | 3612 | 1170 | 436 |
| 2020 | 6515 | 1257 | 309 | 45 |

### Raw log P per test pair without modeldb (equal source weighting)

| prior | family | 2010 all | 2010 seen | 2010 novel | 2015 all | 2015 seen | 2015 novel | 2020 all | 2020 seen | 2020 novel | zero-probability pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tree_ad3 | tree | -6.76 | -3.64 | -28.13 | -6.87 | -3.67 | -30.22 | -5.60 | -5.12 | -18.58 | 0/0/0 |
| hpy2_src | tree | -8.19 | -5.37 | -27.58 | -8.24 | -5.22 | -30.29 | -11.10 | -10.74 | -20.74 | 0/0/0 |
| form_ad3_depth | tree | -6.82 | -3.64 | -28.65 | -6.88 | -3.67 | -30.24 | -5.71 | -5.12 | -21.46 | 0/0/0 |
| hpy3_depth | tree | -7.55 | -4.50 | -28.50 | -7.78 | -4.48 | -31.88 | -10.14 | -9.74 | -20.94 | 0/0/0 |
| form_ad | tree | -6.99 | -3.64 | -29.97 | -7.02 | -3.67 | -31.43 | -5.73 | -5.12 | -22.14 | 0/0/0 |
| hpy | tree | -8.40 | -5.29 | -29.83 | -8.56 | -5.17 | -33.25 | -11.61 | -11.26 | -20.91 | 0/0/0 |
| hpy2 | tree | -8.40 | -5.29 | -29.78 | -8.58 | -5.17 | -33.38 | -11.60 | -11.26 | -20.89 | 0/0/0 |
| fnprior_kn | tree | -8.41 | -5.28 | -29.87 | -8.59 | -5.17 | -33.44 | -11.61 | -11.26 | -20.94 | 0/0/0 |
| fnprior_kn_tuned | tree | -8.41 | -5.28 | -29.92 | -8.60 | -5.17 | -33.56 | -11.60 | -11.26 | -20.95 | 0/0/0 |
| sym0 | tree | -16.23 | -11.91 | -45.94 | -16.03 | -11.50 | -48.98 | -21.15 | -20.69 | -33.60 | 0/0/0 |
| term_root_backoff20 | pcfg | -16.09 | -10.75 | -52.82 | -16.37 | -10.46 | -59.44 | -23.16 | -22.57 | -39.11 | 0/0/0 |
| term_root | pcfg | -16.09 | -10.75 | -52.84 | -16.38 | -10.46 | -59.48 | -23.16 | -22.57 | -39.18 | 0/0/0 |
| position_backoff20 | pcfg | -16.74 | -11.47 | -52.96 | -16.84 | -11.06 | -58.94 | -23.51 | -22.91 | -39.59 | 0/0/0 |
| position | pcfg | -16.74 | -11.47 | -52.98 | -16.84 | -11.06 | -58.97 | -23.51 | -22.91 | -39.63 | 0/0/0 |
| term_backoff20 | pcfg | -16.66 | -11.38 | -53.01 | -16.87 | -11.04 | -59.35 | -23.50 | -22.90 | -39.59 | 0/0/0 |
| term | pcfg | -16.67 | -11.37 | -53.02 | -16.88 | -11.04 | -59.39 | -23.50 | -22.90 | -39.67 | 0/0/0 |
| cf | pcfg | -17.92 | -12.77 | -53.34 | -17.91 | -12.33 | -58.54 | -24.65 | -24.04 | -41.13 | 0/0/0 |
| parent_backoff20 | pcfg | -18.15 | -12.69 | -55.63 | -18.29 | -12.29 | -62.01 | -25.39 | -24.80 | -41.06 | 0/0/0 |
| parent | pcfg | -18.15 | -12.69 | -55.64 | -18.29 | -12.29 | -62.02 | -25.39 | -24.81 | -41.09 | 0/0/0 |
| default_prior | pcfg | -20.36 | -14.30 | -62.03 | -20.10 | -13.77 | -66.23 | -27.56 | -26.81 | -47.89 | 0/0/0 |

### Raw log P per test pair without modeldb (pooled source weighting)

| prior | family | 2010 all | 2010 seen | 2010 novel | 2015 all | 2015 seen | 2015 novel | 2020 all | 2020 seen | 2020 novel | zero-probability pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tree_ad3 | tree | -6.45 | -3.54 | -26.42 | -6.54 | -3.51 | -28.59 | -5.76 | -5.34 | -17.07 | 0/0/0 |
| form_ad3_depth | tree | -6.51 | -3.54 | -26.92 | -6.55 | -3.52 | -28.65 | -5.87 | -5.35 | -19.91 | 0/0/0 |
| hpy2_src | tree | -8.34 | -5.62 | -27.05 | -8.39 | -5.54 | -29.18 | -11.05 | -10.73 | -19.84 | 0/0/0 |
| hpy3_depth | tree | -7.37 | -4.45 | -27.43 | -7.46 | -4.33 | -30.29 | -10.41 | -10.03 | -20.75 | 0/0/0 |
| form_ad | tree | -6.67 | -3.54 | -28.20 | -6.71 | -3.52 | -29.99 | -5.91 | -5.36 | -20.60 | 0/0/0 |
| hpy2 | tree | -8.29 | -5.25 | -29.15 | -8.36 | -5.19 | -31.49 | -11.67 | -11.33 | -20.67 | 0/0/0 |
| hpy | tree | -8.29 | -5.25 | -29.17 | -8.38 | -5.19 | -31.61 | -11.67 | -11.34 | -20.68 | 0/0/0 |
| fnprior_kn_tuned | tree | -8.30 | -5.25 | -29.23 | -8.38 | -5.18 | -31.64 | -11.66 | -11.32 | -20.69 | 0/0/0 |
| fnprior_kn | tree | -8.30 | -5.25 | -29.23 | -8.38 | -5.19 | -31.67 | -11.67 | -11.33 | -20.69 | 0/0/0 |
| sym0 | tree | -16.30 | -11.86 | -46.76 | -16.02 | -11.50 | -49.00 | -21.48 | -20.99 | -34.42 | 0/0/0 |
| term_root | pcfg | -15.85 | -10.58 | -52.04 | -15.98 | -10.31 | -57.27 | -22.71 | -22.07 | -39.80 | 0/0/0 |
| term_root_backoff20 | pcfg | -15.85 | -10.58 | -52.07 | -15.99 | -10.31 | -57.31 | -22.73 | -22.10 | -39.76 | 0/0/0 |
| term | pcfg | -16.52 | -11.23 | -52.90 | -16.62 | -10.91 | -58.21 | -23.47 | -22.85 | -40.33 | 0/0/0 |
| term_backoff20 | pcfg | -16.53 | -11.23 | -52.93 | -16.62 | -10.91 | -58.25 | -23.50 | -22.87 | -40.29 | 0/0/0 |
| position | pcfg | -16.67 | -11.32 | -53.40 | -16.76 | -10.99 | -58.78 | -23.63 | -23.01 | -40.22 | 0/0/0 |
| position_backoff20 | pcfg | -16.68 | -11.32 | -53.46 | -16.77 | -10.99 | -58.87 | -23.65 | -23.04 | -40.17 | 0/0/0 |
| cf | pcfg | -17.88 | -12.68 | -53.63 | -17.97 | -12.30 | -59.25 | -25.09 | -24.48 | -41.55 | 0/0/0 |
| parent_backoff20 | pcfg | -18.07 | -12.55 | -56.01 | -18.19 | -12.23 | -61.61 | -25.40 | -24.79 | -41.68 | 0/0/0 |
| parent | pcfg | -18.08 | -12.55 | -56.06 | -18.19 | -12.23 | -61.63 | -25.40 | -24.79 | -41.71 | 0/0/0 |
| default_prior | pcfg | -20.36 | -14.30 | -62.03 | -20.10 | -13.77 | -66.23 | -27.56 | -26.81 | -47.89 | 0/0/0 |

## Why novel forms score low

As in holdout.md (hsr.gap_diagnostics): novel forms are larger than seen ones and every prior pays per node; sym0 (symbol frequencies only) measures what size and symbol use alone cost. per node: OLS slope of log P on node count; novel penalty: OLS coefficient of the novel indicator at fixed node count (log P = a + b nodes + g novel, over model-form pairs); vs sym0: log P minus sym0's log P, seen / novel; novel (non-redundant): mean log P of novel pairs whose constants are identifiable.

### Size against genuine novelty

Per cutoff: seen-minus-novel gap in mean log P per pair, and the OLS novelty penalty at fixed node count (gap / penalty; the penalty is the OLS coefficient, negative when novel forms score lower than seen forms of the same size; roughly, gap + penalty is the part of the gap due to size), on all test pairs and without modeldb.

| prior | weighting | 2010 gap / penalty | 2010 no modeldb | 2015 gap / penalty | 2015 no modeldb | 2020 gap / penalty | 2020 no modeldb |
|---|---|---|---|---|---|---|---|
| tree_ad3 | equal | 24.57 / -17.37 | 24.49 / -10.93 | 26.97 / -20.56 | 26.55 / -11.48 | 24.11 / -18.84 | 13.46 / -12.33 |
| tree_ad3 | pooled | 23.32 / -16.60 | 22.88 / -9.32 | 25.85 / -19.89 | 25.07 / -10.08 | 23.06 / -18.16 | 11.73 / -10.43 |
| form_ad3_depth | equal | 26.67 / -18.52 | 25.01 / -11.38 | 28.50 / -21.30 | 26.57 / -11.27 | 27.98 / -21.40 | 16.34 / -15.15 |
| form_ad3_depth | pooled | 25.44 / -17.78 | 23.39 / -9.74 | 27.55 / -20.75 | 25.13 / -9.83 | 26.79 / -20.62 | 14.56 / -13.19 |
| hpy3_depth | equal | 25.41 / -5.99 | 24.00 / -5.63 | 27.04 / -6.63 | 27.41 / -7.17 | 25.72 / -6.53 | 11.20 / -4.45 |
| hpy3_depth | pooled | 23.22 / -6.86 | 22.98 / -4.90 | 25.02 / -7.35 | 25.96 / -5.84 | 23.82 / -7.11 | 10.72 / -3.65 |
| hpy2 | equal | 26.03 / -1.54 | 24.50 / -2.82 | 28.37 / -2.57 | 28.21 / -5.01 | 25.96 / -1.87 | 9.63 / -1.94 |
| hpy2 | pooled | 24.65 / -2.18 | 23.90 / -2.55 | 26.52 / -2.95 | 26.30 / -4.14 | 24.46 / -2.87 | 9.34 / -1.77 |
| fnprior_kn | equal | 26.07 / -1.58 | 24.59 / -2.92 | 28.38 / -2.59 | 28.26 / -5.07 | 25.93 / -1.85 | 9.68 / -1.99 |
| fnprior_kn | pooled | 24.68 / -2.21 | 23.98 / -2.57 | 26.57 / -3.01 | 26.48 / -4.29 | 24.42 / -2.84 | 9.36 / -1.79 |
| sym0 | equal | 33.75 / -1.83 | 34.04 / -2.41 | 35.99 / -2.11 | 37.48 / -4.24 | 34.86 / -1.85 | 12.92 / -1.28 |
| sym0 | pooled | 34.20 / -2.43 | 34.90 / -2.62 | 35.87 / -2.48 | 37.50 / -4.22 | 34.15 / -2.26 | 13.42 / -1.88 |

Mean node count, seen / novel pairs: 2010: 12.7 / 32.7 (without modeldb 7.3 / 27.4); 2015: 14.3 / 35.7 (without modeldb 7.1 / 29.1); 2020: 15.9 / 36.6 (without modeldb 12.5 / 19.9).

### Y = 2010

Log P by node count (seen / novel; pooled weighting):

| prior | 1-5 nodes | 6-9 nodes | 10-14 nodes | 15-20 nodes | 21-10000 nodes |
|---|---|---|---|---|---|
| pairs (seen / novel) | 6644 / 33 | 5134 / 102 | 3423 / 270 | 1512 / 549 | 4051 / 1561 |
| tree_ad3 | -3.13 / -7.99 | -3.94 / -11.46 | -5.63 / -13.25 | -7.10 / -16.88 | -7.60 / -36.31 |
| form_ad3_depth | -3.13 / -8.25 | -3.94 / -11.50 | -5.63 / -13.19 | -7.10 / -17.09 | -7.60 / -39.66 |
| hpy3_depth | -3.17 / -7.96 | -4.11 / -10.83 | -7.36 / -12.97 | -10.82 / -17.92 | -22.76 / -41.64 |
| form_ad | -3.13 / -8.05 | -3.94 / -11.41 | -5.63 / -13.90 | -7.11 / -18.30 | -7.60 / -44.46 |
| hpy | -3.12 / -7.85 | -5.72 / -10.59 | -9.03 / -13.44 | -15.57 / -18.70 | -31.83 / -48.07 |
| hpy2 | -3.12 / -7.80 | -5.72 / -10.69 | -9.04 / -13.43 | -15.57 / -18.71 | -31.83 / -48.06 |
| fnprior_kn | -3.12 / -7.83 | -5.72 / -10.57 | -9.03 / -13.41 | -15.57 / -18.78 | -31.83 / -48.09 |
| fnprior_kn_tuned | -3.12 / -7.85 | -5.72 / -10.58 | -9.03 / -13.42 | -15.57 / -18.78 | -31.83 / -48.09 |
| hpy2_src | -3.14 / -6.53 | -6.68 / -9.61 | -10.01 / -12.47 | -17.46 / -18.44 | -35.74 / -48.56 |
| sym0 | -6.26 / -10.69 | -12.12 / -15.97 | -19.69 / -22.58 | -26.72 / -31.69 | -52.03 / -71.55 |

Novelty at fixed size, relative to sym0, and without redundant forms:

| prior | weighting | seen | novel | gap | per node | novel penalty | vs sym0 seen / novel | novel (non-redundant) |
|---|---|---|---|---|---|---|---|---|
| tree_ad3 | equal | -5.40 | -29.97 | 24.57 | -0.36 | -17.37 | 15.15 / 24.32 | -19.29 |
| tree_ad3 | pooled | -4.90 | -28.22 | 23.32 | -0.34 | -16.60 | 15.44 / 26.32 | -17.80 |
| form_ad3_depth | equal | -5.39 | -32.07 | 26.67 | -0.41 | -18.52 | 15.15 / 22.22 | -20.05 |
| form_ad3_depth | pooled | -4.90 | -30.34 | 25.44 | -0.38 | -17.78 | 15.44 / 24.20 | -18.53 |
| hpy3_depth | equal | -9.52 | -34.93 | 25.41 | -0.97 | -5.99 | 11.02 / 19.36 | -21.20 |
| hpy3_depth | pooled | -8.47 | -31.70 | 23.22 | -0.82 | -6.86 | 11.87 / 22.84 | -19.69 |
| form_ad | equal | -5.39 | -35.40 | 30.00 | -0.46 | -20.87 | 15.15 / 18.90 | -21.46 |
| form_ad | pooled | -4.90 | -33.65 | 28.74 | -0.43 | -20.06 | 15.44 / 20.89 | -19.92 |
| hpy | equal | -11.83 | -37.87 | 26.03 | -1.23 | -1.55 | 8.71 / 16.43 | -21.55 |
| hpy | pooled | -11.25 | -35.89 | 24.64 | -1.12 | -2.18 | 9.10 / 18.65 | -21.13 |
| hpy2 | equal | -11.83 | -37.87 | 26.03 | -1.23 | -1.54 | 8.71 / 16.42 | -21.52 |
| hpy2 | pooled | -11.25 | -35.89 | 24.65 | -1.12 | -2.18 | 9.10 / 18.65 | -21.14 |
| fnprior_kn | equal | -11.84 | -37.91 | 26.07 | -1.23 | -1.58 | 8.71 / 16.38 | -21.60 |
| fnprior_kn | pooled | -11.25 | -35.92 | 24.68 | -1.12 | -2.21 | 9.10 / 18.62 | -21.18 |
| fnprior_kn_tuned | equal | -11.83 | -37.97 | 26.13 | -1.22 | -1.66 | 8.71 / 16.33 | -21.70 |
| fnprior_kn_tuned | pooled | -11.24 | -35.92 | 24.68 | -1.12 | -2.22 | 9.10 / 18.62 | -21.19 |
| hpy2_src | equal | -13.19 | -38.66 | 25.48 | -1.30 | 0.41 | 7.36 / 15.63 | -22.31 |
| hpy2_src | pooled | -12.55 | -35.98 | 23.43 | -1.18 | 0.08 | 7.79 / 18.56 | -20.75 |
| sym0 | equal | -20.54 | -54.29 | 33.75 | -1.60 | -1.83 | 0.00 / 0.00 | -34.63 |
| sym0 | pooled | -20.34 | -54.54 | 34.20 | -1.59 | -2.43 | 0.00 / 0.00 | -34.91 |

Redundant novel forms (constants not identifiable; Jacobian rank test): 525 of 944 distinct (1502 of 2515 pairs). Examples (all in temporal.json): `(c0 + 1/(c1 + c2*exp(c3*x1)))*(c4*exp(c5*x1) + c6*exp(c7*x1))`; `(c0 + 1/(c4 + c5*exp(c6*x1)))*(c1 + c2*exp(c3*x1))`; `(c0 + 1/(c7 + exp(c1*(c2 - x1))))*(c5 + c6*exp(c3*(c4 - x1)))`; `(c0 + c1*exp(c3*x1) + c12*exp(c2*x1) + c13*exp(c11*x1) + c14*exp(c6*x1) + c15*exp(c7*x1))/(c4*c9 + c4*exp(c10*x1) + c5*c9*exp(c8*x1) + c5*exp(c10*x1)*exp(c8*x1))`; `(c0 + c1*exp(c6*x1))*(c7 + exp(c8*x1))/((c11 + c12*exp(c13*x1))*(c14 + c2*x1)*exp(c3*x1) + (c4 + c5*x1)*(c9 + exp(c10*x1)))`; `(c0 + c1*x1)/(c2 + exp(c3*(c4 - x1)))`; `(c0 + c1*x1)/(c2 - exp(c3*(c4 + x1)))`; `(c0 + c1*x1)/(c2 - exp(c3*(c4 - x1)))`; `(c0 + c1*x1**2)/(c2 + (c3*x1 + c4)**2)`; `(c0 + c1*x1**4)/(c2 + (c3*x1 + c4)**4)`.

### Y = 2015

Log P by node count (seen / novel; pooled weighting):

| prior | 1-5 nodes | 6-9 nodes | 10-14 nodes | 15-20 nodes | 21-10000 nodes |
|---|---|---|---|---|---|
| pairs (seen / novel) | 3988 / 5 | 3263 / 39 | 2132 / 112 | 1204 / 250 | 3293 / 764 |
| tree_ad3 | -3.18 / -9.22 | -4.15 / -12.57 | -5.68 / -14.65 | -7.06 / -18.16 | -7.54 / -38.70 |
| form_ad3_depth | -3.18 / -9.08 | -4.15 / -12.39 | -5.68 / -14.45 | -7.07 / -18.27 | -7.54 / -41.31 |
| hpy3_depth | -3.23 / -8.75 | -4.33 / -11.73 | -7.43 / -14.09 | -11.03 / -18.64 | -23.93 / -44.38 |
| form_ad | -3.18 / -9.66 | -4.15 / -12.21 | -5.68 / -15.80 | -7.07 / -20.12 | -7.55 / -46.91 |
| hpy | -3.18 / -8.81 | -6.01 / -11.25 | -9.36 / -14.41 | -15.47 / -20.24 | -32.46 / -50.90 |
| hpy2 | -3.18 / -8.90 | -6.01 / -11.22 | -9.36 / -14.43 | -15.47 / -20.07 | -32.47 / -50.92 |
| fnprior_kn | -3.18 / -9.22 | -6.01 / -11.08 | -9.35 / -14.44 | -15.47 / -20.25 | -32.46 / -50.94 |
| fnprior_kn_tuned | -3.18 / -9.23 | -6.01 / -11.13 | -9.35 / -14.43 | -15.47 / -20.27 | -32.46 / -50.94 |
| hpy2_src | -3.27 / -8.12 | -6.98 / -9.84 | -10.62 / -13.09 | -17.76 / -19.99 | -37.51 / -51.78 |
| sym0 | -6.44 / -12.00 | -12.41 / -16.46 | -19.72 / -22.88 | -26.34 / -33.40 | -53.02 / -74.44 |

Novelty at fixed size, relative to sym0, and without redundant forms:

| prior | weighting | seen | novel | gap | per node | novel penalty | vs sym0 seen / novel | novel (non-redundant) |
|---|---|---|---|---|---|---|---|---|
| tree_ad3 | equal | -5.73 | -32.69 | 26.97 | -0.30 | -20.56 | 17.25 / 26.28 | -22.81 |
| tree_ad3 | pooled | -5.16 | -31.01 | 25.85 | -0.28 | -19.89 | 17.50 / 27.52 | -21.24 |
| form_ad3_depth | equal | -5.73 | -34.22 | 28.50 | -0.34 | -21.30 | 17.25 / 24.75 | -23.70 |
| form_ad3_depth | pooled | -5.16 | -32.71 | 27.55 | -0.32 | -20.75 | 17.50 / 25.82 | -22.22 |
| hpy3_depth | equal | -10.89 | -37.93 | 27.04 | -0.95 | -6.63 | 12.09 / 21.05 | -25.51 |
| hpy3_depth | pooled | -9.72 | -34.74 | 25.02 | -0.82 | -7.35 | 12.94 / 23.80 | -23.72 |
| form_ad | equal | -5.73 | -38.44 | 32.71 | -0.38 | -24.54 | 17.25 / 20.54 | -25.47 |
| form_ad | pooled | -5.17 | -36.89 | 31.73 | -0.36 | -24.00 | 17.49 / 21.64 | -24.04 |
| hpy | equal | -13.53 | -41.83 | 28.30 | -1.20 | -2.50 | 9.45 / 17.14 | -25.99 |
| hpy | pooled | -12.81 | -39.35 | 26.54 | -1.10 | -2.98 | 9.85 / 19.18 | -25.12 |
| hpy2 | equal | -13.53 | -41.90 | 28.37 | -1.20 | -2.57 | 9.45 / 17.07 | -26.13 |
| hpy2 | pooled | -12.81 | -39.33 | 26.52 | -1.10 | -2.95 | 9.85 / 19.20 | -25.07 |
| fnprior_kn | equal | -13.53 | -41.91 | 28.38 | -1.20 | -2.59 | 9.45 / 17.06 | -26.15 |
| fnprior_kn | pooled | -12.81 | -39.38 | 26.57 | -1.10 | -3.01 | 9.85 / 19.15 | -25.15 |
| fnprior_kn_tuned | equal | -13.53 | -42.03 | 28.50 | -1.20 | -2.71 | 9.45 / 16.95 | -26.43 |
| fnprior_kn_tuned | pooled | -12.81 | -39.39 | 26.58 | -1.10 | -3.02 | 9.86 / 19.15 | -25.19 |
| hpy2_src | equal | -15.35 | -43.23 | 27.88 | -1.32 | 0.36 | 7.63 / 15.75 | -26.80 |
| hpy2_src | pooled | -14.65 | -39.70 | 25.05 | -1.18 | 0.34 | 8.01 / 18.83 | -24.89 |
| sym0 | equal | -22.98 | -58.97 | 35.99 | -1.58 | -2.11 | 0.00 / 0.00 | -39.93 |
| sym0 | pooled | -22.66 | -58.53 | 35.87 | -1.56 | -2.48 | 0.00 / 0.00 | -39.68 |

Redundant novel forms (constants not identifiable; Jacobian rank test): 302 of 541 distinct (735 of 1170 pairs). Examples (all in temporal.json): `(c0 + 1/(c1 + c2*exp(c3*x1)))*(c4*exp(c5*x1) + c6*exp(c7*x1))`; `(c0 + 1/(c4 + c5*exp(c6*x1)))*(c1 + c2*exp(c3*x1))`; `(c0 + 1/(c7 + exp(c1*(c2 - x1))))*(c5 + c6*exp(c3*(c4 - x1)))`; `(c0 + c1/(c2 + c3*exp(c4*x1)))*(c5 + c6/(c7 + c8*exp(c9*x1)))`; `(c0 + exp(c1*(c2 + x1)))/(c3 + exp(c1*(c2 + x1)))`; `(c0 - 1/(c1 + c4*exp(c5*x1)))*(c6*exp(c7*x1) + (c10 + c2*x1)*exp(c3*x1)/(c8 + exp(c9*x1)))`; `(c0*x1 + c1)*exp(c2*(c3 + x1))`; `(c0*x1 + c2)**2/(c0*x1 + c1)**2`; `(c0/(c5 + exp(c1*(c2 - x1))) + c3)*(c0/(c6 + exp(c1*(c4 + x1))) + c3)`; `(c1 + c2*x1)/(c0*c4*x1*((c3*x1 + c7)/(c0*x1 + c5*x1 + c6))**c8 + c0*x1 + c4*c5*x1*((c3*x1 + c7)/(c0*x1 + c5*x1 + c6))**c8 + c4*c6*((c3*x1 + c7)/(c0*x1 + c5*x1 + c6))**c8 + c5*x1 + c6)`.

### Y = 2020

Log P by node count (seen / novel; pooled weighting):

| prior | 1-5 nodes | 6-9 nodes | 10-14 nodes | 15-20 nodes | 21-10000 nodes |
|---|---|---|---|---|---|
| pairs (seen / novel) | 1568 / 2 | 1249 / 13 | 1011 / 39 | 655 / 46 | 1723 / 209 |
| tree_ad3 | -3.34 / -8.21 | -4.27 / -12.26 | -5.78 / -13.92 | -7.46 / -14.65 | -7.52 / -35.59 |
| form_ad3_depth | -3.34 / -8.26 | -4.27 / -12.26 | -5.79 / -13.83 | -7.46 / -14.57 | -7.52 / -41.14 |
| hpy3_depth | -3.39 / -7.96 | -4.45 / -11.58 | -7.60 / -14.03 | -12.17 / -15.52 | -24.12 / -44.60 |
| form_ad | -3.34 / -8.19 | -4.27 / -11.86 | -5.79 / -14.57 | -7.47 / -15.98 | -7.52 / -47.02 |
| hpy | -3.41 / -8.10 | -6.18 / -10.72 | -9.55 / -13.74 | -16.53 / -17.41 | -31.99 / -50.12 |
| hpy2 | -3.41 / -8.23 | -6.18 / -10.71 | -9.55 / -13.76 | -16.53 / -17.40 | -31.99 / -50.14 |
| fnprior_kn | -3.41 / -8.31 | -6.18 / -10.77 | -9.54 / -13.82 | -16.52 / -17.36 | -31.99 / -50.08 |
| fnprior_kn_tuned | -3.41 / -8.15 | -6.18 / -10.80 | -9.54 / -13.83 | -16.51 / -17.40 | -31.98 / -50.14 |
| hpy2_src | -3.54 / -7.62 | -7.22 / -9.12 | -10.52 / -13.77 | -18.10 / -17.92 | -36.20 / -51.75 |
| sym0 | -6.71 / -11.94 | -12.51 / -15.17 | -19.84 / -21.17 | -28.67 / -29.26 | -52.63 / -76.12 |

Novelty at fixed size, relative to sym0, and without redundant forms:

| prior | weighting | seen | novel | gap | per node | novel penalty | vs sym0 seen / novel | novel (non-redundant) |
|---|---|---|---|---|---|---|---|---|
| tree_ad3 | equal | -6.19 | -30.30 | 24.11 | -0.25 | -18.84 | 19.46 / 30.21 | -22.14 |
| tree_ad3 | pooled | -5.52 | -28.58 | 23.06 | -0.24 | -18.16 | 19.56 / 30.65 | -20.59 |
| form_ad3_depth | equal | -6.19 | -34.17 | 27.98 | -0.32 | -21.40 | 19.46 / 26.34 | -23.45 |
| form_ad3_depth | pooled | -5.52 | -32.31 | 26.79 | -0.30 | -20.62 | 19.56 / 26.91 | -21.75 |
| hpy3_depth | equal | -12.32 | -38.04 | 25.72 | -0.93 | -6.53 | 13.33 / 22.47 | -23.77 |
| hpy3_depth | pooled | -10.97 | -34.79 | 23.82 | -0.81 | -7.11 | 14.11 / 24.44 | -23.63 |
| form_ad | equal | -6.19 | -38.39 | 32.21 | -0.36 | -24.84 | 19.46 / 22.11 | -24.84 |
| form_ad | pooled | -5.53 | -36.58 | 31.05 | -0.34 | -24.07 | 19.56 / 22.65 | -23.23 |
| hpy | equal | -15.37 | -41.32 | 25.95 | -1.16 | -1.85 | 10.28 / 19.19 | -23.90 |
| hpy | pooled | -14.29 | -38.73 | 24.44 | -1.04 | -2.85 | 10.79 / 20.50 | -24.26 |
| hpy2 | equal | -15.37 | -41.33 | 25.96 | -1.16 | -1.87 | 10.28 / 19.18 | -23.95 |
| hpy2 | pooled | -14.29 | -38.74 | 24.46 | -1.04 | -2.87 | 10.79 / 20.48 | -24.28 |
| fnprior_kn | equal | -15.37 | -41.31 | 25.93 | -1.16 | -1.85 | 10.28 / 19.20 | -23.90 |
| fnprior_kn | pooled | -14.29 | -38.71 | 24.42 | -1.04 | -2.84 | 10.80 / 20.52 | -24.23 |
| fnprior_kn_tuned | equal | -15.37 | -41.46 | 26.09 | -1.16 | -2.00 | 10.28 / 19.05 | -24.14 |
| fnprior_kn_tuned | pooled | -14.28 | -38.76 | 24.48 | -1.04 | -2.90 | 10.80 / 20.47 | -24.32 |
| hpy2_src | equal | -17.04 | -46.06 | 29.02 | -1.29 | -2.30 | 8.61 / 14.45 | -25.67 |
| hpy2_src | pooled | -16.02 | -39.84 | 23.81 | -1.09 | -1.27 | 9.06 / 19.39 | -24.13 |
| sym0 | equal | -25.65 | -60.51 | 34.86 | -1.60 | -1.85 | 0.00 / 0.00 | -38.11 |
| sym0 | pooled | -25.08 | -59.23 | 34.15 | -1.54 | -2.26 | 0.00 / 0.00 | -37.75 |

Redundant novel forms (constants not identifiable; Jacobian rank test): 141 of 246 distinct (184 of 309 pairs). Examples (all in temporal.json): `(c0 + 1/(c1 + c2*exp(c3*x1)))*(c4*exp(c5*x1) + c6*exp(c7*x1))`; `(c0 + c1/(c2 + c3*exp(c4*x1)))*(c5 + c6/(c7 + c8*exp(c9*x1)))`; `(c0 + exp(c1*(c2 + x1)))/(c3 + exp(c1*(c2 + x1)))`; `(c0*x1 + c1)*exp(c2*(c3 + x1))`; `(c0/(c5 + exp(c1*(c2 - x1))) + c3)*(c0/(c6 + exp(c1*(c4 + x1))) + c3)`; `(c1 + c2*x1**c6)/(c0 + (c3*x1 + c4)**c5)`; `(c1 + c4*x1)*(c2 - exp(c0 + c3*x1))/((c1 + c9*x1)*(c7 - exp(c0 + c8*x1)) + (c10 + c6*exp(c11*x1))*(c12*x1 + c5))`; `(c1 + c8*x1)*(c5 + c6*exp(c0 + c7*x1))/((c1 + c14*x1)*(c11 + c12*exp(c0 + c13*x1)) + (c10*exp(c3*x1) + c2)*(c4*x1 + c9))`; `(c1*(c0*exp(c11*x1) + c10) + c5*x1*(c0*exp(c3*x1) + c2) + c6*x1*exp(c4*x1) + c7*exp(c12*x1))/(c0*exp(c9*x1) + c8)`; `(c1*x1*(c9 + exp(c0 + c10*x1))**3 + c2*exp(2*c0 + c5*x1) + c4*exp(c0 + c8*x1) + c6*(c9 + exp(c0 + c10*x1))**3)/(c3 + exp(c0 + c7*x1))**3`.

## Novel pairs by test source

Mean raw log P of each source's novel test pairs (pooled weighting; the other weighting is in temporal.json), with the mean node count of the source's seen and novel pairs.

### Y = 2010

| source | test pairs | novel pairs | distinct novel | nodes seen / novel | tree_ad3 | form_ad3_depth | hpy3_depth | hpy2 | fnprior_kn | sym0 | term_root |
|---|---|---|---|---|---|---|---|---|---|---|---|
| allodb | 25 | 1 | 1 | 8.0 / 17.0 | -41.40 | -41.63 | -46.75 | -51.76 | -52.73 | -44.21 | -48.36 |
| biomodels | 2273 | 244 | 147 | 6.6 / 24.4 | -22.82 | -23.11 | -22.99 | -24.32 | -24.33 | -42.62 | -45.69 |
| biomodels_nc | 2003 | 400 | 102 | 7.3 / 29.5 | -30.17 | -29.86 | -30.45 | -32.30 | -32.48 | -50.36 | -58.45 |
| cellml | 909 | 124 | 59 | 8.7 / 18.6 | -16.32 | -17.42 | -18.65 | -20.17 | -20.12 | -32.81 | -37.19 |
| ecocurves | 25 | 7 | 7 | 8.2 / 11.6 | -14.21 | -16.30 | -16.74 | -12.37 | -12.43 | -18.44 | -23.25 |
| jws | 1464 | 212 | 66 | 9.0 / 33.9 | -30.57 | -32.53 | -33.17 | -35.49 | -35.53 | -55.84 | -60.13 |
| modeldb | 15219 | 1491 | 611 | 15.5 / 36.3 | -29.45 | -32.68 | -34.62 | -40.52 | -40.52 | -59.88 | -63.46 |
| sabiork | 1331 | 19 | 18 | 5.9 / 21.6 | -19.85 | -19.93 | -19.94 | -22.19 | -22.26 | -36.20 | -31.02 |
| thermal | 24 | 17 | 17 | 12.7 / 18.2 | -23.35 | -23.39 | -24.34 | -24.06 | -24.00 | -33.74 | -35.15 |

### Y = 2015

| source | test pairs | novel pairs | distinct novel | nodes seen / novel | tree_ad3 | form_ad3_depth | hpy3_depth | hpy2 | fnprior_kn | sym0 | term_root |
|---|---|---|---|---|---|---|---|---|---|---|---|
| biomodels | 985 | 44 | 38 | 6.4 / 29.5 | -23.32 | -25.38 | -30.59 | -30.78 | -30.74 | -49.95 | -62.90 |
| biomodels_nc | 1380 | 316 | 64 | 7.6 / 30.4 | -31.26 | -30.63 | -32.17 | -33.50 | -33.74 | -51.45 | -60.93 |
| cellml | 299 | 23 | 18 | 9.5 / 26.4 | -18.01 | -18.92 | -20.09 | -23.57 | -23.57 | -40.79 | -48.28 |
| ecocurves | 22 | 7 | 7 | 8.5 / 11.6 | -14.58 | -15.97 | -16.44 | -12.14 | -12.16 | -18.28 | -23.29 |
| jws | 157 | 25 | 8 | 7.8 / 27.6 | -21.86 | -25.91 | -24.71 | -26.26 | -26.26 | -43.81 | -40.09 |
| modeldb | 11438 | 734 | 400 | 16.4 / 39.7 | -32.45 | -35.12 | -37.38 | -43.99 | -43.96 | -64.20 | -66.60 |
| sabiork | 733 | 9 | 9 | 6.0 / 21.4 | -20.63 | -22.10 | -20.68 | -22.60 | -22.74 | -36.18 | -29.86 |
| thermal | 17 | 12 | 12 | 13.0 / 17.8 | -25.87 | -25.13 | -26.21 | -25.11 | -25.42 | -34.90 | -33.86 |

### Y = 2020

| source | test pairs | novel pairs | distinct novel | nodes seen / novel | tree_ad3 | form_ad3_depth | hpy3_depth | hpy2 | fnprior_kn | sym0 | term_root |
|---|---|---|---|---|---|---|---|---|---|---|---|
| biomodels | 171 | 1 | 1 | 5.2 / 10.0 | -14.81 | -17.02 | -16.16 | -13.98 | -13.95 | -20.47 | -18.32 |
| biomodels_nc | 794 | 20 | 19 | 15.4 / 20.6 | -16.67 | -19.10 | -20.53 | -21.08 | -21.16 | -37.98 | -48.16 |
| cellml | 118 | 9 | 9 | 10.6 / 20.0 | -18.50 | -18.36 | -19.98 | -20.36 | -20.38 | -32.33 | -39.93 |
| ecocurves | 21 | 7 | 7 | 8.7 / 11.6 | -14.80 | -16.29 | -16.43 | -12.06 | -12.08 | -18.10 | -23.20 |
| jws | 31 | 6 | 2 | 11.1 / 30.7 | -17.98 | -29.38 | -28.80 | -30.53 | -30.53 | -47.73 | -37.49 |
| modeldb | 5258 | 264 | 206 | 16.7 / 39.4 | -30.54 | -34.43 | -37.18 | -41.83 | -41.78 | -63.46 | -66.90 |
| thermal | 3 | 2 | 2 | 13.0 / 15.0 | -21.02 | -20.64 | -19.59 | -21.81 | -21.35 | -32.30 | -31.45 |

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

