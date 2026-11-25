# Blended priors

P_w(f) = sum_k w_k P_k(f) over tree priors, computed per form as logsumexp_k(log w_k + log P_k(f)) (hsr.blend). Every tree prior is proper over the one shared space of coded trees (docs/prior_models.tex), so every blend is proper. For two components P_w = w P_A + (1 - w) P_B and w is the weight of the first.

Selection uses the leave-one-source-out runs only: for each source weighting, w maximises the mean over the 12 left-out sources of the mean raw log P_w per model-form pair of that source (each source counts equally), with the priors trained under that weighting. Two components: grid over w in [0, 1] (spacing 0.005, plus spacing 0.05 in logit(w) out to |logit w| = 12); three or more: EM (the objective is concave in w, so both reach the global maximum; the EM fit of every pair agrees to within the grid spacing with its grid choice, see blend.json `loso_em_check`). Jackknife: w re-chosen with each LOSO source left out of the objective in turn. The chosen w is then applied, untouched, to the hold-out sources (holdout.md) and the temporal hold-out (temporal.md), with the priors trained under the same weighting. "all tree priors": EM mixture of every tree prior. Built by `snakemake --cores 1 blend`.

## Weights chosen on LOSO

LOSO columns: mean over left-out sources of mean raw log P per pair (the selection objective).

| blend | weighting | chosen w | jackknife w range | LOSO first component | LOSO second component | LOSO blend | gain over best component |
|---|---|---|---|---|---|---|---|
| tree_ad3 + hpy3_depth | equal | 0.715 | 0.690 - 0.755 | -8.51 | -9.25 | -8.18 | 0.33 |
| tree_ad3 + hpy3_depth | pooled | 0.800 | 0.775 - 0.835 | -8.36 | -9.39 | -8.23 | 0.12 |
| tree_ad3 + hpy2 | equal | 0.735 | 0.711 - 0.785 | -8.51 | -9.66 | -8.00 | 0.51 |
| tree_ad3 + hpy2 | pooled | 0.839 | 0.820 - 0.870 | -8.36 | -10.14 | -8.16 | 0.20 |
| tree_ad3 + fnprior_kn | equal | 0.735 | 0.711 - 0.785 | -8.51 | -9.66 | -8.00 | 0.50 |
| tree_ad3 + fnprior_kn | pooled | 0.839 | 0.825 - 0.870 | -8.36 | -10.14 | -8.17 | 0.19 |
| form_ad3_depth + hpy3_depth | equal | 0.720 | 0.695 - 0.750 | -8.48 | -9.25 | -8.24 | 0.24 |
| form_ad3_depth + hpy3_depth | pooled | 0.840 | 0.815 - 0.880 | -8.34 | -9.39 | -8.30 | 0.04 |
| form_ad3_depth + hpy2 | equal | 0.731 | 0.705 - 0.780 | -8.48 | -9.66 | -8.09 | 0.39 |
| form_ad3_depth + hpy2 | pooled | 0.865 | 0.852 - 0.895 | -8.34 | -10.14 | -8.24 | 0.10 |
| form_ad3_depth + fnprior_kn | equal | 0.730 | 0.705 - 0.780 | -8.48 | -9.66 | -8.10 | 0.39 |
| form_ad3_depth + fnprior_kn | pooled | 0.870 | 0.855 - 0.896 | -8.34 | -10.14 | -8.24 | 0.10 |
| tree_ad3 + sym0 | equal | 0.989 | 0.987 - 0.993 | -8.51 | -17.63 | -8.48 | 0.02 |
| tree_ad3 + sym0 | pooled | 0.994 | 0.994 - 0.999 | -8.36 | -18.06 | -8.35 | 0.01 |
| tree_ad3 + hpy3_depth + fnprior_kn | equal | tree_ad3 0.656, fnprior_kn 0.182, hpy3_depth 0.163 | see blend.json |  |  | -7.93 | 0.57 |
| tree_ad3 + hpy3_depth + fnprior_kn | pooled | tree_ad3 0.758, hpy3_depth 0.123, fnprior_kn 0.119 | see blend.json |  |  | -8.12 | 0.23 |
| tree_ad3 + form_ad3_depth + hpy3_depth | equal | tree_ad3 0.452, form_ad3_depth 0.298, hpy3_depth 0.250 | see blend.json |  |  | -8.11 | 0.37 |
| tree_ad3 + form_ad3_depth + hpy3_depth | pooled | tree_ad3 0.477, form_ad3_depth 0.388, hpy3_depth 0.135 | see blend.json |  |  | -8.17 | 0.17 |
| all tree priors | equal | tree_ad3 0.533, hpy3_depth 0.151, hpy 0.121, form_ad3_depth 0.102, hpy2_src 0.079, form_ad 0.011, sym0 0.003 | see blend.json |  |  | -7.88 | 0.60 |
| all tree priors | pooled | tree_ad3 0.575, form_ad3_depth 0.177, hpy2_src 0.121, hpy3_depth 0.081, hpy 0.031, hpy2 0.014, fnprior_kn_tuned 0.002 | see blend.json |  |  | -8.02 | 0.32 |

Tree priors alone on LOSO (same objective): form_ad3_depth -8.48 / -8.34; tree_ad3 -8.51 / -8.36; form_ad -8.69 / -8.56; hpy3_depth -9.25 / -9.39; hpy -9.65 / -10.14; hpy2 -9.66 / -10.14; fnprior_kn -9.66 / -10.14; fnprior_kn_tuned -9.69 / -10.15; hpy2_src -10.37 / -10.22; sym0 -17.63 / -18.06 (equal / pooled).

## Hold-out sources

All hold-out pairs pooled (holdout.md "all"): mean raw log P per pair on all / seen / novel pairs, and the novelty penalty at fixed node count (OLS coefficient of the novel indicator; negative = novel forms score lower than seen forms of the same size). Blends use the LOSO weights above.

### equal weighting

| prior | weights | all | seen | novel | novel penalty |
|---|---|---|---|---|---|
| tree_ad3 |  | -6.31 | -4.23 | -25.96 | -15.48 |
| hpy3_depth |  | -7.04 | -5.19 | -24.52 | -9.44 |
| hpy2 |  | -7.88 | -6.29 | -22.88 | -6.59 |
| fnprior_kn |  | -7.91 | -6.29 | -23.14 | -6.72 |
| form_ad3_depth |  | -6.26 | -4.23 | -25.43 | -15.34 |
| sym0 |  | -15.17 | -13.52 | -30.70 | -2.31 |
| tree_ad3 + hpy3_depth | 0.715 | -6.17 | -4.31 | -23.69 | -12.79 |
| tree_ad3 + hpy2 | 0.735 | -6.17 | -4.38 | -23.12 | -12.39 |
| tree_ad3 + fnprior_kn | 0.735 | -6.19 | -4.38 | -23.30 | -12.46 |
| form_ad3_depth + hpy3_depth | 0.720 | -6.13 | -4.31 | -23.31 | -12.82 |
| form_ad3_depth + hpy2 | 0.731 | -6.16 | -4.38 | -22.99 | -12.36 |
| form_ad3_depth + fnprior_kn | 0.730 | -6.17 | -4.38 | -23.04 | -12.38 |
| tree_ad3 + sym0 | 0.989 | -6.29 | -4.24 | -25.69 | -15.23 |
| tree_ad3 + hpy3_depth + fnprior_kn | tree_ad3 0.656, fnprior_kn 0.182, hpy3_depth 0.163 | -6.13 | -4.38 | -22.72 | -11.79 |
| tree_ad3 + form_ad3_depth + hpy3_depth | tree_ad3 0.452, form_ad3_depth 0.298, hpy3_depth 0.250 | -6.12 | -4.30 | -23.28 | -12.78 |
| all tree priors | tree_ad3 0.533, hpy3_depth 0.151, hpy 0.121, form_ad3_depth 0.102, hpy2_src 0.079, form_ad 0.011, sym0 0.003 | -6.09 | -4.39 | -22.18 | -11.54 |

### pooled weighting

| prior | weights | all | seen | novel | novel penalty |
|---|---|---|---|---|---|
| tree_ad3 |  | -6.38 | -4.53 | -23.79 | -12.89 |
| hpy3_depth |  | -7.40 | -5.67 | -23.77 | -7.98 |
| hpy2 |  | -8.24 | -6.61 | -23.66 | -6.91 |
| fnprior_kn |  | -8.26 | -6.61 | -23.80 | -7.03 |
| form_ad3_depth |  | -6.33 | -4.53 | -23.33 | -12.70 |
| sym0 |  | -15.34 | -13.63 | -31.49 | -2.65 |
| tree_ad3 + hpy3_depth | 0.800 | -6.36 | -4.59 | -23.02 | -11.87 |
| tree_ad3 + hpy2 | 0.839 | -6.38 | -4.61 | -23.05 | -11.92 |
| tree_ad3 + fnprior_kn | 0.839 | -6.38 | -4.61 | -23.05 | -11.93 |
| form_ad3_depth + hpy3_depth | 0.840 | -6.33 | -4.58 | -22.82 | -12.02 |
| form_ad3_depth + hpy2 | 0.865 | -6.33 | -4.60 | -22.66 | -11.82 |
| form_ad3_depth + fnprior_kn | 0.870 | -6.33 | -4.60 | -22.66 | -11.83 |
| tree_ad3 + sym0 | 0.994 | -6.37 | -4.54 | -23.70 | -12.80 |
| tree_ad3 + hpy3_depth + fnprior_kn | tree_ad3 0.758, hpy3_depth 0.123, fnprior_kn 0.119 | -6.36 | -4.63 | -22.71 | -11.49 |
| tree_ad3 + form_ad3_depth + hpy3_depth | tree_ad3 0.477, form_ad3_depth 0.388, hpy3_depth 0.135 | -6.31 | -4.57 | -22.71 | -11.90 |
| all tree priors | tree_ad3 0.575, form_ad3_depth 0.177, hpy2_src 0.121, hpy3_depth 0.081, hpy 0.031, hpy2 0.014, fnprior_kn_tuned 0.002 | -6.30 | -4.65 | -21.82 | -10.95 |

## Temporal hold-out

Per cutoff Y (temporal.md): all / seen / novel / novelty penalty at fixed size. Blends use the LOSO weights above.

### equal weighting

| prior | 2010 all | 2010 seen | 2010 novel | 2010 penalty | 2015 all | 2015 seen | 2015 novel | 2015 penalty | 2020 all | 2020 seen | 2020 novel | 2020 penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tree_ad3 | -8.05 | -5.40 | -29.97 | -17.37 | -7.82 | -5.73 | -32.69 | -20.56 | -7.33 | -6.19 | -30.30 | -18.84 |
| hpy3_depth | -12.27 | -9.52 | -34.93 | -5.99 | -12.99 | -10.89 | -37.93 | -6.63 | -13.54 | -12.32 | -38.04 | -6.53 |
| hpy2 | -14.65 | -11.83 | -37.87 | -1.54 | -15.74 | -13.53 | -41.90 | -2.57 | -16.60 | -15.37 | -41.33 | -1.87 |
| fnprior_kn | -14.65 | -11.84 | -37.91 | -1.58 | -15.74 | -13.53 | -41.91 | -2.59 | -16.60 | -15.37 | -41.31 | -1.85 |
| form_ad3_depth | -8.27 | -5.39 | -32.07 | -18.52 | -7.94 | -5.73 | -34.22 | -21.30 | -7.51 | -6.19 | -34.17 | -21.40 |
| sym0 | -24.19 | -20.54 | -54.29 | -1.83 | -25.78 | -22.98 | -58.97 | -2.11 | -27.30 | -25.65 | -60.51 | -1.85 |
| tree_ad3 + hpy3_depth | -8.08 | -5.52 | -29.27 | -16.34 | -7.89 | -5.87 | -31.95 | -19.52 | -7.45 | -6.35 | -29.48 | -17.69 |
| tree_ad3 + hpy2 | -8.09 | -5.53 | -29.24 | -16.28 | -7.90 | -5.87 | -31.99 | -19.50 | -7.44 | -6.35 | -29.43 | -17.64 |
| tree_ad3 + fnprior_kn | -8.09 | -5.53 | -29.25 | -16.29 | -7.90 | -5.87 | -32.00 | -19.51 | -7.44 | -6.35 | -29.45 | -17.65 |
| form_ad3_depth + hpy3_depth | -8.31 | -5.51 | -31.40 | -17.51 | -8.03 | -5.86 | -33.75 | -20.49 | -7.63 | -6.34 | -33.47 | -20.37 |
| form_ad3_depth + hpy2 | -8.32 | -5.52 | -31.45 | -17.53 | -8.04 | -5.87 | -33.77 | -20.51 | -7.62 | -6.35 | -33.15 | -20.08 |
| form_ad3_depth + fnprior_kn | -8.33 | -5.52 | -31.46 | -17.54 | -8.04 | -5.87 | -33.78 | -20.51 | -7.62 | -6.35 | -33.17 | -20.09 |
| tree_ad3 + sym0 | -8.06 | -5.40 | -29.96 | -17.35 | -7.83 | -5.73 | -32.69 | -20.54 | -7.34 | -6.20 | -30.26 | -18.77 |
| tree_ad3 + hpy3_depth + fnprior_kn | -8.08 | -5.55 | -28.96 | -15.91 | -7.91 | -5.90 | -31.65 | -19.11 | -7.47 | -6.39 | -29.16 | -17.27 |
| tree_ad3 + form_ad3_depth + hpy3_depth | -8.05 | -5.50 | -29.10 | -16.28 | -7.85 | -5.84 | -31.59 | -19.25 | -7.42 | -6.32 | -29.31 | -17.59 |
| all tree priors | -8.05 | -5.56 | -28.60 | -15.64 | -7.87 | -5.91 | -31.06 | -18.57 | -7.47 | -6.40 | -28.91 | -17.03 |

### pooled weighting

| prior | 2010 all | 2010 seen | 2010 novel | 2010 penalty | 2015 all | 2015 seen | 2015 novel | 2015 penalty | 2020 all | 2020 seen | 2020 novel | 2020 penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tree_ad3 | -7.42 | -4.90 | -28.22 | -16.60 | -7.17 | -5.16 | -31.01 | -19.89 | -6.61 | -5.52 | -28.58 | -18.16 |
| hpy3_depth | -10.98 | -8.47 | -31.70 | -6.86 | -11.67 | -9.72 | -34.74 | -7.35 | -12.10 | -10.97 | -34.79 | -7.11 |
| hpy2 | -13.91 | -11.25 | -35.89 | -2.18 | -14.87 | -12.81 | -39.33 | -2.95 | -15.45 | -14.29 | -38.74 | -2.87 |
| fnprior_kn | -13.91 | -11.25 | -35.92 | -2.21 | -14.87 | -12.81 | -39.38 | -3.01 | -15.44 | -14.29 | -38.71 | -2.84 |
| form_ad3_depth | -7.65 | -4.90 | -30.34 | -17.78 | -7.31 | -5.16 | -32.71 | -20.75 | -6.79 | -5.52 | -32.31 | -20.62 |
| sym0 | -24.04 | -20.34 | -54.54 | -2.43 | -25.45 | -22.66 | -58.53 | -2.48 | -26.70 | -25.08 | -59.23 | -2.26 |
| tree_ad3 + hpy3_depth | -7.45 | -4.98 | -27.79 | -16.02 | -7.22 | -5.26 | -30.46 | -19.20 | -6.69 | -5.63 | -27.97 | -17.37 |
| tree_ad3 + hpy2 | -7.47 | -4.98 | -27.95 | -16.16 | -7.23 | -5.26 | -30.62 | -19.31 | -6.70 | -5.63 | -28.21 | -17.61 |
| tree_ad3 + fnprior_kn | -7.47 | -4.98 | -27.96 | -16.16 | -7.23 | -5.26 | -30.64 | -19.34 | -6.70 | -5.63 | -28.22 | -17.62 |
| form_ad3_depth + hpy3_depth | -7.67 | -4.96 | -30.05 | -17.35 | -7.35 | -5.24 | -32.47 | -20.34 | -6.85 | -5.60 | -31.91 | -20.06 |
| form_ad3_depth + hpy2 | -7.70 | -4.97 | -30.21 | -17.50 | -7.37 | -5.24 | -32.59 | -20.48 | -6.86 | -5.61 | -32.05 | -20.21 |
| form_ad3_depth + fnprior_kn | -7.69 | -4.97 | -30.21 | -17.51 | -7.37 | -5.24 | -32.60 | -20.49 | -6.86 | -5.60 | -32.05 | -20.22 |
| tree_ad3 + sym0 | -7.42 | -4.90 | -28.21 | -16.59 | -7.18 | -5.17 | -31.01 | -19.89 | -6.62 | -5.52 | -28.55 | -18.13 |
| tree_ad3 + hpy3_depth + fnprior_kn | -7.47 | -5.01 | -27.70 | -15.86 | -7.24 | -5.29 | -30.30 | -18.97 | -6.72 | -5.67 | -27.90 | -17.24 |
| tree_ad3 + form_ad3_depth + hpy3_depth | -7.41 | -4.95 | -27.72 | -16.04 | -7.17 | -5.22 | -30.24 | -19.06 | -6.65 | -5.59 | -27.92 | -17.39 |
| all tree priors | -7.44 | -5.03 | -27.36 | -15.55 | -7.21 | -5.31 | -29.78 | -18.47 | -6.73 | -5.69 | -27.72 | -17.04 |

## Oracle weights (for reference only; NOT a valid selection)

Weights maximising the mean raw log P over all pairs of each test set itself, and that maximum minus the score at the LOSO-chosen weights (the most any re-tuning on the test set could gain). Two components: weight of the first.

| blend | weighting | LOSO w | hold-out oracle w (gain) | temporal 2010 oracle w (gain) | temporal 2015 oracle w (gain) | temporal 2020 oracle w (gain) |
|---|---|---|---|---|---|---|
| tree_ad3 + hpy3_depth | equal | 0.715 | 0.830 (0.01) | 0.950 (0.07) | 0.971 (0.09) | 0.984 (0.13) |
| tree_ad3 + hpy3_depth | pooled | 0.800 | 0.917 (0.01) | 0.969 (0.04) | 0.981 (0.06) | 0.991 (0.08) |
| tree_ad3 + hpy2 | equal | 0.735 | 0.875 (0.03) | 0.950 (0.08) | 0.968 (0.10) | 0.981 (0.13) |
| tree_ad3 + hpy2 | pooled | 0.839 | 0.948 (0.02) | 0.985 (0.06) | 0.987 (0.06) | 0.996 (0.09) |
| tree_ad3 + fnprior_kn | equal | 0.735 | 0.880 (0.03) | 0.953 (0.08) | 0.968 (0.10) | 0.982 (0.13) |
| tree_ad3 + fnprior_kn | pooled | 0.839 | 0.948 (0.02) | 0.985 (0.06) | 0.988 (0.07) | 0.996 (0.09) |
| form_ad3_depth + hpy3_depth | equal | 0.720 | 0.820 (0.01) | 0.945 (0.06) | 0.978 (0.10) | 0.987 (0.12) |
| form_ad3_depth + hpy3_depth | pooled | 0.840 | 0.928 (0.01) | 0.979 (0.03) | 0.994 (0.05) | 0.995 (0.06) |
| form_ad3_depth + hpy2 | equal | 0.731 | 0.890 (0.04) | 0.953 (0.08) | 0.979 (0.11) | 0.983 (0.13) |
| form_ad3_depth + hpy2 | pooled | 0.865 | 0.950 (0.02) | 0.996 (0.05) | 0.998 (0.06) | 0.998 (0.07) |
| form_ad3_depth + fnprior_kn | equal | 0.730 | 0.891 (0.04) | 0.955 (0.08) | 0.979 (0.12) | 0.983 (0.13) |
| form_ad3_depth + fnprior_kn | pooled | 0.870 | 0.950 (0.02) | 0.996 (0.05) | 0.998 (0.06) | 0.998 (0.07) |
| tree_ad3 + sym0 | equal | 0.989 | 0.995 (0.00) | 1.000 (0.00) | 1.000 (0.01) | 0.999 (0.00) |
| tree_ad3 + sym0 | pooled | 0.994 | 0.998 (0.00) | 1.000 (0.00) | 1.000 (0.00) | 1.000 (0.00) |
| tree_ad3 + hpy3_depth + fnprior_kn | equal | tree_ad3 0.656, fnprior_kn 0.182, hpy3_depth 0.163 | tree_ad3 0.812, hpy3_depth 0.121, fnprior_kn 0.067 (0.03) | tree_ad3 0.931, hpy3_depth 0.037, fnprior_kn 0.033 (0.09) | tree_ad3 0.955, fnprior_kn 0.024, hpy3_depth 0.021 (0.12) | tree_ad3 0.976, hpy3_depth 0.012, fnprior_kn 0.011 (0.16) |
| tree_ad3 + hpy3_depth + fnprior_kn | pooled | tree_ad3 0.758, hpy3_depth 0.123, fnprior_kn 0.119 | tree_ad3 0.903, hpy3_depth 0.064, fnprior_kn 0.033 (0.03) | tree_ad3 0.965, hpy3_depth 0.027, fnprior_kn 0.009 (0.07) | tree_ad3 0.976, hpy3_depth 0.016, fnprior_kn 0.008 (0.08) | tree_ad3 0.991, hpy3_depth 0.008, fnprior_kn 0.001 (0.12) |
| tree_ad3 + form_ad3_depth + hpy3_depth | equal | tree_ad3 0.452, form_ad3_depth 0.298, hpy3_depth 0.250 | form_ad3_depth 0.462, tree_ad3 0.370, hpy3_depth 0.168 (0.01) | tree_ad3 0.718, form_ad3_depth 0.242, hpy3_depth 0.039 (0.06) | tree_ad3 0.632, form_ad3_depth 0.352, hpy3_depth 0.016 (0.09) | tree_ad3 0.748, form_ad3_depth 0.241, hpy3_depth 0.010 (0.11) |
| tree_ad3 + form_ad3_depth + hpy3_depth | pooled | tree_ad3 0.477, form_ad3_depth 0.388, hpy3_depth 0.135 | form_ad3_depth 0.477, tree_ad3 0.459, hpy3_depth 0.064 (0.01) | tree_ad3 0.729, form_ad3_depth 0.263, hpy3_depth 0.008 (0.04) | tree_ad3 0.654, form_ad3_depth 0.345, hpy3_depth 0.001 (0.05) | tree_ad3 0.753, form_ad3_depth 0.246, hpy3_depth 0.001 (0.06) |
| all tree priors | equal | tree_ad3 0.533, hpy3_depth 0.151, hpy 0.121, form_ad3_depth 0.102, hpy2_src 0.079, form_ad 0.011, sym0 0.003 | tree_ad3 0.379, form_ad 0.241, form_ad3_depth 0.193, hpy3_depth 0.125, hpy 0.056, sym0 0.004, hpy2_src 0.002 (0.05) | tree_ad3 0.801, form_ad3_depth 0.124, hpy2_src 0.031, hpy3_depth 0.029, hpy 0.011, form_ad 0.005 (0.10) | tree_ad3 0.710, form_ad3_depth 0.213, form_ad 0.039, hpy2_src 0.020, hpy3_depth 0.011, hpy2 0.005, hpy 0.002 (0.13) | tree_ad3 0.751, form_ad3_depth 0.230, hpy2 0.010, hpy3_depth 0.008, hpy2_src 0.001, sym0 0.001 (0.18) |
| all tree priors | pooled | tree_ad3 0.575, form_ad3_depth 0.177, hpy2_src 0.121, hpy3_depth 0.081, hpy 0.031, hpy2 0.014, fnprior_kn_tuned 0.002 | tree_ad3 0.575, form_ad3_depth 0.321, hpy2_src 0.048, hpy3_depth 0.042, fnprior_kn 0.014 (0.03) | tree_ad3 0.778, form_ad3_depth 0.158, form_ad 0.039, hpy2_src 0.016, hpy3_depth 0.010 (0.08) | tree_ad3 0.722, form_ad3_depth 0.235, form_ad 0.030, hpy2_src 0.011, hpy3_depth 0.002 (0.10) | tree_ad3 0.769, form_ad3_depth 0.228, hpy3_depth 0.001, hpy2_src 0.001 (0.14) |
