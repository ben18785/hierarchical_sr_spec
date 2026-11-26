# Paper 1: generated tables and figures

Built by `snakemake --use-conda --cores 1 paper1` (rules `paper1_data` and `paper1_report` in `workflow/Snakefile`; settings in `config/config.yaml`, section `paper1`). Do not edit these files: rerun the target. Tables are booktabs `tabular` fragments: put each inside a `table` float with `\input{...}` and write the caption there. Figures are PDFs for `\includegraphics`. Scores are raw log P per model-form pair in nats; compact tables use the equal source weighting. `data/` holds the JSON extracts the files are drawn from.

Section names refer to `docs/paper1_outline.tex`.

## Main text

- `tab_corpus.tex` (Corpus: sources). The 12 training and 3 hold-out sources: models, model-form pairs, distinct forms, licence and extraction route (details in `hsr/data/sources/README.md`).
- `fig_source_overlap.pdf` (Corpus: overlap). Distinct forms shared between sources (upset plot; the 25 largest of 154 intersections; 2185 distinct training forms).
- `fig_form_sizes.pdf` (Corpus: size distribution). Node count of each distinct form, per source (median and interquartile range; distinct forms in brackets).
- `tab_loso_main.tex` (Results, learned priors beat the baseline: LOSO). Leave one source out: mean over left-out sources of log P per pair, all, seen and novel pairs.
- `tab_holdout_main.tex` (Results, learned priors beat the baseline: hold-out sources). Hold-out sources: log P per pair by source, seen and novel pairs, and the novelty penalty at fixed node count.
- `fig_logp_size.pdf` (Results, why novel forms score low). Log P against node count for seen and novel pairs: hold-out sources (dots: pairs) and the temporal split at Y = 2015; means and 95% intervals per size bin.
- `tab_temporal_main.tex` (Predicting the future literature). Temporal split: log P per test pair, all, seen and novel, per cutoff.
- `tab_temporal_no_modeldb_main.tex` (Predicting the future literature (ModelDB to-do)). As tab_temporal_main with ModelDB test pairs removed.
- `tab_blend_main.tex` (Priors, blended prior (to-do)). Blend of the main adaptor and context priors and the mixture of all tree priors, with their components: LOSO objective, hold-out and temporal scores.
- `tab_mechanisms_main.tex` (Mechanism-derived forms). Mechanism-derived forms: log P per random mechanism, all and novel, per mechanism class.
- `tab_lookalike_main.tex` (The look-alike test). Look-alike test (D = 5 nats): rank and renormalised mass of the truth within the look-alike set, posterior probability of the truth and MAP accuracy, seen and novel truths.

## Supplement

- `tab_loso_by_source_equal.tex, tab_loso_by_source_pooled.tex` (per-source tables). LOSO log P per pair for every left-out source and tree prior.
- `tab_loso_novel_by_source_equal.tex, tab_loso_novel_by_source_pooled.tex` (per-source tables). As above, novel pairs only.
- `tab_holdout_full.tex` (per-source tables). Hold-out log P per pair, every prior (tree and PCFG), both weightings.
- `tab_holdout_seen_novel_full.tex` (per-source tables). Hold-out seen and novel pairs, per distinct form, and renormalised scores.
- `tab_holdout_gap.tex` (Why novel forms score low, or supplement). Why novel forms score low: gap, per-node slope, novelty at fixed size, gain over sym0, non-redundant novel forms.
- `tab_temporal_sets.tex` (Predicting the future literature, or supplement). Temporal test sets: sizes, novel pairs and forms, ModelDB share.
- `tab_temporal_full_equal.tex, tab_temporal_full_pooled.tex` (per-source tables). Temporal log P for every prior, with and without ModelDB.
- `tab_temporal_gap.tex` (Why novel forms score low (temporal to-do)). Seen-novel gap and novelty at fixed size on the temporal split, with and without ModelDB.
- `fig_logp_size_full_equal.pdf, fig_logp_size_full_pooled.pdf` (Why novel forms score low, or supplement). As fig_logp_size for every tree prior, every cutoff and Y = 2015 without ModelDB.
- `tab_blend_full.tex` (supplement). Every blend and component, both weightings, every cutoff.
- `tab_mechanisms_full.tex` (supplement). Mechanism classes: all, seen and novel, every prior.
- `tab_lookalike_full.tex` (supplement). Look-alike test, all datasets, every prior and threshold D.
- `tab_lookalike_strata.tex` (supplement). Renormalised mass on the truth by stratum (D = 5).
- `tab_properness.tex` (supplement, properness). Probability mass on finite trees for every fitted prior (branching-process fixed point at the first and last kept snapshot of every fit); uniform symbols is improper.
- `tab_mcmc.tex` (supplement, MCMC diagnostics). Fit time, change in P(finite) between snapshots, adaptor acceptance and the range of root hyperparameters across fits.
- `fig_adaptor_acceptance.pdf` (supplement, MCMC diagnostics). Adaptor Metropolis-Hastings rejection rate per sweep (median and 10-90% range over fits; traces of the log joint and hyperparameters are not saved).

## Notes

- The LOSO runs predate the current corpus files: their test sets differ from the corpus table by CellML (PMR) -1, ModelDB +1, JWS Online +3 pairs. Rerunning the LOSO fits (on Actions) removes the difference.
- fnprior's properness row is from the calibration fit on curated BioModels only (rule `hpyv_calib`); the 12-source fits of fnprior do not save a termination check.
