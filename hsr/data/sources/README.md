# Equation corpora: sources, snapshots and terms of use

Every corpus file `hsr/data/corpus_<name>.json` has the format of `hsr/data/corpus.json`
(`per_model` {model id: [distinct keys]}, `freq` {key: number of models}, `stats`, `source`).
Keys are canonical one-variable skeletons (`hsr.canonical.canonicalise`) with numeric literals
generalised to free constants (`hsr.grammar_compare.generalise`). Forms our grammars cannot
score are listed under `not_expressible` (or in `verification`) and left out of `per_model`.
Each file is built by Snakemake rules (`workflow/Snakefile`); raw downloads are cached under
`results/corpora/raw/<source>/` (gitignored) with a `manifest.json` of URLs, the download date and
sha256 checksums. The corpus files record the snapshot date and a checksum over the manifest.

Transfer tables (prior trained on all BioModels, scored on the corpus):
`snakemake --use-conda --cores 4 results/grammar/transfer_<name>/table.md`.

## BioModels (`corpus.json`)

Kinetic laws of the manually curated BioModels (built on GitHub Actions, `python -m hsr.corpus build`).
BioModels content is free to use (CC0 for the curated models).

## R curve libraries (`curve_libraries.csv` -> `corpus_curvelib.json`)

Self-start models of R `stats`, `nlraa` and `drc`, transcribed and verified numerically
(rules `curvelib_eval`, `curvelib_corpus`). The packages are GPL; only the formulas are used.

## SABIO-RK (`corpus_sabiork.json`)

* Source: SABIO-RK, HITS gGmbH, https://sabiork.h-its.org, through its export API
  (`https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json`; the legacy
  `sabioRestWebServices` endpoints no longer exist). Rule `sabiork_fetch` downloads every
  kinetic-law entry as JSON in EntryID ranges of 1000 (about 80 requests at most 40 per
  minute; the API allows 60 per minute per IP) and records date, URLs and checksums.
* Parsing (rule `sabiork_corpus`, `hsr.corpus_sources.sabiork_build`): the entry's rate
  equation (`kineticlaw.formula`) is reduced with the BioModels machinery (`hsr.corpus._to_sympy`
  and `project`): each parameter with role `Variable` (a concentration) in turn becomes x1 and
  every other symbol a constant. A model is one SABIO-RK publication: it contributes each of its
  distinct skeletons once, as a BioModels model does. Entries without a rate equation (about half)
  are skipped.
* The JSON export is used rather than SBML: the SBML export's kinetic-law MathML refers to
  parameter names while the local parameters have different ids, so the species/parameter roles
  are not recoverable from the SBML alone; the JSON states each parameter's role explicitly.
* Terms of use: SABIO-RK's "Non-Commercial Purpose License" (https://sabiork.h-its.org/ui/terms):
  use is granted without fee for internal non-commercial research and academic purposes; commercial
  use requires a licence from HITS. The committed corpus contains only derived, abstracted
  skeletons and publication ids, no SABIO-RK records. Cite: Wittig U et al. (2018) SABIO-RK:
  an updated resource for manually curated biochemical reaction kinetics. Nucleic Acids Res
  46:D656-D660.

## CellML / Physiome Model Repository (`corpus_cellml.json`)

* Source: Physiome Model Repository, https://models.physiomeproject.org. Rule `cellml_fetch`
  lists all exposures (`/exposure`, PMR2 JSON API), reads each exposure's workspace and changeset,
  and downloads each workspace once as an archive at that changeset
  (`<workspace>/@@archive/<commit>/tgz`), keeping only `.cellml` files (about 2 requests per
  exposure, at most 40 per minute). The generated OMEX archives offered by PMR were empty when
  tried, so they are not used.
* Parsing (rule `cellml_equations`, `workflow/scripts/cellml_extract.py`, own conda environment
  with libcellml 0.7.2): every CellML file is parsed (CellML 1.0/1.1 translated to 2.0), its
  imports are resolved from the workspace files and flattened, the model is analysed and libcellml's
  code generator writes it as Python; each assignment of `compute_rates`/`compute_variables` is
  an equation. One process per file with a time limit. Rule `cellml_corpus` reduces each right-hand
  side as for SABIO-RK: states, algebraic variables and the variable of integration are variables,
  constants and computed constants are constants. A model is one workspace.
* Failures are recorded per file (`results/corpora/cellml_equations.json`): files that are only
  component libraries, models that do not validate in libcellml (e.g. a variable of integration
  with an initial value, units used but not imported), and equations with piecewise definitions
  or functions outside `hsr.corpus._FUNCS` (which give no key).
* Terms of use: the PMR software is GPL/LGPL/MPL. Licences of the models are set per workspace
  or exposure in PMR (the curated CellML models are generally distributed under Creative Commons
  Attribution); none of the downloaded .cellml files states a licence itself, and the licence of
  each model has not been checked one by one. The committed corpus contains only derived,
  abstracted skeletons and workspace ids, no model files. Cite: Yu T et al. (2011) The Physiome Model Repository 2.
  Bioinformatics 27:743-744; Clerx M et al. (2020) CellML 2.0. J Integr Bioinform 17:20200021;
  libCellML, https://libcellml.org.

## Curated catalogue (`bio_catalogue.csv` -> `corpus_biocat.json`)

Pharmacokinetic solutions (one-, two- and three-compartment; bolus, infusion, first-order
absorption), pharmacodynamic models (linear, log-linear, Emax, sigmoid Emax, Imax), enzyme and
microbial kinetics (Michaelis-Menten, Hill, Haldane/Andrews, inhibition as a function of the
inhibitor, Monod, Teissier, Moser, Contois, Aiba), functional responses (Holling I-III, Hassell
type III, Real, Rogers, Ivlev, Beddington-DeAngelis, Crowley-Martin, Hassell-Varley and
Arditi-Ginzburg reduced to one variable), growth (exponential, logistic, three Gompertz
parametrisations, von Bertalanffy length and weight, Richards, Chapman-Richards, monomolecular,
Weibull), allometry, temperature (Arrhenius, Q10) and stock-recruitment (Ricker, Beverton-Holt,
Shepherd), one row per form with a reference. Columns: `variable` (what x is), `params`, `defs`
(sub-expressions substituted in order, e.g. the hybrid rate constants of the two- and
three-compartment solutions), `formula` (Python/sympy syntax in x), `check`, `r_call`.

* `check = library`: verified against an R implementation (rule `biocat_eval`): `linpk`
  (linear compartmental PK by matrix exponential, from the micro rate constants), `drc`
  (MM.2, MM.3, LL.3, LL.4, AR.2), `frair` (functional responses) or `FSA` (growth and
  stock-recruitment functions).
* `check = ode`: no R implementation of that parametrisation; the closed form is checked
  against its defining differential equation integrated with `deSolve` (lsoda, rtol 1e-11) from
  the first grid point where it is non-negligible.
* `check = none`: no independent implementation; algebra checked by hand against the reference,
  stored with `verified: false`.

Verification uses three random parameter draws in [0.5, 2] on 25 points of x in [0.3, 4] and
requires at least 5 finite points with maximum relative error below 1e-6 (as for the curve
libraries). R packages are installed in `config.r_lib`; their versions are recorded in
`corpus_biocat.json`. The forms are standard textbook results, not copyrightable data.
