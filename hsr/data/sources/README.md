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

## R curve catalogues (`thermal_curves.csv` -> `corpus_thermal.json`, `ecocurves_curves.csv` -> `corpus_ecocurves.json`)

Model functions of R packages for ecological curve fitting, one row per function (or per branch
of a piecewise function), transcribed into sympy syntax in x and verified against the package
(rules `rcurves_eval`, `rcurves_corpus`; `workflow/scripts/rcurves_eval.R`,
`hsr.corpus_sources.curves_build`). Columns: `id`, `package`, `function`, `model`, `variable`,
`params`, `defs` (sub-expressions substituted in order), `formula`, `r_call`, `ranges`,
`reference`, `notes`.

* `thermal`: thermal performance and development-rate curves of **rTPC** 1.1.0 (all 49
  models of `get_model_names()`), **devRate** 0.2.6 (all 37 equations of `devRateEqList`) and the
  leaf temperature-response functions of **photosynthesis** 2.1.5 (Arrhenius, peaked Arrhenius
  (Medlyn) in two parametrisations, Heskel, Kruse, MMRT).
* `ecocurves`: species-area models of **sars** 2.1.1 (21 with a model expression, plus the
  linear model, unverified); predictive-microbiology primary and secondary models of
  **nlsMicrobio** 1.0-0 (the smooth ones: Baranyi, Geeraerd, Albert, Mafart, modified Gompertz,
  cardinal temperature/pH/water-activity models; the piecewise Buchanan, bilinear, trilinear and
  Jameson models are left out); the closed-form growth models of **growthrates** 0.8.5; and the
  growth (von Bertalanffy 1-11, Gompertz 1-5, logistic 1-3, Richards 1-3, Schnute 1-4,
  Schnute-Richards) and stock-recruitment functions (Beverton-Holt 1-4, Ricker 1-3, Shepherd,
  Saila-Lorda) of **FSA** 0.10.1 (increment models with two variables and Pauly's piecewise
  seasonal model are left out); the 19 growth functions of **growthmodels** 1.3.1 (Blumberg,
  Brody, Chapman-Richards, generalised logistic and Richards, Gompertz, Janoschek, logistic,
  log-logistic, Mitscherlich, MMF, monomolecular, negative exponential, Richards, Schnute,
  Stannard, von Bertalanffy, Weibull); the xylem vulnerability curves of **fitplc** 1.2-3.1
  (sigmoidal, Weibull); the non-rectangular hyperbola light response of **photosynthesis**; and
  the 23 concentration-response (ecotoxicology) equations of **bayesnec** 2.1.3.1 (NEC models
  as their branch above the no-effect concentration, ECx models, hormesis models). bayesnec is
  not installed (it needs brms/Stan): its `R/pred_equations.R` is sourced from the CRAN source
  tarball (rule `cran_source_fetch`, version in `config.cran_sources`, sha256 in the manifest).
* A "model" is the original publication (column `model`), not the package function: functions
  of one publication (e.g. Briere-1 and Briere-2; Schoolfield full/high/low; the FSA
  parametrisations of von Bertalanffy) and the same model in two packages (rTPC and devRate) form
  one model, which contributes each distinct form once.
* Verification: `r_call` is evaluated by R with x and the parameters bound; helpers evaluate
  the package's own expression objects (`devRateEqList[[name]]$eq`, the `exp` element of the
  model list in the body of a `sars` function, an `nlsMicrobio` formula's right-hand side). Three
  random parameter draws on 25 points of x; defaults x in [0.3, 4] and parameters in [0.5, 2],
  overridden per row by `ranges` where a function is only defined (or returns a penalty value)
  on a domain, e.g. tmin < x < tmax. A row is verified if at least 5 points with non-zero
  finite R values agree to a relative error below 1e-6, relative to |y| itself (unlike the
  biocat/curvelib check, which divides by max(1, |y|) and so cannot detect errors in small
  values such as Arrhenius factors). All rows with an R implementation pass; non-expressible
  rows (absolute values, cosines, seasonal sines) are listed in `verification` with key null.
* Packages: rTPC (GPL >= 3; Padfield D, O'Sullivan H, Pawar S (2021) Methods Ecol Evol
  12:1138-1143), devRate (GPL-2; Rebaudo F, Struelens Q, Dangles O (2018) Methods Ecol Evol
  9:1144-1150), sars (GPL-3; Matthews TJ et al. (2019) Ecography 42:1446-1455), nlsMicrobio
  (GPL-3; Baty F, Delignette-Muller ML, Siberchicot A), growthrates (GPL >= 2; Petzoldt T), FSA
  (GPL >= 2; Ogle DH, Doll JC, Wheeler AP, Dinno A). Only the formulas are used; each row cites
  the original publication of the model. growthmodels (GPL-3; Oliveira D), fitplc (GPL; Duursma
  R, Choat B (2017) J Plant Hydraul 4:e002), photosynthesis (MIT; Stinziano JR et al. (2021)
  Methods Ecol Evol 12:1094-1102), bayesnec (GPL-2; Fisher R, Barneche DR, Ricardo GF, Fox DR
  (2024) J Stat Softw 110(5)). Versions are recorded in the corpus files.

## nlmixr2lib model library (`corpus_nlmixr2lib.json`)

* Source: the CRAN source tarball of nlmixr2lib (version `config.nlmixr2lib_version`, 0.3.2;
  rule `nlmixr2lib_fetch`, one request; sha256 in the manifest): 59 pharmacometric models
  (one- to three-compartment PK with linear, Michaelis-Menten and time-dependent clearance, transit
  absorption, target-mediated disposition; indirect-response, precursor and circadian PD models;
  tumour-growth inhibition (exponential, linear, power-law, logistic, Gompertz, von Bertalanffy,
  Simeoni, Koch); drug-specific published models; endogenous IgG and phenylalanine models).
* Parsing: rule `nlmixr2lib_statements` reads the `model({...})` block of every model file with
  R's parser (`workflow/scripts/nlmixr2lib_extract.R`; nothing is evaluated, so rxode2 is not
  needed); rule `nlmixr2lib_corpus` (`hsr.corpus_sources.nlmixr2lib_build`) substitutes each
  assignment into the later statements and reduces every assignment and `d/dt()` right-hand side.
  Variables: the states, time and the covariates (identifiers neither assigned in `model()` nor
  defined in `ini()`, e.g. WT, AGE); parameters, random effects and derived quantities are
  constants. Statements with conditionals, `linCmt()` or rxode2-specific functions are skipped
  (13 of 835). A model is one file.
* Terms of use: nlmixr2lib is GPL (>= 2) (Fidler M et al., nlmixr2lib: a model library for
  nlmixr2, https://CRAN.R-project.org/package=nlmixr2lib). Each model file cites its source
  publication. Only derived skeletons and file names are committed.

## allodb tree allometries (`corpus_allodb.json`)

* Source: the allodb repository (rOpenSci; Gonzalez-Akre E et al. (2022) allodb: An R package for
  biomass estimation at globally distributed extratropical forest plots. Methods Ecol Evol
  13:330-338), downloaded as a tarball at the commit in `config.allodb_commit` (rule
  `allodb_fetch`, `codeload.github.com`, no authentication; sha256 in the manifest). Its
  `equations` table holds 570 published allometric equations (biomass, height, bark, foliage, ...)
  as R expression strings, 562 with DBH as the only independent variable.
* Parsing: rule `allodb_equations` (`workflow/scripts/allodb_extract.R`) loads the table and
  evaluates every DBH-only equation with R at DBH = 5, 10, 20, 40, 80 cm; rule `allodb_corpus`
  (`hsr.corpus_sources.allodb_build`) transcribes each string into sympy (`r_infix`) and keeps it
  only if it agrees with R's values (relative error 1e-9; all 562 do), then reduces it with DBH as
  x1. A model is one reference (`ref_id`, 54 publications), contributing each distinct form once.
* Terms of use: allodb is GPL-3. Only derived skeletons and reference ids are committed; cite
  Gonzalez-Akre et al. (2022) and the original references listed in allodb.

## Overlap between sources (`results/corpora/overlap.md`)

Rule `corpus_overlap` tabulates, for every source of the leave-one-source-out comparison, its
models, model-form pairs and distinct generalised forms, the forms found in no other source, the
forms not in the union of the original five sources (BioModels curated, SABIO-RK, CellML,
curvelib, biocat), and the number of forms shared by each pair of sources.
