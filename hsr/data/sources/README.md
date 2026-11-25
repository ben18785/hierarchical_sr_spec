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

Kinetic laws of the manually curated BioModels (rule `biomodels_corpus`, `python -m hsr.corpus build`,
run locally since EBI blocks GitHub-hosted runners; ids `config.biomodels_first`..`biomodels_last`).
BioModels content is free to use (CC0 for the curated models).

* Fixed 2026-11-24: the build downloaded `<id>?filename=<id>_url.xml` only. That file exists for
  older entries only; for most curated models from about BIOMD0000000700 on, the main SBML file
  carries the name of the submitted file (e.g. MODEL1712050009.xml) and the request failed
  (HTTP 400), so 403 of BIOMD1-1100 were missing from the corpus. `hsr.corpus.download_model`
  now falls back to the file named in the model's record (`/<id>?format=json`). Also fixed: laws
  containing `pi` gave no key (see `hsr.corpus.project`).

## BioModels non-curated (`corpus_biomodels_nc.json`)

* Source: every non-curated SBML entry of BioModels (search `curationstatus:"Non-curated" AND
  modelformat:"SBML"`, 1474 entries on 2026-11-24; path2models entries are not in this set).
  Rule `biomodels_nc_fetch` (`hsr.corpus_sources.biomodels_fetch`) pages through the search
  API, reads each model's record (main file name and the server's sha256) and downloads the main
  file, rejecting files whose checksum differs from the server's (<= 40 requests per minute).
* Parsing: rule `biomodels_nc_corpus` (`biomodels_build`) reduces kinetic laws exactly as the
  curated corpus (`hsr.corpus.model_keys`), keys generalised; one model per entry. Non-curated
  entries include genome-scale and constraint-based models without kinetic laws (no keys) and
  some models that also exist, curated, under a BIOMD id.
* Terms of use: BioModels states that all models are available under CC0 (public domain).
  Cite: Malik-Sheriff RS et al. (2020) BioModels - 15 years of sharing computational models in
  life science. Nucleic Acids Res 48:D407-D415.

## ModelDB NMODL (`corpus_modeldb.json`)

* Source: ModelDB (https://modeldb.science): the models whose modelling application is NEURON
  (`/api/v1/models?modeling_application=NEURON`), each downloaded as its zip (`/download/<id>`;
  rule `modeldb_fetch`, <= 40 requests per minute), keeping only the `.mod` (NMODL) files; the
  manifest records each zip's URL, size and sha256.
* Parsing (rule `modeldb_equations`, `hsr.nmodl`): a light NMODL parser reads the FUNCTION,
  PROCEDURE, BREAKPOINT, DERIVATIVE and INITIAL blocks; every assignment (`x = e`, `x' = e`) is an
  equation, with LOCAL variables substituted and calls of the file's own FUNCTIONs inlined (a
  function's value is its last assignment to its name, i.e. the `else` branch of guards such as
  `vtrap`); unit annotations after numbers (`1 (mV)`) are dropped; both branches of `if`/`else`
  appear as equations. Variables: `v`, `t`, STATE variables, ion concentrations READ, and the
  arguments of the enclosing FUNCTION/PROCEDURE; PARAMETERs, ASSIGNED variables and `celsius`
  are constants. Assignments with comparisons, arrays or unknown functions are skipped (counted
  in `stats.assignments_not_parsed`); identical files are parsed once. Rule `modeldb_corpus`
  reduces each equation as for CellML; a model is one ModelDB entry.
* Terms of use: ModelDB asks users of its models, or of information derived from them, to cite
  ModelDB (McDougal RA et al. (2017) Twenty years of ModelDB and beyond. J Comput Neurosci
  42:1-10) and the papers describing the models (accession numbers are the model ids in
  `per_model`). The model code is the authors'; only derived skeletons and accession numbers are
  committed.

## JWS Online (`corpus_jws.json`)

* Source: JWS Online model database (https://jjj.bio.vu.nl): the model list
  (`/rest/models/`) and the SBML export of each kinetic model (`/models/<slug>/sbml/`;
  constraint-based models are skipped). Rule `jws_fetch`, <= 40 requests per minute; manifest of
  URLs and sha256; rule `jws_corpus` reduces as BioModels. A model is one JWS entry.
* Many JWS models are also in BioModels (JWS curates models for BioModels), so this source
  overlaps the BioModels corpora; the overlap is reported in `results/corpora/overlap.md`.
* Terms of use: the JWS Online site states no licence for the model files; they are published
  for reuse in research. Only derived skeletons and model slugs are committed. Cite: Peters M
  et al. (2017) JWS Online: a reproducible database of kinetic models. Bioinformatics
  33:1589-1590 (or Olivier BG, Snoep JL (2004) Bioinformatics 20:2143).

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
  tarball (rule `cran_source_fetch`, version in `config.cran_sources`, sha256 in the manifest);
  and the dose-response models of **DoseFinding** 1.4-2 (Emax, sigmoid Emax, exponential,
  quadratic, linear-in-log, logistic, beta).
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
  (2024) J Stat Softw 110(5)), DoseFinding (GPL-3; Bornkamp B, Pinheiro J, Bretz F). Versions are
  recorded in the corpus files.

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

# Hold-out sources

Three further sources are hold-outs (config `holdout_sources`,
`hsr.grammar_compare.HOLDOUT_SOURCES`): they are never part of `SOURCES`, so no prior
(`hsr.fixed_prior`, the LOSO runs) trains on them. The `holdout` target trains every prior on all
twelve training sources (equal and pooled weighting) and scores each hold-out and their union
(`results/grammar/holdout.md`; the command, with `--allowed-rules` so that the committed corpus
files are used as they are, is in the Snakefile). Their rules are in `workflow/rules/<name>.smk`
and their settings in `config/<name>.yaml`.

## Hold-out: nonlinear fits in CRAN package code (`corpus_rnls.json`)

* Source: the source tarballs of CRAN packages in five task views that cover biological and
  ecological modelling: Agriculture (version 2026-10-05), Environmetrics (2023-12-18),
  Epidemiology (2025-03-03), Pharmacokinetics (2026-01-30) and Tracking (animal movement,
  2025-02-12), as listed in CRAN's task-view index `src/contrib/Views.rds` downloaded on
  2026-11-25 (`config/rnls.yaml`; rule `rnls_index`, which also downloads the package database
  `web/packages/packages.rds`; both with sha256 in the manifest). Hydrology is left out
  (physical rather than biological); Phylogenetics, Omics and ChemPhys rarely fit regression
  curves of one regressor.
* Bio/eco filter (rule `rnls_packages`, `hsr.rnls.select_packages`): of the 495 packages in these
  views, a package is kept if its CRAN Title or Description contains one of the keywords in
  `hsr.rnls.BIO_KEYWORDS` (word-start match: ecolog, species, population, fish, forest, plant,
  crop, soil, epidemi, disease, infect, pharmacokinet, drug, dose, growth curve, ...) and is not
  in `config rnls.exclude` (randomForest, rpart, party, maptree, VGAM, nsRFA, eyetrackingR,
  Rbeast: matched only through "tree"/"forest"/"Fisher"/"growth curve" in a non-biological
  sense, or hydrological/time-series tools). General statistical infrastructure (nlme, lme4,
  mgcv, MASS, deSolve, ...) is thereby dropped. 354 packages are kept; the list with the pinned
  version, licence, views and the matched keyword is committed as
  `hsr/data/sources/rnls_packages.tsv`. Rule `rnls_fetch` downloads each listed version
  (CRAN, falling back to the CRAN archive; <= 40 requests per minute; manifest with URL, date and
  sha256 of every tarball, cached under `results/corpora/raw/rnls/`).
* Parsing (rule `rnls_extract`, `workflow/scripts/rnls_extract.R`, R 4.5.3 of `envs/r.yaml`;
  nothing is evaluated): R's parser reads R/, tests/, demo/, inst/ (including vignette code
  purled into inst/doc), the code chunks of vignettes (Rmd, qmd, Rnw) and the `\examples` of
  man/*.Rd (`tools::Rd2ex`, `\dontrun` and `\donttest` kept). It finds calls of nls, nlme, gnls,
  nlsLM and nls2 (also as `pkg::fn`) and `selfStart` definitions. The formula is the argument as
  written, or resolved when it is a name assigned a formula earlier in the same file (or
  `formula()`/`as.formula()` of a literal). Parameter names come from `start=` (named
  list/c/data.frame/expand.grid, resolved like formulas), nlme `fixed=` and gnls `params=`
  (left-hand sides), `lower=`/`upper=` names, and the parameter arguments of self-starting
  models. Calls of stats' self-starting models (SSlogis, SSasymp, SSmicmen, SSfol, ...) and of
  functions or selfStart models defined in the same package are expanded into closed form when
  the function body is a closed-form expression; selfStart models are shared across packages
  (a second pass expands e.g. `nlraa::SSagauss` called from pacu). Every other variable of the right-hand side is
  a regressor (a data column), except names assigned a number earlier in the file (fixed
  constants). A selfStart definition contributes its own model, with the function arguments that
  are not parameters as regressors. The response (left-hand side) is recorded but the form is the
  right-hand side as fitted (a fit of log(y) is therefore a form for log(y)).
* Reduction (rule `rnls_corpus`, `hsr.rnls.build`): each right-hand side is transcribed into
  sympy (`hsr.corpus_sources.r_infix`; formulas with indexing, `$`, comparisons, `ifelse` or
  functions other than exp, log, sqrt, ... are dropped), identical formulas are counted once per
  package, and each regressor in turn becomes x1 with every other symbol a constant
  (`hsr.corpus.project`, as for rate laws; formulas with several regressors give one key per
  regressor); keys are generalised. A model is one package (`cran:<package>`), contributing each
  distinct form once, so `freq` counts packages. `stats` gives the calls found, parsed and dropped
  by reason (`drop_reasons`); `results/corpora/rnls_formulas.tsv.files.tsv` lists, per package,
  the code units read and those R could not parse.
* Snapshot of 2026-11-25: 354 packages downloaded; 27 contain at least one fit or selfStart
  definition (494 calls: 279 nls, 82 nlme, 60 nlsLM, 29 gnls, 44 selfStart; no nls2). Dropped:
  111 not arithmetic (piecewise models written with comparisons or `ifelse`, e.g. most of nlraa's
  linear-plateau models), 55 formula not resolvable (a function argument or built at run time),
  20 without parameter names (start values computed at run time), 13 calling a function that is
  not closed form (e.g. plantecophys' `acifun_wrap`, `plnorm`), 4 without a regressor, 1 with no
  canonical key. The 291 parsed formulas give 174 distinct (package, formula) pairs; 21 packages
  contribute 110 model-form pairs and 70 distinct forms (all expressible), 14 of them in none of
  the twelve training sources. The largest contributors are nlraa (28 forms), AgroReg (19),
  bigleaf and plantecophys (10 each) and vegan (9).
* Limitations: a regressor is "any non-parameter variable", so a constant passed in from the
  enclosing function (e.g. a fixed g0) is treated as a regressor and adds a trivial projection;
  forms fitted inside generic wrappers (e.g. `nls(formula, ...)` where the formula is a function
  argument) are not resolvable; packages that fit through optim/mle/brms are not seen.
* Terms of use: each CRAN package carries its own licence (column `license` of
  `rnls_packages.tsv`, also recorded per model under `licenses` in the corpus file; all FOSS: of the
  21 contributing packages, 15 GPL/AGPL variants and 6 MIT). Only derived skeletons and package names are committed.
  Cite the packages (`citation("<package>")`) and the CRAN task views (maintainers: Agriculture,
  J Piaskowski, A Sparks, A Correndo; Environmetrics, G L Simpson; Epidemiology, T Jombart,
  M Rolland, H Gruson; Pharmacokinetics, B Denney, S Nayak; Tracking, R Joo, M Basille;
  https://CRAN.R-project.org/view=<name>).

## Rate-law libraries (`corpus_ratelaws.json`; hold-out)

* Source: (a) the Systems Biology Ontology OWL export (`SBO_OWL.owl` in the EBI-BioModels/SBO
  repository at the commit in `config/ratelaws.yaml`; the export is dated 2021-11-03, last commit
  2023-05-16): every term under SBO:0000001 "rate law" with a MathML definition (127 terms);
  (b) COPASI's built-in function database (`copasi/function/FunctionDB.xml.cpp`, which embeds the
  CopasiML function list as a string) at tag Build-311 (80 functions). Rule `ratelaws_fetch`
  (two GETs from raw.githubusercontent.com; manifest of URLs, date and sha256).
* Parsing: rule `ratelaws_corpus` (`hsr.ratelaws`). SBO: the `lambda` of each term's MathML is
  converted to sympy (plus, minus, times, divide, power, exp, ln, log, root, tanh); its bound
  variables carry SBO terms, and a bound variable is a variable if its term is, or descends from,
  SBO:0000360 "quantity of an entity pool" (concentrations and particle numbers of reactants,
  products, enzymes, inhibitors, activators, modifiers), SBO:0000010 "reactant", SBO:0000015
  "substrate" or SBO:0000586 "model time"; all other bound variables (rate, Michaelis, inhibition
  and equilibrium constants, Hill coefficients, stoichiometries, ...) are constants. COPASI: the
  infix expression with the parameters whose role is substrate, product or modifier (volume and
  time would also count; none occur) as variables; the mass-action `PRODUCT<substrate_i>` is read
  as one substrate. Each variable in turn is x1 (`hsr.corpus_sources.law_keys`), keys generalised.
  A model is one rate-law definition (`sbo:SBO_xxxxxxx`, `copasi:<name>`); several MathML
  definitions of one term (continuous and discrete) are pooled.
* Losses (snapshot 2026-11-25): 13 of 207 definitions give no key and are listed under `dropped`:
  9 SBO laws written with indexed sums or products over an arbitrary number of species
  (`<sum>`/`<product>` with a bound index: generic mass action, inhibition by n inhibitors), 2
  zeroth-order SBO laws and COPASI's two constant-flux functions (no variable). All 194 others
  reduce: 194 models, 344 model-form pairs, 31 distinct forms. Only 1 form is absent from the
  twelve training sources, and it is a non-canonical rendering of a hyperbolic form (known
  canonicalisation gap), i.e. these libraries are almost entirely covered by the kinetic laws of
  BioModels, SABIO-RK, JWS and CellML.
* Terms of use: SBO is distributed under the Artistic License 2.0 (repository licence); cite
  Courtot M et al. (2011) Controlled vocabularies and semantics in systems biology. Mol Syst Biol
  7:543. COPASI is open source under the Artistic License 2.0; cite Hoops S et al. (2006)
  COPASI - a COmplex PAthway SImulator. Bioinformatics 22:3067-3074. Only derived skeletons and
  term ids / function names are committed.

## Ecology and demography catalogue, hold-out (`ecocat2_curves.csv` -> `corpus_ecocat2.json`)

A hold-out source (config `holdout_sources`; `hsr.grammar_compare.HOLDOUT_SOURCES`): it is scored
by the `holdout` target and never used to train a prior. Same format as the R curve catalogues
above (columns `id`, `package`, `function`, `model`, `variable`, `params`, `defs`, `formula`,
`r_call`, `ranges`, `reference`, `notes`); rules `ecocat2_cran_fetch`, `ecocat2_eval`,
`ecocat2_corpus` (`workflow/rules/ecocat2.smk`, `workflow/scripts/ecocat2_eval.R`,
`hsr.corpus_sources.curves_build`). 102 rows:

* Mortality laws (37 rows): every closed-form hazard of **MortalityLaws** 3.0.0 (de Moivre,
  Gompertz in three parametrisations, inverse Gompertz, Makeham (2), Opperman, Thiele, Wittstein,
  Weibull, inverse Weibull, Perks, Steffensen, Pareto II, Scholey (2), van der Maen (2),
  Strehler-Mildvan, quadratic, Beard, Beard-Makeham, gamma-Gompertz, Siler, Heligman-Pollard 1-4,
  Kostaki (both branches), Martinelle, Rogers-Planck, Kannisto, Kannisto-Makeham, the two
  log-quadratic laws), hazard as a function of age. The Carriere mixtures (hazard by numerical
  differencing) are left out.
* Species abundance and accumulation (19 rows): the rank-abundance models of **vegan** 2.7-5
  `radfit` with a closed form (preemption, Zipf, Zipf-Mandelbrot; the log-normal model uses the
  normal quantile function and is left out); the species-accumulation models of `fitspecaccum`
  (vegan's own self-starting Arrhenius, Gleason, Gitay and Lomolino models, and the stats models it
  uses: asymptotic regression, Gompertz, Michaelis-Menten, logistic, Weibull); the closed-form
  species-abundance distributions of **sads** 0.6.5 (log-series, metacommunity zero-sum
  multinomial, power, Zipf, Zipf-Mandelbrot, broken stick, geometric series).
* Photosynthesis-irradiance curves (19 rows): the nine net-photosynthesis light-response models
  of **photosynthesisLRC** 1.0.6 (Lobo et al. 2013: Baly, Kaipiainen, Smith, Abe, Jassby-Platt,
  Prioul-Chartier non-rectangular hyperbola, Webb, Prado-Moraes, Ye); the phytoplankton P-E
  models inside **phytotools** 1.0 (archived on CRAN) `fitJP`, `fitEP`, `fitPGH`, `fitWebb`
  (Jassby-Platt, Eilers-Peeters, Platt-Gallegos-Harrison, Webb); the ecosystem GPP light
  responses of **REddyProc** 1.3.4 (rectangular, non-rectangular, hyperbolic tangent);
  **plantecophys** 1.4-6 `Jfun`; Bannister (1979) and Steele (1962), unverified.
* Soil respiration (6 rows): Lloyd-Taylor (REddyProc `fLloydTaylor`, verified); Q10,
  exponential, Arrhenius, Kirschbaum (1995) and Gaussian (Tuomi et al. 2008), unverified.
* Stomatal conductance (4 rows): the nls formulas of **plantecophys** `fitBB` (Medlyn et al.
  2011, its full form with gk (Duursma et al. 2013) and Leuning 1995 as functions of VPD; Ball,
  Woodrow & Berry 1987 as a function of net assimilation A; the other inputs constant).
* Pathogen transmission (7 rows, unverified; no R implementation): the transmission functions
  compiled by McCallum, Barlow & Hone (2001) Trends Ecol Evol 16:295 (mass action, frequency
  dependent, power, negative binomial, asymptotic) plus saturating incidence (Capasso & Serio
  1978) and Reed-Frost, with the density of infected hosts I as x and S constant.
* Microbial dose-response (4 rows, unverified): exponential, approximate beta-Poisson (two
  parametrisations), Weibull-gamma.
* Phenology and degree days (6 rows): the growing-degree-hours model of **chillR** 0.77 (`GDH`,
  Anderson et al. 1986, both branches; the rising branch is a cosine, which our grammar cannot
  express, so it is verified but left out of the corpus) and the dynamic chill model's
  temperature functions `xsfct`, `k1fct` (Fishman et al. 1987); Yan & Hunt (1999) and the
  single-triangle degree-day formula (Lindsey & Newman 1956), unverified. Wang-Engel and the
  other development-rate curves are already in `thermal`.
* Verification (rule `ecocat2_eval`, R 4.5.3 from `envs/r.yaml`): every row with an `r_call` is
  evaluated by the package code at three random parameter draws on a grid of x and compared with
  the sympy transcription (relative error < 1e-6 at >= 5 points). vegan and sads are installed in
  the pinned conda environment. MortalityLaws, phytotools, photosynthesisLRC, REddyProc,
  plantecophys and chillR are not on conda-forge: their CRAN source tarballs (versions in
  `config/ecocat2.yaml`; rule `ecocat2_cran_fetch`, cached under
  `results/corpora/raw/ecocat2/<pkg>/` with URL, date and sha256) are read and only the top-level
  function definitions of the listed R files are evaluated (models defined inside a fitting
  function, e.g. `JP` in `fitJP`, or as nls formulas, as in `fitBB`, are extracted from the
  function body). vegan's rank-abundance models are checked by fitting them to three BCI sites
  and comparing vegan's fitted values with the formula at vegan's coefficients; the normalising
  constants of the sads Zipf, Zipf-Mandelbrot and mZSM distributions are passed from R as
  parameters. All 82 rows with an R implementation agree (max relative error 1e-13); 20 rows have
  none and are kept with verified = false.
* A "model" is the original publication (column `model`), as in the other catalogues: the same
  law in two packages (e.g. Jassby-Platt in photosynthesisLRC, phytotools and REddyProc) counts
  once. 101 rows enter the corpus: 75 models, 89 model-form pairs, 69 distinct generalised forms,
  32 of them (35 model-form pairs) not in any of the twelve training sources (snapshot of
  2026-11-25). Forms that also occur in the training sources (e.g. Michaelis-Menten, logistic,
  Gompertz, Arrhenius) are kept; overlap is reported separately.
* Terms of use and citation: MortalityLaws (MIT; Pascariu MD (2018) MortalityLaws: Parametric
  Mortality Models, Life Tables and HMD, https://CRAN.R-project.org/package=MortalityLaws), vegan
  (GPL-2; Oksanen J et al., vegan: Community Ecology Package), sads (GPL-2; Prado PI, Miranda MD,
  Chalom A, sads: Maximum Likelihood Models for Species Abundance Distributions), photosynthesisLRC
  (MIT; Davis R, Goolsby E, photosynthesisLRC: Nonlinear Least Squares Models for Photosynthetic Light Response), phytotools (GPL-2; Silsbe GM, Malkin SY,
  phytotools), REddyProc (GPL (>= 2); Wutzler T et al. (2018) Biogeosciences 15:5015),
  plantecophys (GPL; Duursma RA (2015) PLoS ONE 10:e0143346), chillR (GPL-3; Luedeling E, chillR).
  Cite the original references in the `reference` column. Only the transcribed formulas, derived
  skeletons and identifiers are committed; package code is not.

# Overlap

## Overlap between sources (`results/corpora/overlap.md`)

Rule `corpus_overlap` tabulates, for every source of the leave-one-source-out comparison, its
models, model-form pairs and distinct generalised forms, the forms found in no other source, the
forms not in the union of the original five sources (BioModels curated, SABIO-RK, CellML,
curvelib, biocat), and the number of forms shared by each pair of sources.

Snapshot of 2026-11-24 (forms are generalised keys; ModelDB forms are long because called
FUNCTIONs are inlined, median 8 constants against 4 for CellML; many are singletons):

| source | models | model-form pairs | distinct forms | not in any other source | not in baseline union |
|---|---|---|---|---|---|
| biomodels | 947 | 3526 | 341 | 104 | - |
| sabiork | 5057 | 7976 | 175 | 104 | - |
| cellml | 347 | 4571 | 493 | 275 | - |
| curvelib | 38 | 38 | 30 | 8 | - |
| biocat | 54 | 54 | 30 | 3 | - |
| biomodels_nc | 531 | 2480 | 223 | 76 | 87 |
| thermal | 62 | 82 | 69 | 49 | 53 |
| ecocurves | 83 | 99 | 67 | 30 | 33 |
| modeldb | 778 | 19848 | 1256 | 1051 | 1063 |
| nlmixr2lib | 56 | 225 | 41 | 19 | 21 |
| allodb | 54 | 80 | 23 | 8 | 9 |
| jws | 673 | 3004 | 255 | 58 | 66 |

Baseline: 874 distinct forms; all twelve sources: 2185 (1311 not in the baseline).
