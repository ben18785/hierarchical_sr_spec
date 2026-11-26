# Handoff: where paper-1 work stands (2026-11-26)

This file is for an agent picking up the work. It records state that is not obvious from the code.
Read `docs/paper1_outline.tex` (the paper plan, with current numbers in blue and to-dos in red) and
`docs/prior_models.tex` (a careful description of every prior) first.

## The project in one paragraph

`hsr` is a hierarchical symbolic regression package. The work is split into two papers. **Paper 1**
(current focus) is a learned prior over model forms for biology and ecology. It uses a corpus of model-form
pairs from 12 training sources, a family of proper priors over expression trees, and an evaluation on forms
the priors were not trained on. **Paper 2** is hierarchical SR that uses this prior as its base measure;
it is on hold until the paper-1 prior is chosen.

## Ground rules (from the user; still in force)

- **Branches.**
  - Work on feature branches only. Never commit to `prototype` (the default branch). Pushing feature
    branches is fine.
  - Main integration branch: `themes`. Merge feature branches into it once their tests pass.
  - The `canon` branch is the exception: **the user reviews it before any merge**.
- **Compute.**
  - Fits and benchmarks go on GitHub Actions, not the user's Mac.
  - Cheap local Snakemake report steps are OK at 1–2 cores. Four cores caused an out-of-memory failure, and
    the Mac ran low on memory again on 2026-11-26.
  - Don't restart a killed background job unless the user asks.
- **BioModels (EBI)** is blocked from GitHub runners. Raw BioModels data are fetched locally only, and the
  workflow has `--config biomodels_offline=true` for rebuilding from the cache.
- **Reproducibility.** Everything goes through the Snakemake workflow (`workflow/Snakefile`,
  `config/config.yaml`) with pinned envs. Every table in the paper should come from a Snakemake target.
- **Priors** must be well-defined and proper, not heuristics.
- **Scope:** stay within biological and ecological modelling.
- **External text:** no bare GitHub @mentions in GitHub text, and no personal details in external requests.

## How to run things locally

- **Python:** the main checkout's venv, `/Users/bcl206/Documents/GitHub/hierarchical_sr_spec/.venv/bin/`.
  It has `python`, `pytest` and `snakemake`. Run from a worktree with `PYTHONPATH=$PWD`.
- **Tests:** `pytest -q tests/`. On Actions, `.github/workflows/tests.yml` installs `.[test,fnprior,corpus]`.
  The full suite takes about 48 min there; it last passed on `themes` (run 45717082613).
- **Snakemake in a fresh worktree** wants to re-fetch every raw source.
  - Either link the raw caches into `results/corpora` from a worktree that has them, or use
    `--rerun-triggers mtime --allowed-rules ...` to run only the report steps.
  - The holdouts worktree (`.claude/worktrees/agent-a44c7f87e610b2445`) holds the rnls, ratelaws and ecocat2
    raw caches. Don't delete it until those caches are copied somewhere durable.

## Branch state

| branch | head | status |
|---|---|---|
| `themes` (= `paper1`) | 224dd8f | integrated: hold-outs, temporal, decomposition and mechanisms, blend, look-alike, CI fix, paper-1 outline |
| `canon` | 0370eaf (on bea2b95, **not** rebased onto `themes`) | canonicaliser fix; code done, corpora **not rebuilt**; awaiting user review |
| `prototype` | — | default branch; do not commit |

The worktrees under `.claude/worktrees/` (`holdouts`, `temporal`, `blend`, `lookalike`, `canon`, …) are
earlier agents' checkouts. Their branches are already merged into `themes`, except `canon`.

## Results so far (all in `results/grammar/`)

Scores are raw log P per test model-form pair, in nats. They are not renormalised.

- **Hold-out sources** (`holdout.md`): curated catalogues never trained on.
  - The sources are rnls (110 pairs, 14 novel), ratelaws (344, 1) and ecocat2 (89, 32).
  - Adaptors score −6.3, `hpy3_depth` −7.0, `fnprior` −7.9 and `sym0` −15.2. The PCFGs don't beat `sym0`.
  - **Gap between seen and novel forms**, about 17 nats for `fnprior`: roughly 10 from size, 2 from rarer
    symbols and 4–5 from structure. The structural part is about 13 nats for the adaptors.
  - Gain over `sym0` is +7.2 on seen forms and +7.6 on novel ones.
- **Temporal split** (`temporal.md`): cutoffs 2010, 2015 and 2020, each with equal and pooled weighting.
  - `tree_ad3` is best on all and on novel pairs in all 6 settings.
  - ModelDB makes up 65–80% of the test pairs. The ranking holds without ModelDB.
  - "525 of 944 novel 2010 forms are redundant" is **probably inflated** by false positives in the
    redundancy test (see the canon notes below).
- **Blend** (`blend.md`): `w·P_adaptor + (1−w)·P_context`, with w chosen by leave-one-source-out.
  - w comes out at about 0.72–0.84 and is stable.
  - It gains 0.4–1.2 nats on novel forms and costs 0.05–0.1 nats on seen forms.
- **Mechanism-derived forms** (`mechanism_forms.md`, `mechanisms.md`): random enzyme (King–Altman),
  compartment (time courses) and temperature (Arrhenius/Eyring) mechanisms.
  - Training already covers 93%, 86% and 29% of mechanisms respectively. The novel forms are the complex
    ones.
  - Log P per mechanism for enzyme / compartment / temperature: `hpy3_depth` −6.2 / −5.6 / −19.9,
    `tree_ad3` −6.0 / −5.5 / −27.6, `sym0` −17.9 / −14.3 / −35.3.
- **Look-alike test** (`lookalike.md`): evidence fits ran on Actions (run 45719012957,
  `.github/workflows/lookalike.yml`).
  - The design has 49 true forms and 588 datasets, with about 5,100 candidates per dataset.
  - The truth is within 5 nats of the best evidence on 43% of datasets.
  - **Seen truths:** the learned priors raise P(true) from 0.03 (flat prior) to 0.13 (`hpy3_depth`), and the
    truth's mean rank goes from 53 to 6.4.
  - **Novel truths:** the priors make things worse, with P(true) at 0.03 against 0.06 flat. Hold-out novel
    forms get almost zero.
  - The adaptors are no better than `hpy2`/`fnprior` here.
  - **Follow-ups on branch `lookalike2`** (`lookalike.md`, `lookalike_tempering.md`; both cover both designs
    and both weightings):
    - **Design B** (`lookalike_c7`: up to 7 constants and 25 nodes; 5,792 candidates, 73 true forms including all
      41 eligible mechanism forms, 876 datasets; Actions run 45902288688, 7.6 CPU h). Seen truths: P(true)
      0.04 flat → 0.17 `hpy3_depth`. Novel truths: 0.017 flat → 0.007. Same picture as design A, sharper.
    - **Pooled weighting** changes little: within ±0.01 in P(true) for the tree priors; the PCFGs gain slightly.
    - **Tempering** P^τ/Z: leave-one-truth-form-out picks τ ≈ 0.1–0.2 for the tree priors (0–0.1 for the
      PCFGs). Seen truths alone prefer τ ≈ 0.4–0.5; novel truths prefer τ ≈ 0–0.1. Mean log P(true) is
      dominated by datasets where the truth fits badly, so the chosen τ is small.
  - **Caveats:**
    - With so many candidates, a wrong form usually fits best by chance.
    - The evidence is a Laplace approximation only; the true-form fit fails on 6% (A) / 11% (B) of datasets.
    - The truth's posterior rank (median) is in the tens to hundreds, so the absolute numbers are small.
  - The Actions workflow is started by a push that changes `.github/lookalike-trigger`; a last line with
    `[design=c7]` runs design B instead of the base design.
    `gh workflow run` doesn't work because the workflow isn't on `prototype`, and merging it there needs
    the user's OK.

## Open work, in rough priority order

1. **Finish `canon` (needs the user's go-ahead to run).** The code is done and its tests pass. Still to do:
   - **Rebuild the corpora:** about 3–4 h at 1 core, locally (BioModels), with no network. The script is
     `scratchpad/canon/smk.sh`. Scratchpad files may be gone; it ran
     `snakemake --cores 1 --rerun-triggers mtime --config biomodels_offline=true` over the 16 corpus and
     generalisation rules.
   - **Write the change report:** `snakemake --cores 1 canon_changes` produces
     `results/grammar/canon_changes.md/json`. Commit it with the rebuilt corpora on `canon`, then **report
     to the user and do not merge**.
   - **What changed:**
     - The time limit is now CPU time (`ITIMER_PROF`) instead of wall-clock time; wall-clock limits were the
       cause of the nondeterminism.
     - `reduce_key` re-canonicalises each key until it stops changing; 305 of 2,368 old keys were not stable.
     - Redundant constants are removed, but only in rank-deficient forms.
     - Each identity rewrite is checked numerically.
     - A row-scaled `is_redundant` fixes the false positives in `hsr/gap_diagnostics.py::redundant`.
   - **Known gaps left:**
     - Sign-flipped Hill forms are not merged, e.g. `c0/(c1 + x1**c2)` and `c0*x1**c1/(c2 + x1**c1)`.
     - Neither are `c0/(c1 + x1) + c2` and `(c0 + c1*x1)/(c2 + x1)`.
     - Fixing a scaling constant at 1 assumes it is positive.
   - **If adopted:** rebase onto `themes`, use the new redundancy test in `gap_diagnostics.py`, and rerun
     everything downstream on Actions (generalised map → folds/scores → LOSO, hold-out, temporal, blend,
     mechanisms, look-alike → reports).
2. **Look-alike follow-ups:** pooled weighting, the larger cap and tempering are done (branch `lookalike2`).
   Left: a more accurate evidence than Laplace; deciding whether a tempered prior (τ from LOTO) is used in
   paper 1.
3. **Prior sensitivity:** tempering is in `lookalike_tempering.md`; other sensitivity checks are still open.
4. **Paper-1 outline:** fill the red `\todo{}` items in `docs/paper1_outline.tex` from the reports above.
   Generate the tables with Snakemake targets rather than typing them. The user will need to create a new
   Overleaf project for paper 1; the old Overleaf clone at `~/Documents/GitHub/hierarchical_sr_overleaf`
   is the general write-up.
5. **After the paper-1 prior is chosen:** retrain the fixed priors on the 12 sources and rerun the paper-2
   grids on Actions.

## Main Snakemake targets

`holdout`, `temporal`, `blend`, `mechanisms`, `lookalike` (report only; the fits come from Actions
artifacts) and `canon_changes` (on `canon` only). Config sections with matching names are in
`config/config.yaml`.
