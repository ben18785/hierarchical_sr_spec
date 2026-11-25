# Hold-out source ecocat2 (see hsr/data/sources/README.md). Included from workflow/Snakefile;
# conda paths here are relative to workflow/rules/.
#
#     snakemake --use-conda --cores 2 hsr/data/corpus_ecocat2.json
configfile: "config/ecocat2.yaml"

ECOCAT2 = config["ecocat2"]


def _ecocat2_tarball(pkg):
    return f"results/corpora/raw/ecocat2/{pkg}/{pkg}_{ECOCAT2['cran'][pkg]}.tar.gz"


rule ecocat2_cran_fetch:
    """A CRAN source tarball for the ecology catalogue (current or archived; version in config/ecocat2.yaml), with its sha256."""
    output:
        "results/corpora/raw/ecocat2/{pkg}/manifest.json",
    wildcard_constraints:
        pkg="|".join(ECOCAT2["cran"]),
    params:
        raw="results/corpora/raw/ecocat2/{pkg}",
        version=lambda wc: ECOCAT2["cran"][wc.pkg],
    conda:
        "../envs/hsr.yaml"
    shell:
        "PYTHONPATH={ROOT} python workflow/scripts/corpus_sources.py cran_fetch {params.raw} {wildcards.pkg} {params.version}"


rule ecocat2_eval:
    """Evaluate the R implementations (vegan, sads installed; the function
    definitions of MortalityLaws, phytotools, photosynthesisLRC, REddyProc,
    plantecophys and chillR read from their CRAN tarballs) of every catalogue
    row at random parameters; package versions are written alongside."""
    input:
        src="hsr/data/sources/ecocat2_curves.csv",
        code="workflow/scripts/ecocat2_eval.R",
        man=expand("results/corpora/raw/ecocat2/{pkg}/manifest.json", pkg=list(ECOCAT2["cran"])),
    output:
        csv="results/corpora/ecocat2_eval.csv",
        versions="results/corpora/ecocat2_eval.csv.versions",
    params:
        pkgs=",".join(ECOCAT2["installed"]),
        sourced=",".join(f"{_ecocat2_tarball(p)}::{m}" for p, ms in ECOCAT2["sourced"].items() for m in ms),
    conda:
        "../envs/r.yaml"
    shell:
        "Rscript workflow/scripts/ecocat2_eval.R {input.src} {output.csv} {params.pkgs} '{params.sourced}'"


rule ecocat2_corpus:
    """Verified, canonicalised ecology catalogue (hold-out), one model per original publication (committed)."""
    input:
        src="hsr/data/sources/ecocat2_curves.csv",
        ev="results/corpora/ecocat2_eval.csv",
        code=["hsr/corpus_sources.py", "hsr/corpus.py", "hsr/canonical.py", "hsr/grammar_compare.py"],
    output:
        "hsr/data/corpus_ecocat2.json",
    params:
        label="Ecology and demography catalogue (hold-out; hsr/data/sources/ecocat2_curves.csv): mortality laws, "
              "species-abundance and species-accumulation curves, photosynthesis-irradiance, soil respiration, "
              "stomatal conductance, transmission, microbial dose-response and phenology functions; verified "
              "numerically against R implementations where one exists; one model per original publication",
    conda:
        "../envs/hsr.yaml"
    shell:
        "PYTHONPATH={ROOT} python workflow/scripts/corpus_sources.py curves_build {input.src} {input.ev} {output} '{params.label}'"
