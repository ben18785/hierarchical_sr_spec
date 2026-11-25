# Hold-out source rnls (see hsr/data/sources/README.md). Included from workflow/Snakefile;
# conda paths here are relative to workflow/rules/.
#
# Nonlinear regression formulas (nls, nlme, gnls, nlsLM, nls2 calls and
# selfStart definitions) mined from CRAN packages of the bio/eco task views:
#
#     snakemake --use-conda --cores 2 hsr/data/corpus_rnls.json \
#         --allowed-rules rnls_index rnls_views rnls_packages rnls_fetch rnls_extract rnls_corpus
#
# The package list hsr/data/sources/rnls_packages.tsv (versions pinned) is
# committed; delete it and results/corpora/raw/rnls/index to take a new snapshot.
configfile: "config/rnls.yaml"

RNLS = config["rnls"]
RNLS_RAW = "results/corpora/raw/rnls"


rule rnls_index:
    """CRAN's task-view index (Views.rds) and package database (packages.rds), with sha256."""
    output:
        f"{RNLS_RAW}/index/manifest.json",
    params:
        raw=f"{RNLS_RAW}/index",
    conda:
        "../envs/hsr.yaml"
    shell:
        "PYTHONPATH={ROOT} python -m hsr.rnls index_fetch {params.raw}"


rule rnls_views:
    """Packages of the task views in config rnls.views, with title, description, version and licence."""
    input:
        man=f"{RNLS_RAW}/index/manifest.json",
        code="workflow/scripts/rnls_views.R",
    output:
        "results/corpora/rnls_views.tsv",
    params:
        views=",".join(RNLS["views"]),
        raw=f"{RNLS_RAW}/index",
    conda:
        "../envs/r.yaml"
    shell:
        "Rscript workflow/scripts/rnls_views.R {params.raw}/Views.rds {params.raw}/packages.rds {output} {params.views}"


rule rnls_packages:
    """Bio/eco keyword filter (hsr.rnls.BIO_KEYWORDS); the list, with pinned versions, is committed."""
    input:
        man=f"{RNLS_RAW}/index/manifest.json",
        views="results/corpora/rnls_views.tsv",
    output:
        "hsr/data/sources/rnls_packages.tsv",
    params:
        raw=f"{RNLS_RAW}/index",
        exclude=RNLS.get("exclude", "") or "",
    conda:
        "../envs/hsr.yaml"
    shell:
        "PYTHONPATH={ROOT} python -m hsr.rnls select_packages {params.raw} {input.views} {output} '{params.exclude}'"


rule rnls_fetch:
    """Source tarball of every listed package version (<= 40 requests per minute), with sha256."""
    input:
        "hsr/data/sources/rnls_packages.tsv",
    output:
        f"{RNLS_RAW}/manifest.json",
    params:
        raw=RNLS_RAW,
    conda:
        "../envs/hsr.yaml"
    shell:
        "PYTHONPATH={ROOT} python -m hsr.rnls fetch {params.raw} {input}"


rule rnls_extract:
    """Formulas of nls/nlme/gnls/nlsLM/nls2 calls and selfStart definitions (R's parser, nothing evaluated)."""
    input:
        man=f"{RNLS_RAW}/manifest.json",
        pkgs="hsr/data/sources/rnls_packages.tsv",
        code="workflow/scripts/rnls_extract.R",
    output:
        "results/corpora/rnls_formulas.tsv",
    params:
        raw=RNLS_RAW,
    conda:
        "../envs/r.yaml"
    shell:
        "Rscript workflow/scripts/rnls_extract.R {params.raw} {input.pkgs} {output}"


rule rnls_corpus:
    """Formulas reduced to generalised one-variable keys, one model per package (committed)."""
    input:
        man=f"{RNLS_RAW}/manifest.json",
        pkgs="hsr/data/sources/rnls_packages.tsv",
        tsv="results/corpora/rnls_formulas.tsv",
        code=["hsr/rnls.py"],
    output:
        "hsr/data/corpus_rnls.json",
    params:
        raw=RNLS_RAW,
    threads: 2
    conda:
        "../envs/hsr.yaml"
    shell:
        "PYTHONPATH={ROOT} python -m hsr.rnls build {params.raw} {input.pkgs} {input.tsv} {output} {threads}"
