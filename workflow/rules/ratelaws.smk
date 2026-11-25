# Hold-out source ratelaws (see hsr/data/sources/README.md). Included from workflow/Snakefile;
# conda paths here are relative to workflow/rules/.
configfile: "config/ratelaws.yaml"

RATELAWS = config["ratelaws"]


rule ratelaws_fetch:
    """The SBO OWL export and COPASI's function database at pinned commits
    (two GETs from raw.githubusercontent.com), with a manifest of URLs and
    sha256. Delete results/corpora/raw/ratelaws to refresh."""
    output:
        "results/corpora/raw/ratelaws/manifest.json",
    params:
        raw="results/corpora/raw/ratelaws",
        sbo=f"https://raw.githubusercontent.com/EBI-BioModels/SBO/{RATELAWS['sbo_commit']}/SBO_OWL.owl",
        copasi=(f"https://raw.githubusercontent.com/copasi/COPASI/{RATELAWS['copasi_commit']}"
                "/copasi/function/FunctionDB.xml.cpp"),
    conda:
        "../envs/hsr.yaml"
    shell:
        "PYTHONPATH={ROOT} python workflow/scripts/ratelaws_steps.py fetch {params.raw} {params.sbo} {params.copasi}"


rule ratelaws_corpus:
    """SBO rate laws (MathML) and COPASI functions reduced to generalised
    one-variable skeletons, one model per rate-law definition (committed; a
    hold-out source, never used for training)."""
    input:
        man="results/corpora/raw/ratelaws/manifest.json",
        code=["hsr/ratelaws.py", "hsr/corpus_sources.py", "hsr/corpus.py", "hsr/canonical.py"],
    output:
        "hsr/data/corpus_ratelaws.json",
    params:
        raw="results/corpora/raw/ratelaws",
    threads: 2
    conda:
        "../envs/hsr.yaml"
    shell:
        "PYTHONPATH={ROOT} python workflow/scripts/ratelaws_steps.py build {params.raw} {output} {threads}"
