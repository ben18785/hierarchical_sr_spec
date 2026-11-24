"""Snakemake entry points for the further corpora (see hsr/corpus_sources.py).

    python corpus_sources.py sabiork_fetch <raw_dir>
    python corpus_sources.py sabiork_build <raw_dir> <out.json>
"""
import sys

from hsr import corpus_sources as cs

if __name__ == "__main__":
    getattr(cs, sys.argv[1])(*sys.argv[2:])
