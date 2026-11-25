"""Snakemake entry points for the rate-law libraries (hsr.ratelaws).

    python ratelaws_steps.py fetch <raw_dir> <sbo_url> <copasi_url>
    python ratelaws_steps.py build <raw_dir> <out.json> <processes>
"""
import sys

from hsr import ratelaws as rl

if __name__ == "__main__":
    {"fetch": rl.fetch, "build": rl.build}[sys.argv[1]](*sys.argv[2:])
