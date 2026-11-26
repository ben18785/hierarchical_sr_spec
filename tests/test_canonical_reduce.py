"""Determinism of canonicalisation and removal of redundant constants.

* canonicalise must give the same keys whatever PYTHONHASHSEED and whatever
  the order in which expressions are canonicalised in one process (sympy's
  caches); checked on a fixed sample of corpus keys in subprocesses.
* reduce_redundant may only remove constants that are not identifiable: the
  new key must have fewer constants, full Jacobian rank, and the same family
  of functions (checked by fitting each form to curves of the other).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from hsr.candidates import skeleton_from_key
from hsr.canonical import canonicalise, consts_of, is_redundant, reduce_key

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "hsr", "data")

# ------------------------------------------------------------ determinism
# a fixed sample of corpus keys of every kind (rate laws with shared
# constants, exponentials, powers, logs, sums of fractions, literals)
SAMPLE = [
    "(c0 + c1*x1)/(c2 + x1)",
    "c0*x1/(c0*x1 + c1)",
    "c0*(c1*x1 + c2)**c3",
    "c0*x1**c1/(c2 + x1**c1)",
    "c0/(c1 + exp(c2*log(x1)))",
    "c0*exp(c1*x1)/(c0*exp(c1*x1) + c2) + c3",
    "c0*(c1 + c2*x1)*(c3 + x1)/(c4 + x1)",
    "c0*x1/sqrt(c1*x1**2 + c2) + c3",
    "c0*log(c1*x1 + c2) + c3",
    "c3*(c5*x1 + c6)/(c3*(c1*(c0*x1 + c7) + c2) + c4)",
    "(c0*x1 + c1)/(c2*c3 + c2*c4*x1 + c3*x1 + c4*x1**2)",
    "c0*exp(c3*c4/x1 + c4)/(c1 - exp(c2)*exp(c2*c3/x1))",
    "2*c0*x1/(c1 + x1) + 0.5*x1",
    "c0*(1 - exp(-c1*x1))**3",
    "(c0*x1**2 + c1*x1 + c2)/(c3*x1 + c4 + x1**2)",
    "c0**x1*c1/(c0**x1*c1 + c2)",
]

_SCRIPT = r"""
import json, sys
sys.path.insert(0, {root!r})
from hsr.canonical import canonicalise
keys = json.loads(sys.argv[1])
if sys.argv[2] == "reverse":
    keys = keys[::-1]
out = {{}}
for k in keys:
    sk = canonicalise(k, 1, max_consts=None)
    out[k] = None if sk is None else sk.key
print(json.dumps(out, sort_keys=True))
"""


def _run(keys, seed, order="forward"):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    res = subprocess.run([sys.executable, "-c", _SCRIPT.format(root=ROOT), json.dumps(keys), order],
                         env=env, capture_output=True, text=True, check=True, timeout=1200)
    return json.loads(res.stdout.strip().splitlines()[-1])


def _corpus_sample(n=12, seed=0):
    """A fixed random sample of keys of the committed corpora."""
    keys = set()
    for name in ("corpus_cellml.json", "corpus_jws.json", "corpus_modeldb.json", "corpus_sabiork.json"):
        with open(os.path.join(DATA, name)) as fh:
            keys |= {k for k in json.load(fh)["freq"] if len(k) < 120}
    keys = sorted(keys)
    rng = np.random.default_rng(seed)
    return [keys[i] for i in rng.choice(len(keys), n, replace=False)]


@pytest.mark.slow
def test_canonicalise_is_independent_of_hash_seed_and_call_order():
    keys = SAMPLE + _corpus_sample()
    ref = _run(keys, 0)
    assert _run(keys, 1) == ref
    assert _run(keys, 4242, order="reverse") == ref


def test_canonicalise_is_independent_of_hash_seed_small():
    keys = SAMPLE[:6]
    assert _run(keys, 0) == _run(keys, 7, order="reverse")


# ------------------------------------------------------------ reduction
REDUCED = [
    ("c0*x1 + c1*x1", "c0*x1"),
    ("c0*x1/(c0*x1 + c1)", "x1/(c0 + x1)"),
    ("c0*(c1*x1 + c2)**c3", "c0*(c1 + x1)**c2"),
    ("c0*(c1*x1 + c2)**c3/x1", "c0*(c1 + x1)**c2/x1"),
    ("c0/(c1*x1 + c2)**2", "c0/(c1 + x1)**2"),
    ("c0*log(c1*x1 + c2) + c3", "c0*log(c1 + x1) + c2"),
    ("c0*(c1 + x1)**2 + c2*x1 + c3", "c0*x1**2 + c1*x1 + c2"),
    ("c0*(c1 + c2*x1)*(c3 + x1)/(c4 + x1)", "(c0 + c1*x1)*(c2 + x1)/(c3 + x1)"),
    ("c0*x1/sqrt(c1*x1**2 + c2) + c3", "c0*x1/sqrt(c1 + x1**2) + c2"),
    ("c3*(c5*x1 + c6)/(c3*(c1*(c0*x1 + c7) + c2) + c4)", "(c0 + c1*x1)/(c2 + x1)"),
    ("c0*exp(c1*x1)/(c0*exp(c1*x1) + c2) + c3", "c0 + exp(c2*x1)/(c1 + exp(c2*x1))"),
    ("c0**x1/(c0**x1 + exp(c1*x1))", "1/(exp(c0*x1) + 1)"),
    ("(c0*(c1 + x1)*(c5 + x1) + (c2 + c3*x1)*(c4 + x1))/((c1 + x1)*(c4 + x1))",
     "c0/(c1 + x1) + c2/(c3 + x1) + c4"),
    # Heligman-Pollard: the scale of the logistic term c6**x1*c7/(c6**x1*c7 + c8)
    ("c5*exp(c4*(c2 + log(x1))**2) + c6**x1*c7/(c6**x1*c7 + c8) + exp(c1*(c0 + x1)**c3)",
     "c5*exp(c4*(c2 + log(x1))**2) + exp(c1*(c0 + x1)**c3) + exp(c6*x1)/(c7 + exp(c6*x1))"),
]

IDENTIFIABLE = [
    "c0*x1/(c1 + x1)",
    "c0*exp(c1*x1)",
    "(c0 + c1*x1)/(c2 + x1)",
    "c0*x1**c1/(c2 + x1**c1)",
    "c0*exp(c1/x1**5) + c2",       # rank test needs row scaling here
    "c0*x1/(c1 + x1) + c2*x1/(c3 + x1)",
    "c0*exp((c1 + c2*(c3 + x1))*(c3 + x1)**2)",
]


def _family_residual(src_key, dst_key, rng, n_tries=3):
    """Worst relative RMS misfit of dst fitted to noiseless curves of src."""
    from hsr.likelihood import CompiledSkeleton, _lm
    a, b = CompiledSkeleton(skeleton_from_key(src_key)), CompiledSkeleton(skeleton_from_key(dst_key))
    x = np.linspace(0.3, 3.0, 40)
    worst, n_ok = 0.0, 0
    for _ in range(20):
        if n_ok == n_tries:
            break
        phi = rng.uniform(0.5, 1.5, a.k)
        with np.errstate(all="ignore"):
            y = a.f(x[None, :, None], phi[None, :])[0]
        if not np.all(np.isfinite(y)) or np.ptp(y) < 1e-3 or np.abs(y).max() > 1e6:
            continue
        S = 40
        starts = np.abs(rng.normal(1.0, 1.0, (S, b.k))) + 0.05
        X = np.broadcast_to(x[None, :, None], (S, 40, 1))
        Y = np.broadcast_to(y[None, :], (S, 40))
        M = np.ones((S, 40))
        phib, s, obj = _lm(b, X, Y, M, starts, np.full(S, 40.0), 3.0, False, 600, 1e-15)
        with np.errstate(all="ignore"):
            rms = np.sqrt(np.mean((b.f(X, phib) - Y) ** 2, axis=1))
        worst = max(worst, np.nanmin(rms) / (np.std(y) + 1e-12))
        n_ok += 1
    assert n_ok > 0
    return worst


@pytest.mark.parametrize("key,expected", REDUCED, ids=[k for k, _ in REDUCED])
def test_redundant_constants_are_removed(key, expected):
    new = reduce_key(key)
    assert new == expected
    old_e, new_e = skeleton_from_key(key).expr, skeleton_from_key(new).expr
    assert len(consts_of(new_e)) < len(consts_of(old_e))
    assert is_redundant(old_e) and is_redundant(old_e, equilibrate=False)
    assert not is_redundant(new_e) and not is_redundant(new_e, equilibrate=False)
    # canonical: the new key is a fixed point
    assert canonicalise(new, 1, max_consts=None).key == new


@pytest.mark.slow
@pytest.mark.parametrize("key,expected", REDUCED, ids=[k for k, _ in REDUCED])
def test_reduced_form_has_the_same_family(key, expected):
    """Curves of the old form are fitted exactly by the new one (nothing is
    lost) and conversely (nothing is added), at random positive constants."""
    rng = np.random.default_rng(0)
    assert _family_residual(key, expected, rng) < 1e-4
    assert _family_residual(expected, key, rng) < 1e-4


@pytest.mark.parametrize("key", IDENTIFIABLE)
def test_identifiable_forms_are_unchanged(key):
    assert not is_redundant(skeleton_from_key(key).expr)
    assert reduce_key(key) == key
    assert canonicalise(key, 1, max_consts=None).key == key


def test_power_written_through_log_is_a_power():
    """Lomolino's Asym/(1 + slope**log(xmid/x)) is Asym/(1 + k*x**-p), a
    Hill-type curve (not a power law): its key has x1**c, not exp(c*log(x1))."""
    k = canonicalise("c0/(1 + c2**log(c1/x1))", 1, max_consts=None).key
    assert k == "c0/(c1 + x1**c2)"
    assert canonicalise("c0*exp(c1*log(x1))", 1, max_consts=None).key == "c0*x1**c1"
    assert canonicalise("c0/(c0 + exp(c1*log(c2 + x1)))", 1, max_consts=None).key == "c0/(c0 + (c1 + x1)**c2)"


def test_shared_constant_base_becomes_exponential():
    # c**x1 = exp(log(c)*x1) also when c occurs more than once
    assert canonicalise("c0**x1/(c0**x1 + c1)", 1, max_consts=None).key == \
        canonicalise("exp(c0*x1)/(exp(c0*x1) + c1)", 1, max_consts=None).key
