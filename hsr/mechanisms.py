"""True forms derived from mechanisms rather than from the literature (test 4).

A random enzyme mechanism is a set of enzyme states joined by first-order
transitions (King & Altman 1956). Binding steps of the ligand x have
pseudo-first-order rate k*x; every other rate is a constant. With the total
enzyme conserved, the steady state of the linear system Q p = 0, sum p = 1 is a
ratio of polynomials in x, and so is the net flux through the product-release
step. The same algebra describes a forager's time budget (searching, handling,
digesting, interfering states, with x the prey density), so the forms cover
saturating feeding rates as well as enzyme kinetics.

The form of a mechanism is the family of rational functions with the same
monomial support,

    f(x) = sum_{i in N} a_i x^i / sum_{i in D} b_i x^i,

with every coefficient free except one denominator coefficient fixed to 1.
This is the identifiable family the mechanism belongs to (coefficient
constraints between a_i and b_i, which a particular mechanism may impose, are
not used). It is canonicalised with hsr.canonical like corpus forms.

Mechanism generator (``random_mechanism``):
* a catalytic cycle of m states (m = 2..max_states), the last step releasing
  product; each step reversible with probability p_rev;
* role "substrate": x binds on the first step; then a geometric number
  (continue with probability p_second, at most max_extra) of further
  x-bindings, each on a later cycle step (several substrate molecules) or as
  an x-bound dead-end complex off any existing state (substrate inhibition,
  chains of bound states);
* role "modifier": x binds only off-cycle: a dead-end complex (inhibition) or
  an x-bound state with its own product release (activation / partial
  inhibition), plus the same geometric number of further x-bound dead-ends;
* with probability p_deadend an extra x-free dead-end state (e.g. an
  isomerisation) is attached.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

X = sp.Symbol("x1", positive=True)


@dataclass
class Mechanism:
    n_states: int
    edges: list = field(default_factory=list)   # (i, k, binds_x, name)
    release: list = field(default_factory=list)  # names of product-release edges (net flux counted)
    description: str = ""

    def add(self, i, k, binds_x=False):
        name = f"k{len(self.edges)}"
        self.edges.append((i, k, binds_x, name))
        return name

    def symbols(self):
        return {name: sp.Symbol(name, positive=True) for *_, name in self.edges}

    def flux(self, values=None):
        """Steady-state net flux through the release edges (sympy, in x1 and the
        k's, or in x1 only when ``values`` maps each k name to a number)."""
        ks = self.symbols() if values is None else {n: sp.Integer(v) for n, v in values.items()}
        m = self.n_states
        Q = sp.zeros(m, m)
        rate = {}
        for i, k, bx, name in self.edges:
            r = ks[name] * (X if bx else 1)
            rate[name] = (i, r)
            Q[k, i] += r
            Q[i, i] -= r
        A = Q.copy()
        A[m - 1, :] = sp.ones(1, m)
        b = sp.zeros(m, 1)
        b[m - 1] = 1
        p = A.LUsolve(b)
        f = 0
        for name in self.release:
            i, r = rate[name]
            f += r * p[i]
            rev = self._reverse.get(name)
            if rev is not None:
                j, rr = rate[rev]
                f -= rr * p[j]
        return sp.cancel(sp.together(f))

    _reverse: dict = field(default_factory=dict)


def random_mechanism(rng, max_states=5, p_rev=0.4, p_second=0.6, p_deadend=0.2, role=None, max_extra=3):
    m = int(rng.integers(2, max_states + 1))
    mech = Mechanism(n_states=m)
    mech._reverse = {}
    role = role or ("substrate" if rng.random() < 0.6 else "modifier")
    desc = [f"cycle of {m} states", f"x is {role}"]
    cycle = []
    for i in range(m):
        k = (i + 1) % m
        bx = role == "substrate" and i == 0
        f = mech.add(i, k, bx)
        cycle.append(f)
        if rng.random() < p_rev:
            r = mech.add(k, i, False)
            if i == m - 1:
                mech._reverse[f] = r
            desc.append(f"step {i} reversible")
    mech.release = [cycle[-1]]
    n = m

    def dead_end(state, bx):
        nonlocal n
        mech.n_states += 1
        new = n
        n += 1
        mech.add(state, new, bx)
        mech.add(new, state, False)
        return new

    n_extra = 0
    while n_extra < max_extra and rng.random() < p_second:
        n_extra += 1
    for _ in range(n_extra if role == "substrate" else 0):
        if True:
            if m >= 3 and rng.random() < 0.4:
                i = int(rng.integers(1, m - 1))
                # make the cycle step i -> i+1 an x-binding step
                for t, (a, b, bx, name) in enumerate(mech.edges):
                    if a == i and b == i + 1:
                        mech.edges[t] = (a, b, True, name)
                desc.append(f"second substrate binding on step {i}")
            else:
                s = int(rng.integers(0, mech.n_states))
                dead_end(s, True)
                desc.append(f"x-bound dead-end off state {s}")
    if role != "substrate":
        s = int(rng.integers(0, m))
        if rng.random() < 0.6:
            dead_end(s, True)
            desc.append(f"inhibitor: x dead-end off state {s}")
        else:
            # x-bound state with its own product release back to state 0
            mech.n_states += 1
            new = n
            n += 1
            mech.add(s, new, True)
            mech.add(new, s, False)
            rel = mech.add(new, 0, False)
            mech.release.append(rel)
            desc.append(f"modifier: x-bound state off state {s} with its own release")
    if rng.random() < p_deadend:
        s = int(rng.integers(0, m))
        dead_end(s, False)
        desc.append(f"x-free dead-end off state {s}")
    for _ in range(n_extra if role != "substrate" else 0):
        s = int(rng.integers(0, mech.n_states))
        dead_end(s, True)
        desc.append(f"further x-bound dead-end off state {s}")
    mech.description = "; ".join(desc)
    return mech


def support(expr):
    """(numerator powers, denominator powers) of a rational function of x1."""
    num, den = sp.fraction(sp.cancel(expr))
    pn = sp.Poly(sp.expand(num), X)
    pd = sp.Poly(sp.expand(den), X)
    sn = sorted(int(mon[0]) for mon, c in zip(pn.monoms(), pn.coeffs()) if c != 0)
    sd = sorted(int(mon[0]) for mon, c in zip(pd.monoms(), pd.coeffs()) if c != 0)
    return sn, sd


def family_string(sn, sd):
    """Free-coefficient rational family as a skeleton string (C = free constant);
    the highest denominator power has coefficient 1."""
    def term(i, free=True):
        xp = "" if i == 0 else ("x1" if i == 1 else f"x1**{i}")
        if not free:
            return xp or "1"
        return "C" if i == 0 else f"C*{xp}"
    num = " + ".join(term(i) for i in sn)
    top = max(sd)
    den = " + ".join(term(i, free=(i != top)) for i in sd)
    return f"({num})/({den})"


def mechanism_form(mech, rng=None):
    """Canonical key (or None) and details for a mechanism. The monomial
    support is read from the flux at random integer rate constants (generic
    values; a zero coefficient there is a structural zero with probability ~1)."""
    from .canonical import canonicalise
    rng = rng or np.random.default_rng(0)
    vals = {name: int(rng.integers(2, 10_000)) for *_, name in mech.edges}
    f = mech.flux(vals)
    if not f.has(X):
        return None
    sn, sd = support(f)
    if not sn or not sd:
        return None
    s = family_string(sn, sd)
    sk = canonicalise(s)
    if sk is None:
        return None
    return {"key": sk.key, "n_consts": sk.n_consts, "family": s, "num_support": sn, "den_support": sd,
            "description": mech.description, "n_states": mech.n_states,
            "edges": [[i, k, bool(bx)] for i, k, bx, _ in mech.edges]}


def generate(n, seed=0, **kw):
    """n random mechanisms -> list of form records (mechanisms without an x
    dependence or a canonical key are skipped and counted)."""
    rng = np.random.default_rng(seed)
    out, skipped = [], 0
    for i in range(n):
        mech = random_mechanism(rng, **kw)
        rec = mechanism_form(mech, rng)
        if rec is None:
            skipped += 1
            continue
        rec["id"] = f"mech{i}"
        out.append(rec)
    return out, skipped


def main(argv=None):
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser(description="Forms derived from random enzyme mechanisms")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-states", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs, skipped = generate(a.n, a.seed, max_states=a.max_states)
    freq = Counter(r["key"] for r in recs)
    doc = {"source": "random King-Altman mechanisms (hsr.mechanisms)", "n_mechanisms": a.n, "seed": a.seed,
           "max_states": a.max_states, "skipped": skipped,
           "per_model": {r["id"]: [r["key"]] for r in recs},
           "freq": dict(freq.most_common()), "records": recs}
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"{len(recs)} mechanisms, {len(freq)} distinct forms, {skipped} skipped")


if __name__ == "__main__":
    main()
