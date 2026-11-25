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

Three mechanism classes (``CLASSES``; ``generate_class``) give three corpora:

enzyme (x1 = ligand concentration)
    ``random_mechanism`` as above; form = free-coefficient rational family.

compartment (x1 = time)
    ``random_compartment_system``: a linear compartment system dA/dt = K A
    with m compartments (m uniform on 2..max_comp). Without a transit chain
    every ordered pair (i, j) carries a first-order transfer i -> j with
    probability p_edge and every compartment a first-order elimination with
    probability p_elim. With probability p_transit the system starts with a
    transit chain (as in pharmacokinetic transit-compartment models):
    compartments 0..t-1 (t uniform on 2..m) joined 0 -> 1 -> ... -> t-1 by
    transfers that share one rate constant, with no other flows into or out
    of compartments 0..t-2; the remaining compartments t-1..m-1 are wired at
    random as above. If compartment 0 has no outflow it is given an
    elimination (so that the dosed amount changes in time). Input into
    compartment 0, a bolus (A(0) = e_0) or, with
    probability p_infusion, a constant infusion from t = 0 (A(0) = 0). The
    observed compartment is uniform over the compartments reachable from the
    input. Every distinct rate constant gets a random integer value (1..1000;
    generic), and the observed amount is found exactly through its Laplace
    transform X(s) = [(sI - K)^{-1} b]_o (divided by s for an infusion),
    reduced and split into partial fractions over the rationals: a pole lambda
    of order r contributes x1^j exp(lambda x1) for every power j < r whose
    partial-fraction coefficient is nonzero (a pole at 0 gives the polynomial
    terms 1, x1, ...). Repeated nonzero poles arise only from the tied transit
    rates (x1 exp(C x1) terms); a pole at 0 from mass trapped in compartments
    without elimination (a constant term; under an infusion a linear term).
    The form is the free-coefficient family
        sum_lambda (sum_j a_{lambda j} x1^j) exp(b_lambda x1) + sum_j a_{0j} x1^j,
    every a and b free (one rate b per distinct pole, shared by its powers).
    Systems with complex poles (oscillatory responses; possible with transfer
    cycles of length >= 3) or an identically zero response are dropped and
    counted.

temperature (x1 = absolute temperature)
    ``random_thermal_mechanism``: a random enzyme mechanism from
    ``random_mechanism`` (max_states_T, max_extra_T), plus, with probability
    p_inact, a Sharpe-Schoolfield-style reversible inactivation (an x-free
    dead-end state, both rates temperature dependent, off a uniformly random
    state) and, with probability p_inact again, a second one (low- and
    high-temperature inactivation). Every rate constant has an Arrhenius
    temperature dependence k = A exp(-E/x1) or, with probability p_eyring for
    the whole mechanism, an Eyring one k = A x1 exp(-E/x1). The ligand is held
    at a fixed level (absorbed into the binding rate constants) or, with
    probability p_saturating, saturating (the limit of the flux as the ligand
    level goes to infinity). The steady-state flux is N(k)/D(k) with N and D
    polynomials in the rate constants (matrix-tree theorem), reduced by their
    gcd. Distinct monomials prod k_i^{a_i} have distinct activation energies
    sum a_i E_i for generic E, so the form in x1 is
        x1^p sum_{m in N} a_m exp(b_m/x1) / (1 + sum_{m in D, m != m0} a_m exp(b_m/x1)),
    every term divided by one denominator term m0 and every a and b free
    (p = deg N - deg D under Eyring, 0 under Arrhenius). Dropped and counted:
    mechanisms whose flux vanishes at saturation (substrate inhibition), forms
    with more than max_terms exponential terms in all, and (every class) forms
    that do not canonicalise or have more than max_consts constants.
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


def _canon(expr, max_consts):
    """Canonical skeleton (no constant cap inside canonicalise; the cap is
    applied here so that drops can be counted by reason)."""
    from .canonical import canonicalise
    try:
        sk = canonicalise(expr, max_consts=None)
    except Exception:
        return None, "canonical_none"
    if sk is None:
        return None, "canonical_none"
    if max_consts is not None and sk.n_consts > max_consts:
        return None, "too_many_consts"
    return sk, None


def mechanism_form(mech, rng=None, max_consts=4, why=None):
    """Canonical key (or None) and details for a mechanism. The monomial
    support is read from the flux at random integer rate constants (generic
    values; a zero coefficient there is a structural zero with probability ~1).
    ``why`` (a dict), if given, receives the reason for a None."""
    rng = rng or np.random.default_rng(0)
    why = {} if why is None else why
    vals = {name: int(rng.integers(2, 10_000)) for *_, name in mech.edges}
    f = mech.flux(vals)
    if not f.has(X):
        why["reason"] = "no_x"
        return None
    sn, sd = support(f)
    if not sn or not sd:
        why["reason"] = "no_x"
        return None
    s = family_string(sn, sd)
    sk, reason = _canon(s, max_consts)
    if sk is None:
        why["reason"] = reason
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


# ======================================================== compartment systems
S = sp.Symbol("s")


@dataclass
class CompartmentSystem:
    n_comp: int
    transfers: list = field(default_factory=list)     # (i, j, name): i -> j
    eliminations: list = field(default_factory=list)  # (i, name)
    infusion: bool = False
    observed: int = 0
    transit: int = 0                                  # length t of the tied transit chain (0: none)
    description: str = ""

    def names(self):
        return sorted({n for *_, n in self.transfers} | {n for _, n in self.eliminations})

    def matrix(self, values):
        K = sp.zeros(self.n_comp, self.n_comp)
        for i, j, name in self.transfers:
            k = sp.Integer(values[name])
            K[j, i] += k
            K[i, i] -= k
        for i, name in self.eliminations:
            K[i, i] -= sp.Integer(values[name])
        return K

    def laplace(self, values):
        """Laplace transform of the observed amount (input into compartment 0)."""
        K = self.matrix(values)
        M = S * sp.eye(self.n_comp) - K
        b = sp.zeros(self.n_comp, 1)
        b[0] = 1
        y = M.LUsolve(b)[self.observed]
        if self.infusion:
            y = y / S
        return sp.cancel(sp.together(y))


def _reachable(transfers, start=0):
    seen, todo = {start}, [start]
    while todo:
        i = todo.pop()
        for a, b, _ in transfers:
            if a == i and b not in seen:
                seen.add(b)
                todo.append(b)
    return sorted(seen)


def random_compartment_system(rng, max_comp=4, p_edge=0.35, p_elim=0.5, p_infusion=0.3, p_transit=0.2):
    m = int(rng.integers(2, max_comp + 1))
    sysm = CompartmentSystem(n_comp=m)
    desc = [f"{m} compartments"]
    names = iter(f"k{i}" for i in range(1000))
    free = list(range(m))
    if rng.random() < p_transit:
        t = int(rng.integers(2, m + 1))
        sysm.transit = t
        ktr = next(names)
        for i in range(t - 1):
            sysm.transfers.append((i, i + 1, ktr))
        free = list(range(t - 1, m))
        desc.append(f"transit chain of {t} compartments with one shared rate")
    for i in free:
        for j in free:
            if i != j and rng.random() < p_edge:
                sysm.transfers.append((i, j, next(names)))
        if rng.random() < p_elim:
            sysm.eliminations.append((i, next(names)))
    if not any(i == 0 for i, _, _ in sysm.transfers) and not any(i == 0 for i, _ in sysm.eliminations):
        # the dosed compartment must change in time: give it an elimination
        sysm.eliminations.append((0, next(names)))
    desc.append(f"{len(sysm.transfers)} transfers, eliminations from {sorted(i for i, _ in sysm.eliminations)}")
    sysm.infusion = bool(rng.random() < p_infusion)
    desc.append("constant infusion into 0" if sysm.infusion else "bolus into 0")
    reach = _reachable(sysm.transfers, 0)
    sysm.observed = int(reach[int(rng.integers(0, len(reach)))])
    desc.append(f"observed compartment {sysm.observed}")
    sysm.description = "; ".join(desc)
    return sysm


def exponential_structure(R):
    """Exponential-polynomial structure of the inverse Laplace transform of a
    proper rational function R(s) with rational coefficients: a list of
    (irreducible monic factor q(s), sorted powers j present) and whether every
    root is real. A term A(s)/q(s)^r of the partial fractions over the
    rationals contributes t^(r-1) exp(lambda t) for every root lambda of q (a
    nonzero A of degree < deg q cannot vanish at a root of the irreducible q)."""
    terms = sp.Add.make_args(sp.apart(R, S, full=False))
    struct = {}
    for t in terms:
        num, den = sp.fraction(sp.together(t))
        if not den.has(S):
            raise ValueError("improper rational function")
        _, facs = sp.factor_list(den, S)
        facs = [(q, r) for q, r in facs if q.has(S)]
        if len(facs) != 1:
            raise ValueError(f"unexpected partial fraction {t}")
        q, r = facs[0]
        q = sp.Poly(q, S).monic().as_expr()
        struct.setdefault(q, set()).add(int(r) - 1)
    real = all(sp.Poly(q, S).count_roots() == sp.degree(q, S) for q in struct)
    return [(q, sorted(js)) for q, js in sorted(struct.items(), key=lambda kv: str(kv[0]))], real


def compartment_form(sysm, rng=None, max_consts=8, why=None):
    """Canonical key (or None) and details for a compartment system: the
    free-coefficient exponential-polynomial family of its observed amount."""
    rng = rng or np.random.default_rng(0)
    why = {} if why is None else why
    vals = {n: int(rng.integers(1, 1001)) for n in sysm.names()}
    R = sysm.laplace(vals)
    if R == 0:
        why["reason"] = "zero_response"
        return None
    struct, real = exponential_structure(R)
    if not real:
        why["reason"] = "oscillatory"
        return None
    if all(q == S and js == [0] for q, js in struct):
        why["reason"] = "constant"
        return None
    c = iter(sp.Symbol(f"c{i}", real=True) for i in range(1000))
    expr, terms = 0, []
    for q, js in struct:
        for _ in range(int(sp.degree(q, S))):
            poly = sum(next(c) * X ** j for j in js)
            if q == S:
                expr += poly
                terms.append({"rate": "zero", "powers": js})
            else:
                expr += poly * sp.exp(next(c) * X)
                terms.append({"rate": "free", "powers": js})
    sk, reason = _canon(expr, max_consts)
    if sk is None:
        why["reason"] = reason
        return None
    return {"key": sk.key, "n_consts": sk.n_consts, "family": str(expr), "terms": terms,
            "poles": [[str(q), js] for q, js in struct], "description": sysm.description,
            "n_comp": sysm.n_comp, "transfers": [[i, j, n] for i, j, n in sysm.transfers],
            "eliminations": [[i, n] for i, n in sysm.eliminations], "infusion": sysm.infusion,
            "observed": sysm.observed, "transit": sysm.transit}


# ============================================================ temperature
def flux_polys(mech, ligand):
    """(N, D): steady-state net flux through the release edges as a ratio of
    polynomials in the rate-constant symbols (and ``ligand`` on binding
    steps), by the matrix-tree theorem: p_i is proportional to the principal
    minor of the Laplacian -Q with row and column i deleted."""
    ks = mech.symbols()
    m = mech.n_states
    L = sp.zeros(m, m)
    rate = {}
    for i, k, bx, name in mech.edges:
        r = ks[name] * (ligand if bx else 1)
        rate[name] = (i, r)
        L[k, i] -= r
        L[i, i] += r
    minors = []
    for i in range(m):
        idx = [j for j in range(m) if j != i]
        minors.append(sp.expand(L.extract(idx, idx).det(method="berkowitz")) if m > 1 else sp.Integer(1))
    N = 0
    for name in mech.release:
        i, r = rate[name]
        N += r * minors[i]
        rev = mech._reverse.get(name)
        if rev is not None:
            j, rr = rate[rev]
            N -= rr * minors[j]
    return sp.expand(N), sp.expand(sum(minors))


@dataclass
class ThermalMechanism:
    mech: Mechanism
    eyring: bool = False
    saturating: bool = False
    n_inact: int = 0
    description: str = ""


def random_thermal_mechanism(rng, max_states=4, max_extra=2, p_inact=0.4, p_eyring=0.3, p_saturating=0.5):
    mech = random_mechanism(rng, max_states=max_states, max_extra=max_extra)
    desc = [mech.description]
    n_inact = 0
    for which in range(2):
        if rng.random() >= p_inact:
            break
        s = int(rng.integers(0, mech.n_states))
        new = mech.n_states
        mech.n_states += 1
        mech.add(s, new, False)
        mech.add(new, s, False)
        n_inact += 1
        desc.append(f"reversible inactivation off state {s}")
    th = ThermalMechanism(mech, eyring=bool(rng.random() < p_eyring), saturating=bool(rng.random() < p_saturating),
                          n_inact=n_inact)
    desc.append("Eyring rates" if th.eyring else "Arrhenius rates")
    desc.append("saturating ligand" if th.saturating else "fixed ligand level")
    th.description = "; ".join(desc)
    return th


def thermal_form(th, max_terms=6, max_consts=8, why=None):
    """Canonical key (or None) and details for a mechanism with temperature-
    dependent rate constants (x1 = temperature; see the module docstring)."""
    why = {} if why is None else why
    Lg = sp.Symbol("L", positive=True)
    N, D = flux_polys(th.mech, Lg)
    if th.saturating:
        dn, dd = sp.degree(N, Lg), sp.degree(D, Lg)
        if N == 0 or dn < dd:
            why["reason"] = "zero_at_saturation"
            return None
        if dn > dd:            # cannot happen for a flux bounded by the enzyme total
            why["reason"] = "unbounded_at_saturation"
            return None
        N = sp.Poly(N, Lg).LC()
        D = sp.Poly(D, Lg).LC()
    else:
        N, D = N.subs(Lg, 1), D.subs(Lg, 1)
    if N == 0:
        why["reason"] = "zero_flux"
        return None
    N, D = sp.fraction(sp.cancel(sp.expand(N) / sp.expand(D)))
    ks = sorted(th.mech.symbols().values(), key=str)
    pn, pd = sp.Poly(N, *ks), sp.Poly(D, *ks)
    mon_n, mon_d = pn.monoms(), pd.monoms()
    if len(mon_n) + len(mon_d) > max_terms:
        why["reason"] = "too_many_terms"
        return None
    degs_n = {sum(mm) for mm in mon_n}
    degs_d = {sum(mm) for mm in mon_d}
    if len(degs_n) != 1 or len(degs_d) != 1:     # N and D are homogeneous in the rate constants
        raise AssertionError("flux polynomials not homogeneous")
    p = (degs_n.pop() - degs_d.pop()) if th.eyring else 0
    c = iter(sp.Symbol(f"c{i}", real=True) for i in range(1000))
    m0 = mon_d[0]

    def term(mm):
        if mm == m0:
            return next(c)            # same activation energy as the reference term: a plain constant
        return next(c) * sp.exp(next(c) / X)
    num = sum(term(mm) for mm in mon_n)
    den = 1 + sum(term(mm) for mm in mon_d if mm != m0)
    expr = X ** p * num / den
    sk, reason = _canon(expr, max_consts)
    if sk is None:
        why["reason"] = reason
        return None
    return {"key": sk.key, "n_consts": sk.n_consts, "family": str(expr), "n_num_terms": len(mon_n),
            "n_den_terms": len(mon_d), "x_power": p, "description": th.description,
            "n_states": th.mech.n_states, "edges": [[i, k, bool(bx)] for i, k, bx, _ in th.mech.edges],
            "eyring": th.eyring, "saturating": th.saturating, "n_inact": th.n_inact}


# ============================================================ corpora
CLASSES = ("enzyme", "compartment", "temperature")
X_MEANING = {"enzyme": "ligand concentration", "compartment": "time", "temperature": "absolute temperature"}


def generate_class(cls, n, seed=0, max_consts=8, max_terms=6):
    """n random mechanisms of one class -> (form records, Counter of drop reasons)."""
    from collections import Counter
    rng = np.random.default_rng(seed)
    out, drops = [], Counter()
    for i in range(n):
        why = {}
        if cls == "enzyme":
            rec = mechanism_form(random_mechanism(rng), rng, max_consts=max_consts, why=why)
        elif cls == "compartment":
            rec = compartment_form(random_compartment_system(rng), rng, max_consts=max_consts, why=why)
        elif cls == "temperature":
            rec = thermal_form(random_thermal_mechanism(rng), max_terms=max_terms, max_consts=max_consts, why=why)
        else:
            raise ValueError(cls)
        if rec is None:
            drops[why.get("reason", "unknown")] += 1
            continue
        rec["id"] = f"{cls}{i}"
        out.append(rec)
    return out, drops


def corpus_doc(cls, n, seed, max_consts=8, max_terms=6):
    """Corpus-style document (per_model, freq, records) for one class."""
    from collections import Counter
    recs, drops = generate_class(cls, n, seed, max_consts=max_consts, max_terms=max_terms)
    freq = Counter(r["key"] for r in recs)
    return {"source": f"random {cls} mechanisms (hsr.mechanisms, x1 = {X_MEANING[cls]})", "class": cls,
            "n_mechanisms": n, "seed": seed, "max_consts": max_consts, "max_terms": max_terms,
            "dropped": dict(drops), "n_dropped": sum(drops.values()),
            "per_model": {r["id"]: [r["key"]] for r in recs},
            "freq": dict(freq.most_common()), "records": recs}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Forms derived from random mechanisms")
    ap.add_argument("--class", dest="cls", choices=CLASSES, default="enzyme")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-consts", type=int, default=8)
    ap.add_argument("--max-terms", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = corpus_doc(a.cls, a.n, a.seed, a.max_consts, a.max_terms)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"{a.cls}: {len(doc['records'])} mechanisms, {len(doc['freq'])} distinct forms, "
          f"{doc['n_dropped']} dropped {doc['dropped']}")


if __name__ == "__main__":
    main()
