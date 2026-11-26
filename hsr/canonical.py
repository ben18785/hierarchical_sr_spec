"""Canonicalisation of skeletons and rule counts from expression trees.

Spec sections 3 ("Rule counts from an expression tree") and 4.

A skeleton is held as a sympy expression in which every free constant is a
real Symbol named ``c0, c1, ...`` and every input variable is a Symbol named
``x1, x2, ...``. Canonicalisation follows one principle: a rewrite may only be
applied if it leaves the *family* of functions unchanged (up to the sign and
range caveats of ``f(C) -> C`` that the spec accepts). In particular a
variable-free subexpression is replaced by a fresh constant only when it
contains a constant that occurs nowhere else in the expression; otherwise the
rewrite would add a degree of freedom.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import sympy as sp

CONST_RE = re.compile(r"^c\d+$")


class UnsupportedExpression(ValueError):
    """Raised when a sympy tree uses an operator the grammar cannot express."""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def make_vars(n_vars: int, positive: bool = True):
    kw = {"positive": True} if positive else {"real": True}
    return [sp.Symbol(f"x{i + 1}", **kw) for i in range(n_vars)]


def const_symbol(i: int) -> sp.Symbol:
    return sp.Symbol(f"c{i}", real=True)


def is_const(s) -> bool:
    return isinstance(s, sp.Symbol) and CONST_RE.match(s.name) is not None


def parse_skeleton(s: str, n_vars: int = 1, positive: bool = True) -> sp.Expr:
    """Parse a derivation string; each ``C`` becomes its own constant symbol."""
    counter = iter(range(10_000))
    s2 = re.sub(r"\bC\b", lambda m: f"c{next(counter)}", s)
    xs = make_vars(n_vars, positive)
    local = {x.name: x for x in xs}
    n_c = len(re.findall(r"\bc\d+\b", s2))
    for i in range(n_c):
        local[f"c{i}"] = const_symbol(i)
    local.update({"exp": sp.exp, "log": sp.log, "sin": sp.sin, "sqrt": sp.sqrt, "tanh": sp.tanh})
    return sp.sympify(s2, locals=local)


def _reparse(s: str, n_vars: int = 1, positive: bool = True) -> sp.Expr:
    xs = make_vars(n_vars, positive)
    local = {x.name: x for x in xs}
    n_c = max([int(m) for m in re.findall(r"\bc(\d+)\b", s)] + [19]) + 1
    local.update({f"c{i}": const_symbol(i) for i in range(n_c)})
    return sp.sympify(s, locals=local)


def variables_of(expr):
    return {s for s in expr.free_symbols if not is_const(s)}


def consts_of(expr):
    return {s for s in expr.free_symbols if is_const(s)}


def has_var(expr) -> bool:
    return bool(variables_of(expr))


# --------------------------------------------------------------------------
# Constant absorption
# --------------------------------------------------------------------------
class _Fresh:
    """Generator of fresh constant symbols that never clash with existing ones."""

    def __init__(self, start: int):
        self.i = start

    def __call__(self):
        self.i += 1
        return const_symbol(self.i)


def _occurrences(expr) -> Counter:
    cnt = Counter()
    for node in sp.preorder_traversal(expr):
        if is_const(node):
            cnt[node] += 1
    return cnt


def _absorbable(sub, occ) -> bool:
    """A var-free subexpression can become a fresh constant if it is not
    already a lone constant and contains a constant that occurs only in it."""
    if is_const(sub):
        return False
    inside = _occurrences(sub)
    return any(occ[c] == n for c, n in inside.items())


def _strip_exp_numbers(a):
    if isinstance(a, sp.exp) and isinstance(a.args[0], sp.Add):
        rest = [t for t in a.args[0].args if not t.is_Number]
        if len(rest) < len(a.args[0].args):
            return sp.exp(sp.Add(*rest))
    return a


def _absorb_once(expr, occ, fresh):
    if expr.is_Atom:
        return expr
    if not has_var(expr):
        return fresh() if _absorbable(expr, occ) else expr

    if isinstance(expr, sp.Mul):
        exps = [a for a in expr.args if isinstance(a, sp.exp)]
        if len(exps) > 1:
            # exp(a)*exp(b) = exp(a + b), so that the constants of a and b can
            # merge (exp(c2*x1)*exp(c3*x1) == exp(c*x1))
            others = [a for a in expr.args if not isinstance(a, sp.exp)]
            merged = sp.Mul(*others, sp.exp(sp.Add(*[a.args[0] for a in exps])))
            if merged != expr:
                return _absorb_once(merged, occ, fresh)

    if isinstance(expr, (sp.Add, sp.Mul)):
        args = [_absorb_once(a, occ, fresh) for a in expr.args]
        if isinstance(expr, sp.Mul) and any(is_const(a) and occ[a] == 1 for a in args):
            # c*exp(f + k) = (c e^k) exp(f): a free scale constant absorbs a
            # numeric addend inside an exp factor (c0*exp(c1*x + 1) == c0*exp(c1*x))
            args = [_strip_exp_numbers(a) for a in args]
        free = [a for a in args if not has_var(a)]
        rest = [a for a in args if has_var(a)]
        if free and not (len(free) == 1 and is_const(free[0])) and _group_ok(free, occ):
            return expr.func(*rest, fresh())
        if isinstance(expr, sp.Add):
            combined = _combine_like_terms(args, occ, fresh)
            if combined is not None:
                return combined
        return expr.func(*args)

    if isinstance(expr, sp.exp):
        arg = _absorb_once(expr.args[0], occ, fresh)
        powed = _exp_log_to_pow(arg)
        if powed is not None:
            return powed
        if isinstance(arg, sp.Add):
            free = [a for a in arg.args if not has_var(a)]
            if free and _group_ok(free, occ):
                rest = sp.Add(*[a for a in arg.args if has_var(a)])
                return fresh() * sp.exp(rest)
        return sp.exp(arg)

    if isinstance(expr, sp.log):
        arg = _absorb_once(expr.args[0], occ, fresh)
        if isinstance(arg, sp.Mul):
            free = [a for a in arg.args if not has_var(a)]
            if free and _group_ok(free, occ, allow_numbers=True):
                rest = sp.Mul(*[a for a in arg.args if has_var(a)])
                return sp.log(rest) + fresh()
        if isinstance(arg, sp.Pow) and not has_var(arg.exp) and has_var(arg.base):
            # log(f**p) = p log f; p becomes part of the multiplicative scale
            return arg.exp * sp.log(arg.base)
        return sp.log(arg)

    if isinstance(expr, sp.Pow):
        base = _absorb_once(expr.base, occ, fresh)
        ex = _absorb_once(expr.exp, occ, fresh)
        if not has_var(ex) and isinstance(base, sp.Mul):
            # (c*f)**p = c**p * f**p; c**p is a fresh constant when c is
            # unshared, for numeric or constant p ((c0*x)**c1 == c*x**c1)
            free = [a for a in base.args if not has_var(a)]
            if free and _group_ok(free, occ, allow_numbers=True):
                rest = sp.Mul(*[a for a in base.args if has_var(a)])
                return fresh() * rest**ex
        if not has_var(base) and has_var(ex) and _group_ok([base], occ):
            # C**f(x) = exp(C*f(x))
            return sp.exp(fresh() * ex)
        return base**ex

    return expr.func(*[_absorb_once(a, occ, fresh) for a in expr.args])


def _log_term(t):
    """(A, g) if t is A*log(g) with A free of variables and g not, else None."""
    if isinstance(t, sp.log) and has_var(t.args[0]):
        return sp.Integer(1), t.args[0]
    if isinstance(t, sp.Mul):
        logs = [a for a in t.args if isinstance(a, sp.log) and has_var(a.args[0])]
        if len(logs) == 1:
            A = sp.Mul(*[a for a in t.args if a is not logs[0]])
            if not has_var(A):
                return A, logs[0].args[0]
    return None


def _exp_log_to_pow(arg):
    """exp(A*log(g) + R) = g**A * exp(R) (g > 0 wherever log(g) is real), or
    None if no term of ``arg`` is of that kind. Without this rule power laws
    written as c**log(x) (Lomolino's slope**log(xmid/x)) end as
    exp(c*log(x1)) instead of x1**c."""
    terms = arg.args if isinstance(arg, sp.Add) else (arg,)
    pows, rest = [], []
    for t in terms:
        lt = _log_term(t)
        if lt is None:
            rest.append(t)
        else:
            pows.append(lt[1] ** lt[0])
    if not pows:
        return None
    out = sp.Mul(*pows)
    return out * sp.exp(sp.Add(*rest)) if rest else out


def _split_term(t):
    """term -> (var-free factor, variable part)."""
    if isinstance(t, sp.Mul):
        free = [a for a in t.args if not has_var(a)]
        var = [a for a in t.args if has_var(a)]
        return sp.Mul(*free), sp.Mul(*var)
    return (sp.Integer(1), t) if has_var(t) else (t, sp.Integer(1))


def _combine_like_terms(args, occ, fresh):
    """c1*g - g -> c*g: merge terms sharing a variable part when the merged
    coefficient contains an unshared constant. Returns None if nothing merged."""
    groups = {}
    order = []
    for a in args:
        k, m = _split_term(a)
        key = sp.srepr(m)
        if key not in groups:
            groups[key] = (m, [])
            order.append(key)
        groups[key][1].append(k)
    changed = False
    out = []
    for key in order:
        m, ks = groups[key]
        if len(ks) > 1 and _group_ok(ks, occ):
            out.append(fresh() * m)
            changed = True
        else:
            out.extend(k * m for k in ks)
    return sp.Add(*out) if changed else None


def _group_ok(free, occ, allow_numbers=False) -> bool:
    """Whether a list of var-free factors/terms can be pulled out as a fresh
    constant: true if some constant in it is unshared."""
    group_occ = Counter()
    for a in free:
        group_occ.update(_occurrences(a))
    return any(occ[c] == n for c, n in group_occ.items())


def _normalise_denominator(expr, fresh):
    """x/(a*x + b) -> d*x/(x + e): divide through by the constant multiplying
    the first variable term of a denominator when that constant is a single
    unshared symbol. The map (a, b) -> (1/a, b/a) is a bijection, so the
    family is unchanged. Applied to each term of a sum separately (fractions
    are never combined)."""
    if isinstance(expr, sp.Add):
        return sp.Add(*[_normalise_denominator(t, fresh) for t in expr.args])
    if not isinstance(expr, (sp.Mul, sp.Pow)):
        return expr
    num, den = sp.fraction(expr)
    if den == 1 or not isinstance(den, sp.Add):
        return expr
    occ = _occurrences(expr)
    terms = sorted(den.args, key=sp.default_sort_key)
    var_terms = [t for t in terms if has_var(t)]
    if not var_terms:
        return expr
    if any(_free_factor(t) == 1 for t in var_terms):
        return expr  # a variable term already has unit coefficient
    k = _free_factor(var_terms[0])
    if is_const(k) and occ[k] == 1:
        d = fresh()
        new = (num * d) / sp.expand(den * d)
        return new.subs(k, 1 / d)
    if k.is_Number and k != 0 and _has_unshared_factor(num, occ):
        # x/(c - x) = -x/(x - c): a numeric leading coefficient (often a sign)
        # can be moved into a free multiplicative constant of the numerator
        return (num / k) / sp.expand(den / k)
    return expr


def _has_unshared_factor(num, occ):
    factors = num.args if isinstance(num, sp.Mul) else (num,)
    return any(is_const(f) and occ[f] == 1 for f in factors)


def _free_factor(term):
    if isinstance(term, sp.Mul):
        return sp.Mul(*[a for a in term.args if not has_var(a)])
    return sp.Integer(1) if has_var(term) else term


def _rename(expr) -> sp.Expr:
    """Rename constants to c0..c{k-1}: of all k! namings, keep the one whose
    srepr is smallest. This makes the identity independent of how sympy's
    name-dependent argument ordering happened to fall."""
    consts = sorted(consts_of(expr), key=lambda c: c.name)
    if not consts:
        return expr
    tmp = {c: sp.Symbol(f"_t{i}", real=True) for i, c in enumerate(consts)}
    base = expr.xreplace(tmp)
    best = None
    k = len(consts)
    perms = itertools.permutations(range(k)) if k <= 5 else [tuple(range(k))]
    for perm in perms:
        cand = base.xreplace({sp.Symbol(f"_t{i}", real=True): const_symbol(perm[i]) for i in range(k)})
        key = sp.srepr(cand)
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1]


@dataclass(frozen=True)
class Skeleton:
    key: str          # identity string (canonical srepr-free printed form)
    expr: sp.Expr     # canonical sympy expression with constants c0..c{k-1}
    n_consts: int

    @property
    def display(self) -> str:
        s = str(self.expr)
        occ = _occurrences(self.expr)
        if all(n == 1 for n in occ.values()):
            s = re.sub(r"\bc\d+\b", "C", s)
        return s


def _settle(expr, n_vars=1, positive=True):
    """Repeat the expand-and-absorb selection until the form is stable:
    absorption can expose new merges (e.g. log(sqrt(x)) -> log(x)/2 inside a
    product)."""
    for _ in range(6):
        nxt = _best_of([expr])
        if nxt is None:
            break
        # the key is a printed string, so it must survive re-parsing (sympy
        # re-evaluates e.g. 2*(c + g) into 2*c + 2*g when parsing)
        reparsed = _reparse(str(nxt), n_vars, positive)
        if sp.srepr(reparsed) != sp.srepr(nxt):
            nxt = _best_of([reparsed]) or nxt
        if sp.srepr(nxt) == sp.srepr(expr) and str(_reparse(str(nxt), n_vars, positive)) == str(nxt):
            break
        expr = nxt
    return expr


def canonicalise(expr, n_vars: int = 1, simplify: bool = True, max_consts: int | None = 4,
                 positive: bool = True, reduce: bool = True):
    """Return the canonical Skeleton of an expression (string or sympy), or
    None if it has no variable, too many constants, or unsupported operators.

    With ``reduce`` (default) a form whose constants are not identifiable is
    reparametrised with fewer constants where that can be done soundly
    (``reduce_redundant``)."""
    if isinstance(expr, str):
        expr = parse_skeleton(expr, n_vars, positive)
    if not has_var(expr):
        return None
    if any(e.has(sp.zoo, sp.nan, sp.oo, -sp.oo) for e in [expr]):
        return None
    starts = [expr]
    if simplify:
        try:
            simp = sp.simplify(expr)
            if not has_var(simp):
                return None
            if simp != expr:
                starts.append(simp)
        except Exception:  # pragma: no cover - sympy edge cases
            pass
    expr = _best_of(starts)
    if expr is None:
        return None
    expr = _settle(expr, n_vars, positive)
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I):
        return None
    if not has_var(expr):
        return None
    if reduce:
        expr = reduce_redundant(expr, n_vars, positive)
    n_c = len(consts_of(expr))
    if max_consts is not None and n_c > max_consts:
        return None
    return Skeleton(key=str(expr), expr=expr, n_consts=n_c)


def _together(e):
    """Combine nested fractions into one (c/(a + 1/x) -> c*x/(a*x + 1))."""
    try:
        t = sp.together(e)
        return t if t != e else None
    except Exception:  # pragma: no cover
        return None


def _n_divisions(e):
    """Number of reciprocal nodes; breaks ties in favour of flat fractions."""
    return sum(1 for node in sp.preorder_traversal(e)
               if isinstance(node, sp.Pow) and node.exp.is_Number and node.exp < 0)


def _shallow_expand(e):
    return sp.expand(e, deep=True, power_exp=False, power_base=False, log=False, multinomial=False)


def _best_of(starts):
    """Absorb each start and its shallow expansion; keep the cheapest result
    by (number of constants, number of divisions, structural size, srepr).
    Each start is also tried with its fractions combined."""
    pool = []
    for st in list(starts):
        try:
            # x*x**c -> x**(c + 1), which absorption turns into x**c
            ps = sp.powsimp(st, combine="exp")
            if ps != st:
                starts = list(starts) + [ps]
        except Exception:  # pragma: no cover
            pass
    for st in starts:
        for variant in (st, _together(st)):
            if variant is None:
                continue
            pool.append(variant)
            try:
                ex = _shallow_expand(variant)
                if ex != variant:
                    pool.append(ex)
            except Exception:  # pragma: no cover
                pass
    best = None
    for start in pool:
        cand = _absorb_to_fixed_point(start)
        if cand is None or not has_var(cand):
            continue
        for c2 in (cand, _absorb_to_fixed_point(_shallow_expand(cand))):
            if c2 is None or not has_var(c2):
                continue
            try:
                size = structural_size(rule_counts(c2))
            except UnsupportedExpression:
                continue
            score = (len(consts_of(c2)), _n_divisions(c2), size, sp.srepr(c2))
            if best is None or score < best[0]:
                best = (score, c2)
    return None if best is None else best[1]


def skeleton_score(expr):
    """Preference order used to pick one representative of a family."""
    return (len(consts_of(expr)), _n_divisions(expr), structural_size(rule_counts(expr)), sp.srepr(expr))


def _absorb_to_fixed_point(expr):
    prev = None
    for _ in range(8):
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I):
            return None
        fresh = _Fresh(max([int(c.name[1:]) for c in consts_of(expr)] + [0]) + 100)
        # C**f(x) = exp(log(C) f(x)) also when C is shared, provided it is
        # used only as such a base (c**x1*d/(c**x1*d + e)); the unshared case
        # is the C**f(x) rule of _absorb_once
        expr = _const_bases_to_exp(expr) or expr
        occ = _occurrences(expr)
        expr = _absorb_once(expr, occ, fresh)
        expr = _normalise_denominator(expr, fresh)
        expr = _rename(expr)
        key = sp.srepr(expr)
        if key == prev:
            break
        prev = key
    return expr


# --------------------------------------------------------------------------
# Redundant (non-identifiable) constants
# --------------------------------------------------------------------------
# The absorption rules above only merge constants that occur once. Forms such
# as c0*x1 + c1*x1 inside a repeated factor, c0*(c1*x1 + c2)**2 or
# c0*x1/(c0*x1 + c1) keep more constants than the family has dimensions. A
# form is redundant when the Jacobian of f(x; c) in c is rank deficient at
# generic c; ``reduce_redundant`` then looks for a sound reparametrisation
# with fewer constants:
#   * constant-coefficient symmetries: a vector field v = sum (w_i + z_i c_i)
#     d/dc_i with J v = 0 identically. Its flow leaves f unchanged, so a
#     constant k with v_k = w_k (pure shift) can be fixed at 0, or with
#     v_k = z_k c_k (pure scaling) at 1 (for c_k > 0, the usual sign caveat);
#   * identities that expose such symmetries to the absorption rules:
#     (k*g + h)**p = k**p (g + h/k)**p and log(k*g + h) = log(k) + log(g + h/k)
#     (k > 0), c**f(x) = exp(log(c) f(x)) for a constant used only as a base,
#     and full expansion of integer powers.
# Every symmetry is checked numerically at fresh constants and every identity
# at random points before use; a candidate is kept only if it has strictly
# fewer constants. Forms with identifiable constants are never touched.
_ID_X = np.linspace(0.2, 3.0, 40)
_REDUCE_MAX_CONSTS = 16
_REDUCE_MAX_OPS = 400
_REDUCE_SMALL_OPS = 60
_FINAL_SIMPLIFY_OPS = 150


def _sorted_consts(expr):
    return sorted(consts_of(expr), key=lambda c: c.name)


def _draws(k, seed, n=3):
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.5, 2.0, k) for _ in range(n)]


class _Numeric:
    """f and its gradient in the constants as numpy callables (one variable)."""

    def __init__(self, expr):
        (x,) = tuple(variables_of(expr))
        self.cs = _sorted_consts(expr)
        self.k = len(self.cs)
        self._f = sp.lambdify([x] + self.cs, expr, "numpy")
        self._g = sp.lambdify([x] + self.cs, [sp.diff(expr, c) for c in self.cs], "numpy")

    def f(self, cv, xs=_ID_X):
        with np.errstate(all="ignore"):
            try:
                return np.broadcast_to(np.asarray(self._f(xs, *cv), dtype=float), xs.shape)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return np.full(xs.shape, np.nan)

    def jac(self, cv):
        """(n_x, k) Jacobian; rows with a non-finite entry are dropped."""
        with np.errstate(all="ignore"):
            try:
                cols = self._g(_ID_X, *cv)
                M = np.array([np.broadcast_to(np.asarray(c, dtype=float), _ID_X.shape) for c in cols]).T
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return np.zeros((0, self.k))
        return M[np.all(np.isfinite(M), axis=1)]


def _numeric(expr):
    if len(variables_of(expr)) != 1 or not consts_of(expr):
        return None
    try:
        return _Numeric(expr)
    except Exception:  # pragma: no cover - lambdify edge cases
        return None


def is_redundant(expr, tol=1e-8, num=None, equilibrate=True) -> bool:
    """True if the Jacobian of f(x; c) in c is rank deficient at each of three
    random constant vectors. With ``equilibrate=False`` this is exactly the
    test of workflow/scripts/holdout_report.py; scaling each x row to unit
    norm first removes its false positives on forms whose value spans many
    orders of magnitude over the grid (c0*exp(c1/x1**5) + c2)."""
    num = num or _numeric(expr)
    if num is None or num.k < 2:
        return False
    for cv in _draws(num.k, 0):
        M = num.jac(cv)
        if M.shape[0] < num.k:
            return False
        if equilibrate:
            M = _equilibrate_rows(M)
            if M.shape[0] < num.k:
                return False
        sv = np.linalg.svd(M / (np.linalg.norm(M, axis=0) + 1e-300), compute_uv=False)
        if sv[-1] >= tol * sv[0]:
            return False
    return True


def _affine_symmetries(num, tol=1e-9):
    """[(k, kind)]: constant k can be fixed (kind 'shift': at 0, 'scale': at 1)
    along a constant-coefficient symmetry v = sum (w_i + z_i c_i) d/dc_i of f."""
    k = num.k
    blocks = []
    for cv in _draws(k, 1, n=2 * k + 2):
        M = num.jac(cv)
        blocks.append(np.hstack([M, M * cv[None, :]]))
    A = np.vstack(blocks)
    if A.shape[0] < 2 * k:
        return []
    A = _equilibrate_rows(A)
    D = np.linalg.norm(A, axis=0)
    D[D == 0] = 1.0
    _, s, Vt = np.linalg.svd(A / D, full_matrices=True)
    s = np.concatenate([s, np.zeros(Vt.shape[0] - len(s))])
    null = Vt[s < tol * max(s[0], 1e-300)]
    if not len(null):
        return []
    B = (null / D).T                          # columns: (w, z) null vectors
    out = []
    checks = _draws(k, 2, n=2)
    for i in range(k):
        for kind, zero, one in (("shift", k + i, i), ("scale", i, k + i)):
            S = B[[zero, one]]
            a, *_ = np.linalg.lstsq(S, np.array([0.0, 1.0]), rcond=None)
            if not np.allclose(S @ a, [0.0, 1.0], atol=1e-8):
                continue
            v = B @ a
            w, z = v[:k], v[k:]
            # check the finite flow, not just the field: moving c_i to its
            # fixed value along the flow must leave f unchanged at every x
            ok = True
            for cv in checks:
                t = -cv[i] if kind == "shift" else -np.log(cv[i])
                if not _invariant(num, cv, _flow(cv, w, z, t)):
                    ok = False
                    break
            if ok:
                out.append((i, kind))
    return out


def _equilibrate_rows(A):
    """Scale each row to unit norm (the rank is unchanged; rows where f is
    huge no longer swamp the others)."""
    r = np.linalg.norm(A, axis=1)
    keep = r > 0
    return A[keep] / r[keep, None]


def _flow(cv, w, z, t):
    """Constants after time t along dc_i/dt = w_i + z_i c_i."""
    out = np.empty_like(cv)
    for i in range(len(cv)):
        if abs(z[i]) > 1e-10:
            e = np.exp(z[i] * t)
            out[i] = cv[i] * e + w[i] / z[i] * (e - 1.0)
        else:
            out[i] = cv[i] + w[i] * t
    return out


def _invariant(num, c_a, c_b, rtol=1e-6) -> bool:
    """f(x; c_a) == f(x; c_b) pointwise (relative), on at least half of the grid."""
    ya, yb = num.f(c_a), num.f(c_b)
    fin = np.isfinite(ya) & np.isfinite(yb)
    if fin.sum() < len(_ID_X) // 2:
        return False
    ya, yb = ya[fin], yb[fin]
    scale = np.abs(ya) + 1e-12 * (np.abs(ya).max() + 1e-300)
    return bool(np.all(np.abs(ya - yb) <= rtol * scale))


def _same_function(a, b, n=3) -> bool:
    """a and b (same constants) agree numerically at random positive constants."""
    cs = _sorted_consts(a + b)
    na, nb = _numeric_on(a, cs), _numeric_on(b, cs)
    if na is None or nb is None:
        return False
    n_ok = 0
    for cv in _draws(len(cs), 3, n):
        ya, yb = na(cv), nb(cv)
        fin = np.isfinite(ya) & np.isfinite(yb)
        if fin.sum() < len(_ID_X) // 2:
            continue
        ya, yb = ya[fin], yb[fin]
        if not np.all(np.abs(ya - yb) <= 1e-8 * (np.abs(ya) + 1e-12 * (np.abs(ya).max() + 1e-300))):
            return False
        n_ok += 1
    return n_ok > 0


def _numeric_on(expr, cs):
    xs = list(variables_of(expr))
    if len(xs) != 1:
        return None
    try:
        fn = sp.lambdify([xs[0]] + list(cs), expr, "numpy")
    except Exception:  # pragma: no cover
        return None

    def call(cv):
        with np.errstate(all="ignore"):
            try:
                return np.broadcast_to(np.asarray(fn(_ID_X, *cv), dtype=float), _ID_X.shape)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return np.full(_ID_X.shape, np.nan)
    return call


def _factor_coefficient(node):
    """(k*g + h)**p -> k**p*(g + h/k)**p and log(k*g + h) -> log(k) + log(g + h/k)
    for each variable term k*g whose coefficient k holds a constant; one
    rewritten node per such term."""
    if isinstance(node, sp.Pow):
        base, p = node.base, node.exp
        if not (isinstance(base, sp.Add) and has_var(base) and not has_var(p)):
            return []
    elif isinstance(node, sp.log):
        base, p = node.args[0], None
        if not (isinstance(base, sp.Add) and has_var(base)):
            return []
    else:
        return []
    out = []
    for t in sorted(base.args, key=sp.default_sort_key):
        if not has_var(t):
            continue
        k = _free_factor(t)
        if k == 1 or not consts_of(k):
            continue
        inner = sp.Add(*[a / k for a in base.args])
        out.append(k**p * inner**p if p is not None else sp.log(k) + sp.log(inner))
    return out


def _pretransforms(expr, max_nodes=6):
    """Identity rewrites (and the c -> log c reparametrisation) that can let
    the absorption rules merge constants."""
    out = []
    nodes = [n for n in sp.preorder_traversal(expr) if isinstance(n, (sp.Pow, sp.log))]
    singles, firsts = [], {}
    for n in nodes:
        for new in _factor_coefficient(n):
            singles.append((n, new))
            firsts.setdefault(n, new)
    if firsts:
        out.append(expr.xreplace(firsts))
    out.extend(expr.xreplace({n: new}) for n, new in singles[:max_nodes])
    out = [e for e in out if _same_function(expr, e)]
    try:
        ex = sp.expand(expr, power_exp=False, power_base=False, log=False)
        if ex != expr and sp.count_ops(ex) <= 4 * sp.count_ops(expr) + 10 and _same_function(expr, ex):
            out.append(ex)
    except Exception:  # pragma: no cover
        pass
    out.extend(e2 for e2 in (_const_bases_to_exp(e) for e in [expr] + out[:1]) if e2 is not None)
    seen, uniq = set(), []
    for e in out:
        key = sp.srepr(e)
        if key not in seen and e != expr:
            seen.add(key)
            uniq.append(e)
    return uniq


def _partial_fractions(expr, max_degree=4):
    """Partial fractions of a rational function of the variable (sums of
    Michaelis-Menten terms share their constant parts: (a + b x)/(x + c) +
    (d + e x)/(x + f) has 5 dimensions, not 6), or None."""
    xs = list(variables_of(expr))
    if len(xs) != 1 or sp.count_ops(expr) > 200:
        return None
    x = xs[0]
    try:
        if not expr.is_rational_function(x):
            return None
        den = sp.fraction(sp.together(expr))[1]
        if not 1 <= sp.degree(den, x) <= max_degree:
            return None
        ap = sp.apart(expr, x)
        return None if ap == expr else ap
    except Exception:  # pragma: no cover - sympy polys edge cases
        return None


def _const_bases_to_exp(expr):
    """c**f(x) -> exp(c*f(x)) (c := log c) for every constant used only as the
    base of powers with a variable exponent; None if there is none."""
    tmp = sp.Symbol("_tb", real=True)
    changed = False
    for c in _sorted_consts(expr):
        new = expr.replace(lambda n: isinstance(n, sp.Pow) and n.base == c and has_var(n.exp),
                           lambda n: sp.exp(tmp * n.exp))
        if new != expr and c not in new.free_symbols:
            expr = new.xreplace({tmp: c})
            changed = True
    return expr if changed else None


def _settled_candidate(e, n_vars, positive):
    try:
        r = _best_of([e])
        if r is None:
            return None
        r = _settle(r, n_vars, positive)
        if r.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I) or not has_var(r):
            return None
        rule_counts(r)
        return r
    except (UnsupportedExpression, RecursionError, ValueError, TypeError, ZeroDivisionError):
        return None


def reduce_redundant(expr, n_vars=1, positive=True, max_rounds=None):
    """Reparametrise a canonical form whose constants are not identifiable
    with fewer constants (see the comment above). Identifiable forms, forms
    with more than one variable and very large forms are returned unchanged."""
    if n_vars != 1:
        return expr
    rounds = max_rounds if max_rounds is not None else len(consts_of(expr))
    for _ in range(rounds):
        k = len(consts_of(expr))
        if k < 2 or k > _REDUCE_MAX_CONSTS or sp.count_ops(expr) > _REDUCE_MAX_OPS:
            break
        num = _numeric(expr)
        if num is None or not is_redundant(expr, num=num):
            break
        # fix constants along symmetries: each possible first choice (only the
        # first one for large forms), then greedily until none is left
        syms = _affine_symmetries(num)
        small = sp.count_ops(expr) <= _REDUCE_SMALL_OPS
        sym_cands = []
        for j, kind in (syms if small else syms[:1]):
            first = expr.xreplace({num.cs[j]: sp.Integer(0 if kind == "shift" else 1)})
            sym_cands.append(_fix_symmetries(first))
        best = _best_reduction(sym_cands, k, n_vars, positive)
        if best is None or small:
            # identities that let the absorption rules merge constants
            alt = _best_reduction(_pretransforms(expr), k, n_vars, positive)
            if alt is not None and (best is None or alt[0] < best[0]):
                best = alt
        # partial fractions change the shape of the form (a sum of fractions
        # instead of one), so they are used only if they remove more constants
        ap = _partial_fractions(expr)
        if ap is not None and _same_function(expr, ap):
            n_best = best[0][0] if best is not None else k
            alt = _best_reduction([ap], n_best, n_vars, positive)
            if alt is not None:
                best = alt
        if best is None:
            break
        expr = best[1]
    return expr


_REDUCED: dict = {}


def reduce_key(key: str, n_vars: int = 1, positive: bool = True) -> str:
    """Final form of a canonical key (cached per process): canonicalise it
    again until it is a fixed point, with redundant constants removed
    (``reduce_redundant``). A key reached from a raw law is not always a fixed
    point of canonicalise (sp.simplify of the key can find a cheaper form),
    so two laws of one family could keep different keys. Corpus builders call
    this on their keys outside the per-law time limits, so that it never
    makes a law time out. sp.simplify is skipped for keys of more than
    _FINAL_SIMPLIFY_OPS operations (a size rule, not a time limit, so that
    the result is reproducible)."""
    hit = _REDUCED.get((key, n_vars, positive))
    if hit is not None:
        return hit
    seen = [key]
    cur = key
    for _ in range(4):
        expr = _reparse(cur, n_vars, positive)
        simp = sp.count_ops(expr) <= _FINAL_SIMPLIFY_OPS
        try:
            sk = canonicalise(expr, n_vars, simplify=simp, max_consts=None, positive=positive)
        except Exception:  # pragma: no cover - sympy edge cases
            sk = None
        if sk is None or sk.key == cur:
            break
        if sk.key in seen:      # a cycle: keep its cheapest member
            cyc = seen[seen.index(sk.key):]
            cur = min(cyc, key=lambda k: skeleton_score(_reparse(k, n_vars, positive)))
            break
        seen.append(sk.key)
        cur = sk.key
    _REDUCED[(key, n_vars, positive)] = cur
    return cur


def _best_reduction(cands, k, n_vars, positive):
    """(score, form) of the best settled candidate with fewer than k constants."""
    best = None
    seen = set()
    for c in cands:
        key = sp.srepr(c)
        if key in seen:
            continue
        seen.add(key)
        r = _settled_candidate(c, n_vars, positive)
        if r is None or len(consts_of(r)) >= k:
            continue
        score = skeleton_score(r)
        if best is None or score < best[0]:
            best = (score, r)
    return best


def _fix_symmetries(expr, max_steps=_REDUCE_MAX_CONSTS):
    """Fix one constant per remaining constant-coefficient symmetry (first in
    name order), recomputing the symmetries after each step."""
    for _ in range(max_steps):
        num = _numeric(expr)
        if num is None or num.k < 2:
            break
        syms = _affine_symmetries(num)
        if not syms:
            break
        j, kind = syms[0]
        expr = expr.xreplace({num.cs[j]: sp.Integer(0 if kind == "shift" else 1)})
    return expr


# --------------------------------------------------------------------------
# Rule counts (spec section 3)
# --------------------------------------------------------------------------
class _SqrtItem:
    """Marker for a sqrt(base) factor produced when splitting half-integer powers."""

    def __init__(self, base):
        self.base = base


def rule_counts(expr) -> Counter:
    """Counter over (nonterminal, option) for the unique derivation of expr.

    Conventions (spec section 3): an n-term sum uses n-1 of E+F / E-F and one
    E->F; a product/quotient of k factors uses k-1 of F*T / F/T and one F->T;
    integer power x**p is p-1 multiplications; a lone sign (-1) counts as
    multiplication by a constant. Numeric literals other than -1 count as a
    constant atom. Expressions with no numerator factor (1/x) are counted as
    C/x. Non-integer powers other than halves are counted as exp(p*log(b)).
    """
    cnt = Counter()
    _count_E(expr, cnt)
    return cnt


def _is_negative_term(t) -> bool:
    if isinstance(t, sp.Mul):
        c = t.args[0]
        return c.is_Number and c < 0
    return t.is_Number and t < 0


def _count_E(expr, cnt):
    if isinstance(expr, sp.Add):
        terms = list(expr.as_ordered_terms())
        pos = [t for t in terms if not _is_negative_term(t)]
        neg = [t for t in terms if _is_negative_term(t)]
        cnt[("E", "F")] += 1
        if pos:
            head = pos[0]
            _count_F(head, cnt)
            rest = pos[1:]
        else:
            head = neg[0]
            _count_F(head, cnt)  # its -1 is counted as a constant factor
            neg = neg[1:]
            rest = []
        for t in rest:
            cnt[("E", "E+F")] += 1
            _count_F(t, cnt)
        for t in neg:
            cnt[("E", "E-F")] += 1
            _count_F(-t, cnt)
        return
    cnt[("E", "F")] += 1
    _count_F(expr, cnt)


def _split_factors(expr):
    """Return (numerator_items, denominator_items) of a product."""
    factors = list(expr.args) if isinstance(expr, sp.Mul) else [expr]
    num, den = [], []
    for f in factors:
        if f.is_Number:
            if f == 1:
                continue
            if f == -1:
                num.append(sp.Symbol("c_sign", real=True))
                continue
            if f.is_Rational and abs(f.p) == 1 and f.q > 1:
                den.append(sp.Integer(f.q))
                if f.p < 0:
                    num.append(sp.Symbol("c_sign", real=True))
                continue
            num.append(f)
            continue
        if isinstance(f, sp.Pow) and f.exp.is_Rational:
            p, q = int(f.exp.p), int(f.exp.q)
            target = num if p > 0 else den
            p = abs(p)
            if q == 1:
                target.extend([f.base] * p)
                continue
            if q == 2:
                target.extend([f.base] * (p // 2))
                target.append(_SqrtItem(f.base))
                continue
        num.append(f)
    return num, den


def _count_F(expr, cnt):
    num, den = _split_factors(expr)
    if not num:
        num = [sp.Symbol("c_one", real=True)]  # 1/x counted as C/x
    cnt[("F", "T")] += 1
    cnt[("F", "F*T")] += len(num) - 1
    cnt[("F", "F/T")] += len(den)
    for item in num + den:
        _count_T(item, cnt)


def _count_T(item, cnt):
    if isinstance(item, _SqrtItem):
        cnt[("T", "R(E)")] += 1
        cnt[("R", "sqrt")] += 1
        _count_E(item.base, cnt)
        return
    if isinstance(item, sp.Symbol):
        if is_const(item) or item.name.startswith("c_"):
            cnt[("T", "C")] += 1
        else:
            cnt[("T", "V")] += 1
            cnt[("V", item.name)] += 1
        return
    if item.is_Number:
        cnt[("T", "C")] += 1
        return
    if isinstance(item, (sp.exp, sp.log, sp.sin, sp.tanh)):
        cnt[("T", "R(E)")] += 1
        cnt[("R", type(item).__name__)] += 1
        _count_E(item.args[0], cnt)
        return
    if isinstance(item, sp.Pow):
        if item.exp.is_Rational and item.exp.q in (1, 2):
            cnt[("T", "(E)")] += 1
            _count_E(item, cnt)
            return
        if is_const(item.exp):
            # b**C: the power production T -> (E)^C
            cnt[("T", "(E)^C")] += 1
            _count_E(item.base, cnt)
            return
        # b**p with p non-half or symbolic: exp(p*log(b))
        cnt[("T", "R(E)")] += 1
        cnt[("R", "exp")] += 1
        cnt[("E", "F")] += 1
        cnt[("F", "T")] += 1
        cnt[("F", "F*T")] += 1
        ex = item.exp
        if is_const(ex) or ex.is_Number:
            cnt[("T", "C")] += 1
        else:
            cnt[("T", "(E)")] += 1
            _count_E(ex, cnt)
        cnt[("T", "R(E)")] += 1
        cnt[("R", "log")] += 1
        _count_E(item.base, cnt)
        return
    if isinstance(item, (sp.Add, sp.Mul)):
        cnt[("T", "(E)")] += 1
        _count_E(item, cnt)
        return
    raise UnsupportedExpression(f"cannot count rules for {item!r} ({type(item).__name__})")


def counts_array(expr, grammar) -> np.ndarray:
    """Flat count vector (grammar.rule_index order) for a sympy skeleton."""
    return grammar.counts_vector(rule_counts(expr))


def structural_size(counts: Counter) -> int:
    return sum(n for (nt, _), n in counts.items() if nt in ("E", "F", "T"))
