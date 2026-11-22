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
    local.update({f"c{i}": const_symbol(i) for i in range(20)})
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


def _absorb_once(expr, occ, fresh):
    if expr.is_Atom:
        return expr
    if not has_var(expr):
        return fresh() if _absorbable(expr, occ) else expr

    if isinstance(expr, (sp.Add, sp.Mul)):
        args = [_absorb_once(a, occ, fresh) for a in expr.args]
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
        if isinstance(arg, sp.Pow) and arg.exp.is_Number and has_var(arg.base):
            # log(f**p) = p log f; p becomes part of the multiplicative scale
            return arg.exp * sp.log(arg.base)
        return sp.log(arg)

    if isinstance(expr, sp.Pow):
        base = _absorb_once(expr.base, occ, fresh)
        ex = _absorb_once(expr.exp, occ, fresh)
        if ex.is_Number and isinstance(base, sp.Mul):
            free = [a for a in base.args if not has_var(a)]
            if free and _group_ok(free, occ, allow_numbers=True):
                rest = sp.Mul(*[a for a in base.args if has_var(a)])
                return fresh() * rest**ex
        if not has_var(base) and has_var(ex) and _group_ok([base], occ):
            # C**f(x) = exp(C*f(x))
            return sp.exp(fresh() * ex)
        return base**ex

    return expr.func(*[_absorb_once(a, occ, fresh) for a in expr.args])


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


def canonicalise(expr, n_vars: int = 1, simplify: bool = True, max_consts: int | None = 4,
                 positive: bool = True):
    """Return the canonical Skeleton of an expression (string or sympy), or
    None if it has no variable, too many constants, or unsupported operators."""
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
    # absorption can expose new merges (e.g. log(sqrt(x)) -> log(x)/2 inside a
    # product); repeat the expand-and-absorb selection until the form is stable
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
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I):
        return None
    if not has_var(expr):
        return None
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
