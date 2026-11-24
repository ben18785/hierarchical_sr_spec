"""Expression trees in fnprior's coding, for the tree priors of hsr.pyp and
hsr.adaptor.

A tree is a nested tuple (symbol, child, ...) over the symbols

    leaves  a (any constant), x (the variable)
    unary   sqrt exp log sin tanh inv
    binary  + - * / pow

i.e. fnprior's basis (hsr.fnprior_tools.BASIS) with constants coded "a"
(code_constants=True). fnprior's right-child vocabulary also has "y" (a
second, different variable); with one variable it never occurs, so it is
left out here (fnprior's mass on "y" is mass outside the one-variable space).

Every node is generated top-down given its *position*

    Pos(side, anc, sib, depth)

side "L" (the root or a first child) or "R" (a second child), anc the
ancestor symbols (most distant first, left-padded with None to length
NMAX), sib the left sibling's symbol (second children only) and depth the
node's depth (root 0). fnprior's n = 2 events are: first child (and root)
given (grandparent, parent); second child given (parent, left sibling).
"""
from __future__ import annotations

import re
from collections import namedtuple

LEAVES = ("a", "x")
UNARY = ("sqrt", "exp", "log", "sin", "tanh", "inv")
BINARY = ("+", "-", "*", "/", "pow")
V = LEAVES + UNARY + BINARY
ARITY = {**{s: 0 for s in LEAVES}, **{s: 1 for s in UNARY}, **{s: 2 for s in BINARY}}
NMAX = 3

Pos = namedtuple("Pos", "side anc sib depth")
ROOT = Pos("L", (None,) * NMAX, None, 0)


def child_pos(pos: Pos, sym: str, i: int, left_sym: str | None = None) -> Pos:
    """Position of child i (0 or 1) of a node with symbol ``sym`` at ``pos``;
    a second child also needs its left sibling's symbol."""
    anc = (pos.anc + (sym,))[-NMAX:]
    if i == 0:
        return Pos("L", anc, None, pos.depth + 1)
    return Pos("R", anc, left_sym, pos.depth + 1)


# ------------------------------------------------------------ construction
def from_labels(labels):
    """Tree from fnprior's prefix labels (SymbolCoder.equation2labels):
    x0, x1, ... -> x; a, a0, a1, ... -> a; operators unchanged."""
    syms = []
    for lab in labels:
        if re.fullmatch(r"x\d*", lab):
            syms.append("x")
        elif re.fullmatch(r"a\d*", lab):
            syms.append("a")
        elif lab in ARITY:
            syms.append(lab)
        else:
            raise ValueError(f"symbol {lab!r} outside the basis")
    pos = [0]

    def build():
        if pos[0] >= len(syms):
            raise ValueError("truncated prefix expression")
        s = syms[pos[0]]
        pos[0] += 1
        return (s,) + tuple(build() for _ in range(ARITY[s]))

    t = build()
    if pos[0] != len(syms):
        raise ValueError("trailing symbols in prefix expression")
    return t


def to_prefix(tree):
    out = [tree[0]]
    for c in tree[1:]:
        out += to_prefix(c)
    return out


def from_prefix(syms):
    return from_labels(list(syms))


def to_string(tree):
    """Infix string with x0 for the variable and a fresh a0, a1, ... for each
    constant leaf (fnprior's input convention)."""
    k = [0]

    def rec(t):
        s = t[0]
        if s == "a":
            k[0] += 1
            return f"a{k[0] - 1}"
        if s == "x":
            return "x0"
        if s == "inv":
            return f"(1/({rec(t[1])}))"
        if s in UNARY:
            return f"{s}({rec(t[1])})"
        a, b = rec(t[1]), rec(t[2])
        return f"(({a})**({b}))" if s == "pow" else f"(({a}){s}({b}))"

    return rec(tree)


# ------------------------------------------------------------------ events
def nodes(tree, pos: Pos = ROOT):
    """(position, subtree) of every node, in pre-order (a node before its
    children, a first child's subtree before the second child)."""
    out = [(pos, tree)]
    s = tree[0]
    if len(tree) > 1:
        out += nodes(tree[1], child_pos(pos, s, 0))
    if len(tree) > 2:
        out += nodes(tree[2], child_pos(pos, s, 1, tree[1][0]))
    return out


def events(tree):
    """(position, symbol) of every node, in pre-order."""
    return [(p, t[0]) for p, t in nodes(tree)]


def fnprior_events(tree, n=2):
    """The tree's events in the form of fnprior's split_events at order n,
    with symbol names instead of codes and 'None' padding: (left, right),
    left = [(n ancestors..., symbol)], right = [(n-1 ancestors..., left sibling, symbol)]
    (the right tuples truncated to their last n+1 elements, as fnprior's
    Kneser-Ney model uses them)."""
    left, right = [], []
    for p, s in events(tree):
        anc = tuple("None" if a is None else a for a in p.anc)
        if p.side == "L":
            left.append(anc[-n:] + (s,))
        else:
            right.append(anc[-(n - 1):] + (p.sib, s) if n > 1 else (p.sib, s))
    return left, right


def size(tree):
    return 1 + sum(size(c) for c in tree[1:])


def depth(tree):
    return 1 + max((depth(c) for c in tree[1:]), default=0)


def has_x(tree):
    return tree[0] == "x" or any(has_x(c) for c in tree[1:])
