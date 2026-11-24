"""Equations of NMODL (.mod) files, for the ModelDB corpus.

A light parser of the parts of NMODL that hold closed-form rate functions:
FUNCTION, PROCEDURE, BREAKPOINT, DERIVATIVE and INITIAL blocks. Every
assignment ``name = expr`` (and ``name' = expr``) in them is an equation.
Within a block, LOCAL variables are substituted into later assignments, and
calls of the file's own FUNCTIONs are inlined (a function's value is its last
assignment to its own name, after substituting its locals; for
``if``/``else`` definitions such as ``vtrap`` this is the general ``else``
branch). Unit annotations after numbers (``1 (mV)``) are dropped.

The variables of an equation are the membrane potential ``v``, time ``t``,
the STATE variables, the ion concentrations the mechanism READs, and the
arguments of the enclosing FUNCTION or PROCEDURE; every other identifier
(PARAMETERs, ASSIGNED variables such as ``minf``, ``celsius``) is a constant.
Equations are returned as sympy strings with identifiers prefixed by ``n_``
(so that no NMODL name clashes with a sympy name) for
``hsr.corpus_sources.law_keys``.
"""
from __future__ import annotations

import re

import sympy as sp

KEYWORDS_SKIP = ("LOCAL", "TABLE", "UNITSOFF", "UNITSON", "SOLVE", "printf", "VERBATIM", "ENDVERBATIM",
                 "FROM", "WATCH", "net_send", "net_event", "CONSERVE", "COMPARTMENT", "LONGITUDINAL_DIFFUSION",
                 "LAG", "PROTECT", "MUTEXLOCK", "MUTEXUNLOCK", "return", "RETURN")
BLOCKS = ("FUNCTION", "PROCEDURE", "BREAKPOINT", "DERIVATIVE", "INITIAL")
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# numbers first, so that the exponent of 1.e-6 is not read as an identifier
_TOKEN = re.compile(r"(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|(?P<id>[A-Za-z_][A-Za-z0-9_]*)")
_NUM_UNIT = re.compile(r"((?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\(\s*[1/]?\s*/?\s*[A-Za-z][A-Za-z0-9_]*"
                       r"(?:\s*[-/*]?\s*[A-Za-z0-9_]+)*\s*\)")
_BUILTIN = {"exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "tanh": sp.tanh, "sinh": sp.sinh, "cosh": sp.cosh, "atan": sp.atan, "fabs": sp.Abs,
            "pow": lambda a, b: a ** b, "log10": lambda a: sp.log(a) / sp.log(10)}


class NmodlError(Exception):
    pass


def strip_comments(text: str) -> str:
    """Remove COMMENT ... ENDCOMMENT, VERBATIM ... ENDVERBATIM and ``:``/``?`` line comments."""
    text = re.sub(r"(?s)\bCOMMENT\b.*?\bENDCOMMENT\b", "\n", text)
    text = re.sub(r"(?s)\bVERBATIM\b.*?\bENDVERBATIM\b", "\n", text)
    return "\n".join(re.split(r"[:?]", line, maxsplit=1)[0] for line in text.splitlines())


def _match_brace(text, i):
    """Index just after the brace that closes the one at text[i]."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise NmodlError("unbalanced braces")


def _strip_units(s):
    """Drop parenthesised units: ``v(mV)`` -> ``v`` in declarations."""
    return re.sub(r"\([^()]*\)", "", s)


def blocks(text):
    """[(kind, name, [arguments], body)] of the FUNCTION, PROCEDURE, BREAKPOINT,
    DERIVATIVE and INITIAL blocks, and {block kind: body} of the declaration
    blocks (NEURON, STATE)."""
    out, decl = [], {}
    pat = re.compile(r"\b(FUNCTION|PROCEDURE|BREAKPOINT|DERIVATIVE|INITIAL|NEURON|STATE|KINETIC|NET_RECEIVE|"
                     r"PARAMETER|ASSIGNED|UNITS|CONSTANT|INDEPENDENT|LINEAR|NONLINEAR|DISCRETE|PARTIAL|"
                     r"FUNCTION_TABLE|BEFORE|AFTER|CONSTRUCTOR|DESTRUCTOR)\b")
    pos = 0
    while True:
        m = pat.search(text, pos)
        if not m:
            break
        brace = text.find("{", m.end())
        if brace < 0:
            break
        head = text[m.end():brace]
        try:
            end = _match_brace(text, brace)
        except NmodlError:
            break                     # keep the blocks read so far
        body = text[brace + 1:end - 1]
        kind = m.group(1)
        if kind in ("FUNCTION", "PROCEDURE"):
            hm = re.match(r"\s*([A-Za-z_]\w*)\s*(\((.*)\))?", head, re.S)
            if hm:
                args = [a.strip() for a in _strip_units(hm.group(3) or "").split(",") if a.strip()]
                out.append((kind, hm.group(1), args, body))
        elif kind in BLOCKS:
            out.append((kind, head.strip().split()[0] if head.strip() else kind, [], body))
        else:
            decl[kind] = decl.get(kind, "") + "\n" + body
        pos = end
    return out, decl


def declared_variables(decl):
    """STATE names and ion concentrations READ by the mechanism."""
    names = {"v", "t"}
    st = _strip_units(decl.get("STATE", ""))
    st = re.sub(r"\bFROM\b.*?\bTO\b\s*\S+", " ", st)
    names |= {w for w in _ID.findall(st) if w not in ("FROM", "TO")}
    for m in re.finditer(r"\bUSEION\s+\w+\s+READ\s+([\w\s,]+?)(?=\bWRITE\b|\bVALENCE\b|\bUSEION\b|\n|$)",
                         decl.get("NEURON", "")):
        names |= {w.strip() for w in m.group(1).split(",") if w.strip()}
    return names


def statements(body):
    """[(lhs, rhs)] of the assignments of a block body, in order; ``if (...)`` and
    ``else`` prefixes are dropped, so both branches of a conditional appear."""
    out = []
    flat = body.replace("{", "\n").replace("}", "\n")
    for line in _join_continued(flat.splitlines()):
        s = line.strip()
        while True:
            m = re.match(r"(else\b\s*)?if\s*\(", s)
            if m:
                depth, j = 0, m.end() - 1
                for j in range(m.end() - 1, len(s)):
                    depth += s[j] == "("
                    depth -= s[j] == ")"
                    if depth == 0:
                        break
                s = s[j + 1:].strip()
                continue
            if s.startswith("else"):
                s = s[4:].strip()
                continue
            break
        if not s or s.split()[0].split("(")[0] in KEYWORDS_SKIP or "~" in s:
            continue
        for part in _split_assignments(s):
            m = re.match(r"([A-Za-z_]\w*)\s*('*)\s*=(?!=)\s*(.+)$", part)
            if m:
                out.append((m.group(1) + m.group(2), m.group(3).strip()))
    return out


def _join_continued(lines):
    """Join an expression continued over several lines: a line whose
    parentheses are not balanced, or that ends with an operator, continues
    on the next line."""
    out, cur = [], ""
    for line in lines:
        cur = (cur + " " + line.strip()) if cur else line.strip()
        if cur.count("(") > cur.count(")") or re.search(r"[-+*/^=,]\s*$", cur):
            continue
        out.append(cur)
        cur = ""
    if cur:
        out.append(cur)
    return out


_ASSIGN = re.compile(r"([A-Za-z_]\w*)\s*'*\s*=(?!=)")


def _split_assignments(s):
    """Split ``a = f(v)  b = g(v)`` (several assignments on one line) at each
    ``name =`` that follows a complete expression (balanced parentheses)."""
    cuts = [0]
    for m in _ASSIGN.finditer(s):
        if m.start() == 0:
            continue
        before = s[cuts[-1]:m.start()]
        if (s[m.start() - 1].isspace() and "=" in before and before.count("(") == before.count(")")
                and not re.search(r"[-+*/^(,=<>]\s*$", before)):
            cuts.append(m.start())
    return [s[a:b].strip() for a, b in zip(cuts, cuts[1:] + [len(s)])]


class Module:
    """Parsed NMODL file: functions for inlining and equations of every block."""

    def __init__(self, text, max_ops=400):
        self.blocks, decl = blocks(strip_comments(text))
        self.variables = declared_variables(decl)
        self.max_ops = max_ops
        self.funcs = {name: (args, statements(body)) for kind, name, args, body in self.blocks if kind == "FUNCTION"}
        self._ret = {}
        self._depth = 0

    # ------------------------------------------------------------ expressions
    def _sym(self, name):
        return sp.Symbol("n_" + name, real=True)

    def parse(self, expr, env):
        """sympy expression of an NMODL expression string; ``env`` maps names to
        expressions (locals, function arguments)."""
        expr = _NUM_UNIT.sub(r"\1", expr).replace("^", "**")
        if re.search(r"[<>!&|]|==", expr):
            raise NmodlError("logical expression")
        ns = {}
        names = {m.group("id") for m in _TOKEN.finditer(expr) if m.group("id")}
        for name in names:
            if name in _BUILTIN:
                ns[name] = _BUILTIN[name]
            elif name in self.funcs:
                ns[name] = self._caller(name)
            elif name in env:
                ns["n_" + name] = env[name]
            else:
                ns["n_" + name] = self._sym(name)
        call_names = set(_BUILTIN) | set(self.funcs)
        renamed = _TOKEN.sub(lambda m: m.group(0) if m.group("num") or m.group("id") in call_names
                             else "n_" + m.group("id"), expr)
        try:
            out = sp.sympify(renamed, locals=ns, rational=False)
        except Exception as e:
            raise NmodlError(f"parse: {type(e).__name__}") from e
        if not isinstance(out, sp.Expr):
            raise NmodlError("not an expression")
        return out

    def _caller(self, name):
        def call(*a):
            args, ret = self.function_value(name)
            if len(a) != len(args):
                raise NmodlError(f"{name}: {len(a)} arguments for {len(args)}")
            return ret.xreplace(dict(zip(args, a)))
        return call

    def function_value(self, name):
        """(argument symbols, value expression) of a FUNCTION."""
        if name in self._ret:
            if self._ret[name] is None:
                raise NmodlError(f"recursive function {name}")
            return self._ret[name]
        self._ret[name] = None
        args, stmts = self.funcs[name]
        arg_syms = [sp.Dummy("a_" + a) for a in args]
        env = dict(zip(args, arg_syms))
        ret = None
        for lhs, rhs in stmts:
            val = self.parse(rhs, env)
            env[lhs] = val
            if lhs == name:
                ret = val
        if ret is None:
            raise NmodlError(f"function {name} assigns no value")
        self._ret[name] = (arg_syms, ret)
        return self._ret[name]

    # -------------------------------------------------------------- equations
    def equations(self):
        """([(equation string, [variable names])], [errors]) of every block:
        each assignment's right-hand side with locals substituted and calls
        inlined; the variables of each equation are the declared variables
        and the enclosing block's arguments that occur in it."""
        eqs, errors = [], []
        for kind, name, args, body in self.blocks:
            env = {}
            var_names = self.variables | set(args)
            for lhs, rhs in statements(body):
                try:
                    val = self.parse(rhs, env)
                except (NmodlError, RecursionError) as e:
                    errors.append(f"{kind} {name}: {lhs}: {e}")
                    env.pop(lhs.rstrip("'"), None)
                    continue
                if lhs.rstrip("'") not in var_names:
                    env[lhs.rstrip("'")] = val
                if sp.count_ops(val) > self.max_ops:
                    errors.append(f"{kind} {name}: {lhs}: too large")
                    continue
                present = sorted(n for n in var_names if val.has(self._sym(n)))
                if present:
                    eqs.append((str(val), ["n_" + n for n in present]))
        return eqs, errors
