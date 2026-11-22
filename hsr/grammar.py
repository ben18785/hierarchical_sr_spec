"""Probabilistic context-free grammar over expression skeletons (spec section 3).

    E -> E + F | E - F | F
    F -> F * T | F / T | T
    T -> ( E ) | R ( E ) | V | C
    R -> exp | log | sin | sqrt
    V -> x1 | ... | xd
    C -> const

Derivations are produced as strings with the placeholder ``C`` for every
constant. "Size" is the number of rule applications whose left-hand side is
E, F or T (the R, V and C choices are not counted), as in the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FUNCTIONS = ("exp", "log", "sin", "sqrt")

# Ordered options for the structural nonterminals. Order matters: it is the
# order of the Dirichlet parameters and of the count vectors.
E_RULES = ("E+F", "E-F", "F")
F_RULES = ("F*T", "F/T", "T")
T_RULES = ("(E)", "R(E)", "V", "C")


@dataclass
class Grammar:
    """The PCFG together with its Dirichlet prior on rule probabilities."""

    n_vars: int = 1
    alpha: dict = field(default_factory=dict)
    max_size: int = 16
    max_consts: int = 4

    def __post_init__(self):
        default = {
            "E": np.array([1.0, 1.0, 4.0]),
            "F": np.array([1.0, 1.0, 4.0]),
            "T": np.array([0.5, 1.0, 3.0, 2.0]),
            "R": np.array([0.5, 0.5, 0.5, 0.5]),
            "V": np.ones(self.n_vars),
        }
        for key, value in self.alpha.items():
            default[key] = np.asarray(value, dtype=float)
        self.alpha = default
        self.var_names = tuple(f"x{i + 1}" for i in range(self.n_vars))
        self.options = {
            "E": E_RULES,
            "F": F_RULES,
            "T": T_RULES,
            "R": FUNCTIONS,
            "V": self.var_names,
        }
        for nt, opts in self.options.items():
            if len(self.alpha[nt]) != len(opts):
                raise ValueError(f"alpha[{nt}] needs {len(opts)} entries")
        # Flat indexing of every rule, used for count vectors.
        self.nonterminals = ("E", "F", "T", "R", "V")
        self.rule_index = {}
        self.slices = {}
        k = 0
        for nt in self.nonterminals:
            start = k
            for opt in self.options[nt]:
                self.rule_index[(nt, opt)] = k
                k += 1
            self.slices[nt] = slice(start, k)
        self.n_rules = k

    # ------------------------------------------------------------------ theta
    def prior_mean(self) -> dict:
        return {nt: a / a.sum() for nt, a in self.alpha.items()}

    def sample_theta(self, rng, counts: np.ndarray | None = None) -> dict:
        """Draw theta_A ~ Dirichlet(alpha_A + counts_A) for every nonterminal."""
        theta = {}
        for nt in self.nonterminals:
            a = self.alpha[nt].copy()
            if counts is not None:
                a = a + counts[self.slices[nt]]
            theta[nt] = rng.dirichlet(a)
        return theta

    def log_theta_vector(self, theta: dict) -> np.ndarray:
        out = np.empty(self.n_rules)
        for nt in self.nonterminals:
            out[self.slices[nt]] = np.log(theta[nt])
        return out

    def log_prob(self, counts: np.ndarray, theta: dict) -> np.ndarray:
        """log P_theta(e) = sum_r counts[e, r] log theta_r. counts may be 1-D or 2-D."""
        return np.asarray(counts) @ self.log_theta_vector(theta)

    def counts_vector(self, counts: dict) -> np.ndarray:
        """Convert a {(nt, option): n} dict into a flat count vector."""
        v = np.zeros(self.n_rules)
        for key, n in counts.items():
            v[self.rule_index[key]] += n
        return v

    # --------------------------------------------------------------- sampling
    def sample(self, rng, theta: dict | None = None, max_size: int | None = None):
        """Sample one derivation top-down from E.

        Returns ``(string, size, n_consts)`` or ``None`` when the derivation
        exceeds ``max_size`` (rejection gives the truncated PCFG).
        """
        theta = self.prior_mean() if theta is None else theta
        max_size = self.max_size if max_size is None else max_size
        state = {"size": 0, "consts": 0}

        class _TooBig(Exception):
            pass

        def choose(nt):
            return self.options[nt][rng.choice(len(self.options[nt]), p=theta[nt])]

        def bump():
            state["size"] += 1
            if state["size"] > max_size:
                raise _TooBig

        def gen(nt):
            bump()
            opt = choose(nt)
            if nt == "E":
                if opt == "F":
                    return gen("F")
                left = gen("E")
                return f"{left}{opt[1]}{gen('F')}"
            if nt == "F":
                if opt == "T":
                    return gen("T")
                left = gen("F")
                return f"{left}{opt[1]}{gen('T')}"
            # T
            if opt == "(E)":
                return f"({gen('E')})"
            if opt == "R(E)":
                return f"{choose('R')}({gen('E')})"
            if opt == "V":
                return choose("V")
            state["consts"] += 1
            return "C"

        try:
            s = gen("E")
        except _TooBig:
            return None
        except RecursionError:
            return None
        return s, state["size"], state["consts"]

    # ------------------------------------------------------------ enumeration
    def enumerate(self, max_size: int, max_consts: int | None = None):
        """Enumerate every E-derivation of size <= max_size (bottom-up).

        Variable-free composite subexpressions (``C*C``, ``exp(C)``, ...) are
        pruned because canonicalisation collapses them to a single constant
        anyway. Returns a list of strings.
        """
        max_consts = self.max_consts if max_consts is None else max_consts
        # table[nt][size] = list of (string, n_consts, has_var)
        table = {nt: [[] for _ in range(max_size + 1)] for nt in "EFT"}

        def ok(nc, hv, composite):
            if nc > max_consts:
                return False
            return hv or not composite

        for s in range(1, max_size + 1):
            # T
            if s == 1:
                for v in self.var_names:
                    table["T"][1].append((v, 0, True))
                table["T"][1].append(("C", 1, False))
            else:
                for (es, nc, hv) in table["E"][s - 1]:
                    if not hv:
                        continue
                    table["T"][s].append((f"({es})", nc, hv))
                    for fn in FUNCTIONS:
                        table["T"][s].append((f"{fn}({es})", nc, hv))
            # F
            for (ts, nc, hv) in table["T"][s - 1] if s >= 2 else []:
                table["F"][s].append((ts, nc, hv))
            for a in range(1, s - 1):
                b = s - 1 - a
                for (fs, nc1, hv1) in table["F"][a]:
                    for (ts, nc2, hv2) in table["T"][b]:
                        nc, hv = nc1 + nc2, hv1 or hv2
                        if ok(nc, hv, True):
                            table["F"][s].append((f"{fs}*{ts}", nc, hv))
                            table["F"][s].append((f"{fs}/{ts}", nc, hv))
            # E
            for (fs, nc, hv) in table["F"][s - 1] if s >= 2 else []:
                table["E"][s].append((fs, nc, hv))
            for a in range(1, s - 1):
                b = s - 1 - a
                for (es, nc1, hv1) in table["E"][a]:
                    for (fs, nc2, hv2) in table["F"][b]:
                        nc, hv = nc1 + nc2, hv1 or hv2
                        if ok(nc, hv, True):
                            table["E"][s].append((f"{es}+{fs}", nc, hv))
                            table["E"][s].append((f"{es}-{fs}", nc, hv))
        out = []
        for s in range(1, max_size + 1):
            out.extend(es for (es, nc, hv) in table["E"][s] if hv)
        return out
