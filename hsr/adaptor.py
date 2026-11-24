"""Pitman-Yor adaptor over whole subtrees on top of a symbol-level
hierarchical Pitman-Yor prior: an adaptor grammar (Johnson, Griffiths &
Goldwater 2007) whose base is the tree HPY of hsr.pyp.

Generative model. Each adapted position class u (a restaurant key of the
symbol model, so all positions in it share the same base) has

    H_u ~ PY(a_g, b_g, B_u),
    B_u(T) = G_u(root(T)) * prod_i P_{u_i}(T_i),

where G_u is the symbol-level HPY distribution at u, u_i the position of
child i and P_{u_i} = H_{u_i} if that position is adapted, else B_{u_i}.
A tree is a draw from P at the root. In the Chinese-restaurant
representation a customer of H_u either joins a table of its subtree
(reusing it whole) or opens a table, which draws a symbol event from G_u
(a customer of the symbol HPY) and one customer for each child position.
mode="root": only the root is adapted, i.e. a PY adaptor over whole forms
(Goldwater, Griffiths & Johnson 2006: the symbol model then sees each form
type roughly once per table, interpolating between type and token counts).
mode="all": every position is adapted (subtrees are reused anywhere).

Inference (exact MCMC). For each training tree in turn, its analysis (the
tables its customers sit at, and the tables and symbol events these
create) is removed and a new one proposed top-down: at an adapted node,
join a table of the same subtree with weight n_k - a, or open one with
weight (b + a K) * B^_u(T), B^_u the product of the current predictives
(exact except when one tree visits the same restaurant twice); symbol
events are seated by their Gibbs conditional. The proposal is accepted with
the Metropolis-Hastings ratio W_new / W_old, where W = pi / q is
accumulated along the proposal (and, for the old analysis, along its
removal in reverse order): Z / (b + n) per adapted customer, 1 / B^_u per
opened table, and the predictive probability of every symbol event (the
symbol-level seating is resampled from its conditional, so its pi/q is that
probability). On rejection the old analysis is restored, with its symbol
events re-seated from their conditional (a Gibbs step). Between tree
updates: Gibbs sweeps of the symbol-level seating, Metropolis updates of
(d, theta) and (a, b) (priors Beta(1, 1) on discounts, Gamma(1, 1) on
strengths), and the mixing weights of a graphical symbol model.
"""
from __future__ import annotations

import numpy as np

from . import treecode as tc
from .pyp import ContextSpec, GraphPY


class _Table:
    __slots__ = ("dish", "size", "kids")

    def __init__(self, dish):
        self.dish = dish
        self.size = 0
        self.kids = None


class _Fresh:
    """A node at a non-adapted position (always generated from the base)."""
    __slots__ = ("kids",)

    def __init__(self):
        self.kids = None


class _ARest:
    __slots__ = ("n", "K", "dishes")

    def __init__(self):
        self.n = 0
        self.K = 0
        self.dishes = {}


class TreeAdaptor:
    def __init__(self, spec: ContextSpec, mode="all", seed=0, a0=0.5, b0=1.0):
        if mode not in ("root", "all"):
            raise ValueError(mode)
        self.spec = spec
        self.mode = mode
        self.V = tc.V
        self.vi = {s: i for i, s in enumerate(self.V)}
        self.rng = np.random.default_rng(seed)
        self.hpy = GraphPY(self.V, spec, seed=seed + 7919)
        self.A = {}
        self.a, self.b = {}, {}
        self.a0, self.b0 = a0, b0
        self.n_acc = 0
        self.n_prop = 0

    # ------------------------------------------------------------ structure
    def adapted(self, pos):
        return self.mode == "all" or pos.depth == 0

    @staticmethod
    def agroup(key):
        """(a, b) group: the root restaurant (no parent), first and second children."""
        return ("A", "root") if dict(key[1]).get("a1", 0) is None else ("A", key[0])

    def arest(self, key):
        r = self.A.get(key)
        if r is None:
            r = self.A[key] = _ARest()
            g = self.agroup(key)
            if g not in self.a:
                self.a[g], self.b[g] = self.a0, self.b0
        return r

    def ahyp(self, key):
        g = self.agroup(key)
        return self.a.get(g, self.a0), self.b.get(g, self.b0)

    @staticmethod
    def kids_pos(pos, T):
        if len(T) == 1:
            return []
        out = [tc.child_pos(pos, T[0], 0)]
        if len(T) == 3:
            out.append(tc.child_pos(pos, T[0], 1, T[1][0]))
        return out

    # ----------------------------------------------------------- predictive
    def p_tree(self, pos, src, T):
        """Predictive probability of subtree T at pos (current state; product
        of the children's current predictives)."""
        key = self.spec.key(pos, src)
        B = self.hpy.prob(key, self.vi[T[0]])
        for cp, Tc in zip(self.kids_pos(pos, T), T[1:]):
            B *= self.p_tree(cp, src, Tc)
        if not self.adapted(pos):
            return B
        r = self.A.get(key)
        if r is None or r.n == 0:
            return B
        a, b = self.ahyp(key)
        tabs = r.dishes.get(T, ())
        return (sum(t.size for t in tabs) - a * len(tabs) + (b + a * r.K) * B) / (b + r.n)

    def _base(self, pos, src, T):
        key = self.spec.key(pos, src)
        B = self.hpy.prob(key, self.vi[T[0]])
        for cp, Tc in zip(self.kids_pos(pos, T), T[1:]):
            B *= self.p_tree(cp, src, Tc)
        return B

    # ------------------------------------------------------ add and remove
    def _propose(self, pos, src, T):
        """Add subtree T at pos by the top-down proposal; returns (record, log W)."""
        key = self.spec.key(pos, src)
        w = self.vi[T[0]]
        lw = 0.0
        if self.adapted(pos):
            r = self.arest(key)
            a, b = self.ahyp(key)
            tabs = r.dishes.setdefault(T, [])
            Bh = self._base(pos, src, T)
            wts = np.array([t.size - a for t in tabs] + [(b + a * r.K) * Bh])
            Z = wts.sum()
            lw += np.log(Z) - np.log(b + r.n)
            k = int(self.rng.choice(len(wts), p=wts / Z))
            r.n += 1
            if k < len(tabs):
                tabs[k].size += 1
                return tabs[k], lw
            lw -= np.log(Bh)
            rec = _Table(T)
            rec.size = 1
            tabs.append(rec)
            r.K += 1
        else:
            rec = _Fresh()
        lw += np.log(self.hpy.add(key, w))
        rec.kids = []
        for cp, Tc in zip(self.kids_pos(pos, T), T[1:]):
            kr, klw = self._propose(cp, src, Tc)
            rec.kids.append(kr)
            lw += klw
        return rec, lw

    def _remove(self, pos, src, T, rec):
        """Remove the analysis rec of subtree T at pos, in the reverse order
        of _propose; returns the log W the proposal would have accumulated."""
        key = self.spec.key(pos, src)
        w = self.vi[T[0]]
        lw = 0.0
        if self.adapted(pos):
            r = self.A[key]
            a, b = self.ahyp(key)
            if rec.size > 1:
                rec.size -= 1
                r.n -= 1
                tabs = r.dishes[T]
                Bh = self._base(pos, src, T)
                Z = sum(t.size - a for t in tabs) + (b + a * r.K) * Bh
                return np.log(Z) - np.log(b + r.n)
        kp = self.kids_pos(pos, T)
        for i in reversed(range(len(kp))):
            lw += self._remove(kp[i], src, T[1 + i], rec.kids[i])
        self.hpy.remove(key, w)
        lw += np.log(self.hpy.prob(key, w))
        if self.adapted(pos):
            tabs = r.dishes[T]
            tabs.remove(rec)          # by identity (_Table has no __eq__)
            rec.size = 0
            r.K -= 1
            r.n -= 1
            Bh = self._base(pos, src, T)
            Z = sum(t.size - a for t in tabs) + (b + a * r.K) * Bh
            lw += np.log(Z) - np.log(b + r.n) - np.log(Bh)
        return lw

    def _restore(self, pos, src, T, rec):
        """Re-add a removed analysis with the same choices (symbol events
        re-seated from their conditional)."""
        key = self.spec.key(pos, src)
        if self.adapted(pos):
            r = self.A[key]
            r.n += 1
            if rec.size > 0:          # its table survived the removal: rejoin it
                rec.size += 1
                return
            rec.size = 1
            r.dishes.setdefault(T, []).append(rec)
            r.K += 1
        self.hpy.add(key, self.vi[T[0]])
        for i, cp in enumerate(self.kids_pos(pos, T)):
            self._restore(cp, src, T[1 + i], rec.kids[i])

    # --------------------------------------------------------------- moves
    def update_tree(self, i):
        T, src = self.trees[i], self.sources[i]
        old = self.recs[i]
        lw_old = self._remove(tc.ROOT, src, T, old)
        new, lw_new = self._propose(tc.ROOT, src, T)
        self.n_prop += 1
        if np.log(self.rng.random()) < lw_new - lw_old:
            self.recs[i] = new
            self.n_acc += 1
        else:
            self._remove(tc.ROOT, src, T, new)
            self._restore(tc.ROOT, src, T, old)

    def update_ahyper(self, steps=20):
        st = {}
        for key, r in self.A.items():
            if r.n == 0:
                continue
            s = st.setdefault(self.agroup(key), ([], [], []))
            s[0].append(r.K)
            s[1].append(r.n)
            for tabs in r.dishes.values():
                s[2].extend(t.size for t in tabs)
        for g, s in st.items():
            t, c, sizes = (np.array(x, float) for x in s)
            a0, b0 = self.a[g], self.b[g]
            lp0 = GraphPY.seating_loglik(a0, b0, t, c, sizes) - b0 + np.log(b0)
            for _ in range(steps):
                a1 = a0 + self.rng.normal(0, 0.05)
                b1 = b0 * np.exp(self.rng.normal(0, 0.3))
                lp1 = GraphPY.seating_loglik(a1, b1, t, c, sizes) - b1 + np.log(b1)
                if np.log(self.rng.random()) < lp1 - lp0:
                    a0, b0, lp0 = a1, b1, lp1
            self.a[g], self.b[g] = a0, b0

    def fit(self, trees, sources=None, n_sweeps=20, n_keep=10):
        self.trees = list(trees)
        self.sources = list(sources) if sources is not None else [None] * len(self.trees)
        self.recs = [None] * len(self.trees)
        for i in range(len(self.trees)):
            self.recs[i], _ = self._propose(tc.ROOT, self.sources[i], self.trees[i])
        self.samples = []
        self.acc_rate = []
        for sweep in range(n_sweeps):
            a0, p0 = self.n_acc, self.n_prop
            for i in self.rng.permutation(len(self.trees)):
                self.update_tree(int(i))
            self.hpy.gibbs_sweep()
            self.hpy.update_hyper()
            if self.hpy.lam:
                self.hpy.update_lambda()
            self.update_ahyper()
            self.acc_rate.append((self.n_acc - a0) / max(1, self.n_prop - p0))
            if sweep >= n_sweeps - n_keep:
                self.samples.append(AdaptorSnapshot(self))
        return self

    def check(self):
        """Invariants: restaurant totals, every table reachable from exactly
        as many customers as its size, symbol events = opened tables + fresh nodes."""
        self.hpy.check()
        for key, r in self.A.items():
            assert r.n == sum(t.size for ts in r.dishes.values() for t in ts), key
            assert r.K == sum(len(ts) for ts in r.dishes.values()), key
        cust = {}
        n_sym = [0]

        def walk(rec, pos, src, T):
            if self.adapted(pos):
                cust[id(rec)] = cust.get(id(rec), 0) + 1
                if cust[id(rec)] > 1:
                    return
            n_sym[0] += 1
            for i, cp in enumerate(self.kids_pos(pos, T)):
                walk(rec.kids[i], cp, src, T[1 + i])

        for T, s, rec in zip(self.trees, self.sources, self.recs):
            walk(rec, tc.ROOT, s, T)
        for key, r in self.A.items():
            for ts in r.dishes.values():
                for t in ts:
                    assert cust.get(id(t), 0) == t.size, (key, t.dish, t.size, cust.get(id(t)))
        assert n_sym[0] == sum(sum(r.obs) for r in self.hpy.R.values())

    # ------------------------------------------------------------ predictive
    def logp(self, tree, src=None):
        v = np.array([np.log(s.p_tree(tc.ROOT, src, tree)) for s in self.samples])
        return float(np.logaddexp.reduce(v) - np.log(len(v)))

    def hyper_summary(self):
        return {"d": {str(g): float(v) for g, v in self.hpy.d.items()},
                "theta": {str(g): float(v) for g, v in self.hpy.theta.items()},
                "lambda": {str(g): v.tolist() for g, v in self.hpy.lam.items()},
                "a": {str(g): float(v) for g, v in self.a.items()},
                "b": {str(g): float(v) for g, v in self.b.items()},
                "acceptance": [float(x) for x in self.acc_rate]}


class AdaptorSnapshot:
    """Frozen predictive of a TreeAdaptor."""

    def __init__(self, m: TreeAdaptor):
        self.spec = m.spec
        self.mode = m.mode
        self.vi = m.vi
        self.sym = m.hpy.snapshot()
        self.A = {}
        for key, r in m.A.items():
            if r.n == 0:
                continue
            dishes = {T: (sum(t.size for t in ts), len(ts)) for T, ts in r.dishes.items() if ts}
            self.A[key] = (r.n, r.K, m.ahyp(key), dishes)
        self._memo = {}

    def adapted(self, pos):
        return self.mode == "all" or pos.depth == 0

    def p_tree(self, pos, src, T):
        key = self.spec.key(pos, src)
        ad = self.adapted(pos)
        mk = (key, T, ad)
        if mk in self._memo:
            return self._memo[mk]
        B = self.sym.vec(key)[self.vi[T[0]]]
        for cp, Tc in zip(TreeAdaptor.kids_pos(pos, T), T[1:]):
            B *= self.p_tree(cp, src, Tc)
        p = B
        if ad and key in self.A:
            n, K, (a, b), dishes = self.A[key]
            nT, KT = dishes.get(T, (0, 0))
            p = (nT - a * KT + (b + a * K) * B) / (b + n)
        self._memo[mk] = p
        return p

    def node_dist(self, pos, src=None):
        """(cached [(subtree, prob)], probability of a fresh draw, symbol
        distribution) of the node at pos (for termination and sampling)."""
        key = self.spec.key(pos, src)
        vec = self.sym.vec(key)
        if not self.adapted(pos) or key not in self.A:
            return [], 1.0, vec
        n, K, (a, b), dishes = self.A[key]
        cached = [(T, (nT - a * KT) / (b + n)) for T, (nT, KT) in dishes.items()]
        return cached, (b + a * K) / (b + n), vec
