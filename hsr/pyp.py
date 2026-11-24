"""Hierarchical and graphical Pitman-Yor models over expression-tree events.

A generalisation of hsr.hpy (Teh 2006) in which every restaurant (a
next-symbol distribution G_u for a context u) has one or more parent
restaurants:

    G_u ~ PY(d_g(u), theta_g(u), sum_j lambda_{m(u), j} G_{pi_j(u)}),

with the top restaurant's base uniform over the vocabulary. One parent
gives the usual back-off chain (Teh 2006); several give the graphical
Pitman-Yor process of Wood & Teh (2009), whose mixing weights lambda are
learned. Contexts are sets of named components (ancestors a1 = parent, a2
= grandparent, a3; l = left sibling; d = depth; s = corpus source), and a
``ContextSpec`` says which components a position has and how they are
dropped when backing off:

  * chain components are dropped one at a time in a fixed order (the most
    distant ancestor first, as in fnprior);
  * each graph component present gives an extra parent with that
    component dropped (e.g. graph=("s",) is the doubly hierarchical model of
    Wood & Teh 2009: G_{s,u} backs off both to the shared G_u and to the
    source's shorter context G_{s,pi(u)}).

Inference. Every training event is a customer of its full-context
restaurant; a new table sends a customer to one parent (chosen with
probability proportional to lambda_j p_j(w)). Gibbs sampling of the seating
(remove each customer, then re-add it from its conditional, as in Teh 2006);
random-walk Metropolis on each (d, theta) group with priors d ~ Beta(1, 1),
theta ~ Gamma(1, 1), using the Pitman-Yor seating likelihood of the group's
restaurants; and a Gibbs update of each lambda group from its Dirichlet(1,
..., 1) prior and the number of tables routed to each parent (the routings
are the only terms of the joint that involve lambda).

Predictions use snapshots of the seating taken after burn-in. Imports only
numpy and scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from . import treecode as tc


# ------------------------------------------------------------------ contexts
class ContextSpec:
    """Maps a tree position (hsr.treecode.Pos) and a source to a restaurant
    key, and gives every key's parents, hyperparameter group and mixing group.

    n         ancestors in a first child's context (a1..an); a second child
              has a1..a(n-1) and its left sibling l (fnprior's coding)
    drop_L, drop_R
              order in which components are dropped (first = dropped first);
              components not listed are never in the context
    graph     components that are dropped as an extra, parallel back-off path
    depth_cap if set, the node depth (capped) is a component "d"
    source    if True, the corpus source is a component "s"
    """

    def __init__(self, n=2, drop_L=None, drop_R=None, graph=(), depth_cap=None, source=False):
        self.n = n
        anc = [f"a{k}" for k in range(n, 0, -1)]               # most distant first
        ancR = [f"a{k}" for k in range(n - 1, 0, -1)]
        pre = (["s"] if source else []) + (["d"] if depth_cap is not None else [])
        self.drop = {"L": list(drop_L) if drop_L is not None else pre + anc,
                     "R": list(drop_R) if drop_R is not None else pre + ancR + ["l"]}
        self.graph = tuple(graph)
        self.depth_cap = depth_cap
        self.source = source
        self._parents = {}

    def describe(self):
        return {"n": self.n, "drop": self.drop, "graph": list(self.graph), "depth_cap": self.depth_cap,
                "source": self.source}

    def key(self, pos: tc.Pos, src=None):
        comp = {}
        names = self.drop[pos.side]
        for k in range(1, self.n + 1):
            nm = f"a{k}"
            if nm in names:
                comp[nm] = pos.anc[-k]
        if "l" in names:
            comp["l"] = pos.sib
        if "d" in names:
            comp["d"] = min(pos.depth, self.depth_cap)
        if "s" in names:
            comp["s"] = src
        return (pos.side, tuple(sorted(comp.items())))

    def parents(self, key):
        """Parent keys: one per present graph component (that component
        dropped), plus the chain parent (the first present non-graph
        component in drop order dropped). Empty for the top restaurant."""
        if key in self._parents:
            return self._parents[key]
        side, comp = key
        names = [c for c, _ in comp]
        out = []
        for g in self.graph:
            if g in names:
                out.append((side, tuple(x for x in comp if x[0] != g)))
        for c in self.drop[side]:
            if c in names and c not in self.graph:
                out.append((side, tuple(x for x in comp if x[0] != c)))
                break
        if not out and names:            # only graph components left
            pass
        out = tuple(dict.fromkeys(out))
        self._parents[key] = out
        return out

    @staticmethod
    def group(key):
        """(d, theta) group: side and the set of component names (= depth and
        child position, and the source level)."""
        return (key[0], tuple(c for c, _ in key[1]))

    def mixgroup(self, key):
        return self.group(key)


# ---------------------------------------------------------------- restaurants
class _Rest:
    __slots__ = ("c", "t", "cw", "tw", "tabs", "obs", "parents")

    def __init__(self, nv, parents):
        self.c = 0
        self.t = 0
        self.cw = [0] * nv
        self.tw = [0] * nv
        self.tabs = {}          # w -> [[sizes], [parent index per table]]
        self.obs = [0] * nv     # customers that are observations (not sent by a child)
        self.parents = parents


class GraphPY:
    """Gibbs sampler for a (graphical) hierarchical Pitman-Yor model with a
    uniform base over the vocabulary ``V``; restaurants and their parents
    come from ``spec`` (a ContextSpec or any object with parents, group and
    mixgroup methods)."""

    def __init__(self, V, spec, seed=0, d0=0.5, theta0=1.0):
        self.V = list(V)
        self.nv = len(self.V)
        self.idx = {w: i for i, w in enumerate(self.V)}
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.R = {}
        self.d = {}
        self.theta = {}
        self.lam = {}
        self.d0, self.theta0 = d0, theta0
        self.base = 1.0 / self.nv

    # ---------------------------------------------------------- structure
    def rest(self, key):
        r = self.R.get(key)
        if r is None:
            ps = self.spec.parents(key)
            r = _Rest(self.nv, ps)
            self.R[key] = r
            g = self.spec.group(key)
            if g not in self.d:
                self.d[g], self.theta[g] = self.d0, self.theta0
            if len(ps) > 1:
                m = self.spec.mixgroup(key)
                if m not in self.lam:
                    self.lam[m] = np.full(len(ps), 1.0 / len(ps))
        return r

    def hyper(self, key):
        g = self.spec.group(key)
        return self.d.get(g, self.d0), self.theta.get(g, self.theta0)

    # ---------------------------------------------------------- predictive
    def prob(self, key, w):
        """p(w | key) under the current seating (w an index)."""
        r = self.R.get(key)
        pb = self._base(key, r, w)
        if r is None or r.c == 0:
            return pb
        d, th = self.hyper(key)
        return (r.cw[w] - d * r.tw[w] + (th + d * r.t) * pb) / (th + r.c)

    def _base(self, key, r, w):
        ps = r.parents if r is not None else self.spec.parents(key)
        if not ps:
            return self.base
        if len(ps) == 1:
            return self.prob(ps[0], w)
        lam = self.lam.get(self.spec.mixgroup(key))
        if lam is None:
            lam = np.full(len(ps), 1.0 / len(ps))
        return float(sum(l * self.prob(p, w) for l, p in zip(lam, ps)))

    # ------------------------------------------------------------- seating
    def add(self, key, w, obs=True):
        """Seat a customer eating w (index) in restaurant key; returns its
        predictive probability before it was seated."""
        r = self.rest(key)
        d, th = self.hyper(key)
        ps = r.parents
        if not ps:
            pp, pb = None, self.base
        elif len(ps) == 1:
            pp = None
            pb = self.prob(ps[0], w)
        else:
            lam = self.lam[self.spec.mixgroup(key)]
            pp = np.array([self.prob(p, w) for p in ps]) * lam
            pb = float(pp.sum())
        tab = r.tabs.get(w)
        if tab is None:
            tab = r.tabs[w] = [[], []]
        sizes = tab[0]
        old = r.cw[w] - d * r.tw[w]
        new = (th + d * r.t) * pb
        p = (old + new) / (th + r.c)
        u = self.rng.random() * (old + new)
        if u < old:
            acc = 0.0
            k = len(sizes) - 1
            for i, s in enumerate(sizes):
                acc += s - d
                if u < acc:
                    k = i
                    break
            sizes[k] += 1
        else:
            j = 0 if pp is None else int(self.rng.choice(len(pp), p=pp / pp.sum()))
            sizes.append(1)
            tab[1].append(j)
            r.t += 1
            r.tw[w] += 1
            if ps:
                self.add(ps[j], w, obs=False)
        r.c += 1
        r.cw[w] += 1
        if obs:
            r.obs[w] += 1
        return p

    def remove(self, key, w, obs=True):
        """Remove a customer eating w from key (a table chosen in proportion
        to its size, i.e. a uniformly random customer of that dish)."""
        r = self.R[key]
        sizes, par = r.tabs[w]
        u = self.rng.random() * r.cw[w]
        acc = 0
        k = len(sizes) - 1
        for i, s in enumerate(sizes):
            acc += s
            if u < acc:
                k = i
                break
        sizes[k] -= 1
        r.c -= 1
        r.cw[w] -= 1
        if obs:
            r.obs[w] -= 1
        if sizes[k] == 0:
            sizes.pop(k)
            j = par.pop(k)
            r.t -= 1
            r.tw[w] -= 1
            if r.parents:
                self.remove(r.parents[j], w, obs=False)

    def observations(self):
        """[(key, w)] of every observed customer (with repeats)."""
        out = []
        for key, r in self.R.items():
            for w, n in enumerate(r.obs):
                out += [(key, w)] * n
        return out

    def gibbs_sweep(self, obs=None):
        obs = self.observations() if obs is None else obs
        for i in self.rng.permutation(len(obs)):
            key, w = obs[i]
            self.remove(key, w)
            self.add(key, w)

    # ------------------------------------------------------- hyperparameters
    def _group_stats(self):
        """Per (d, theta) group: arrays of restaurant tables t, customers c
        and every table size."""
        st = {}
        for key, r in self.R.items():
            if r.c == 0:
                continue
            g = self.spec.group(key)
            s = st.setdefault(g, ([], [], []))
            s[0].append(r.t)
            s[1].append(r.c)
            for sizes, _ in r.tabs.values():
                s[2].extend(sizes)
        return {g: tuple(np.array(x, float) for x in s) for g, s in st.items()}

    @staticmethod
    def seating_loglik(d, th, t, c, sizes):
        """log P(seating | d, theta), product over restaurants of the Pitman-Yor EPPF."""
        if not (0 < d < 1 and th > 0):
            return -np.inf
        # prod_{i=1}^{t-1} (theta + i d) = d^(t-1) Gamma(theta/d + t) / Gamma(theta/d + 1)
        a = np.sum((t - 1) * np.log(d) + gammaln(th / d + t) - gammaln(th / d + 1))
        b = np.sum(gammaln(th + c) - gammaln(th + 1))
        s = np.sum(gammaln(sizes - d)) - len(sizes) * gammaln(1 - d)
        return float(a - b + s)

    def update_hyper(self, steps=20):
        for g, (t, c, sizes) in self._group_stats().items():
            d0, t0 = self.d[g], self.theta[g]
            lp0 = self.seating_loglik(d0, t0, t, c, sizes) - t0 + np.log(t0)
            for _ in range(steps):
                d1 = d0 + self.rng.normal(0, 0.05)
                t1 = t0 * np.exp(self.rng.normal(0, 0.3))
                # Beta(1,1) on d, Gamma(1,1) on theta; log-scale proposal for theta (Jacobian log theta)
                lp1 = self.seating_loglik(d1, t1, t, c, sizes) - t1 + np.log(t1)
                if np.log(self.rng.random()) < lp1 - lp0:
                    d0, t0, lp0 = d1, t1, lp1
            self.d[g], self.theta[g] = d0, t0

    def update_lambda(self):
        cnt = {m: np.zeros(len(l)) for m, l in self.lam.items()}
        for key, r in self.R.items():
            if len(r.parents) > 1:
                v = cnt[self.spec.mixgroup(key)]
                for _, par in r.tabs.values():
                    for j in par:
                        v[j] += 1
        for m, v in cnt.items():
            self.lam[m] = self.rng.dirichlet(1.0 + v)

    # ---------------------------------------------------------------- fitting
    def fit(self, data, n_sweeps=20, n_keep=10, callback=None):
        """data: [(key, w)] observations (w a symbol or an index). Keeps a
        snapshot after each of the last n_keep sweeps."""
        data = [(k, self.idx[w] if not isinstance(w, (int, np.integer)) else int(w)) for k, w in data]
        for k, w in data:
            self.add(k, w)
        self.samples = []
        for sweep in range(n_sweeps):
            self.gibbs_sweep(data)
            self.update_hyper()
            if self.lam:
                self.update_lambda()
            if sweep >= n_sweeps - n_keep:
                self.samples.append(self.snapshot())
            if callback:
                callback(self, sweep)
        return self

    def snapshot(self):
        return Snapshot(self)

    def check(self):
        """Seating invariants; raises AssertionError if broken."""
        sent = {}
        for key, r in self.R.items():
            assert r.c == sum(sum(s) for s, _ in r.tabs.values()), key
            assert r.t == sum(len(s) for s, _ in r.tabs.values()), key
            for w, (s, par) in r.tabs.items():
                assert r.cw[w] == sum(s) and r.tw[w] == len(s) and len(par) == len(s), key
                assert all(x > 0 for x in s)
                for j in (par if r.parents else ()):
                    sent[(r.parents[j], w)] = sent.get((r.parents[j], w), 0) + 1
        for key, r in self.R.items():
            for w in range(self.nv):
                assert r.cw[w] == r.obs[w] + sent.get((key, w), 0), (key, w)


class Snapshot:
    """Frozen predictive of a GraphPY: counts, hyperparameters and mixing
    weights at one sweep. ``vec(key)`` is the next-symbol distribution."""

    def __init__(self, m: GraphPY):
        self.nv = m.nv
        self.V = m.V
        self.spec = m.spec
        self.base = np.full(m.nv, 1.0 / m.nv)
        self.counts = {k: (r.c, r.t, np.array(r.cw, float), np.array(r.tw, float))
                       for k, r in m.R.items() if r.c > 0}
        self.hyp = {k: m.hyper(k) for k in self.counts}
        self.lam = {g: v.copy() for g, v in m.lam.items()}
        self._vec = {}

    def vec(self, key):
        v = self._vec.get(key)
        if v is not None:
            return v
        ps = self.spec.parents(key)
        if not ps:
            base = self.base
        elif len(ps) == 1:
            base = self.vec(ps[0])
        else:
            lam = self.lam.get(self.spec.mixgroup(key), np.full(len(ps), 1.0 / len(ps)))
            base = sum(l * self.vec(p) for l, p in zip(lam, ps))
        cnt = self.counts.get(key)
        if cnt is None:
            v = base
        else:
            c, t, cw, tw = cnt
            d, th = self.hyp[key]
            v = (cw - d * tw + (th + d * t) * base) / (th + c)
        self._vec[key] = v
        return v


# ------------------------------------------------------------- tree models
class TreeHPY:
    """A (graphical) hierarchical PY prior over expression trees: every node's
    symbol is drawn from the distribution of its context, children follow
    from the symbol's arity (hsr.treecode). Training trees can carry a source
    label (for ContextSpec(source=True))."""

    def __init__(self, spec: ContextSpec, seed=0):
        self.spec = spec
        self.V = tc.V
        self.seed = seed

    def fit(self, trees, sources=None, n_sweeps=20, n_keep=10):
        sources = sources or [None] * len(trees)
        data = [(self.spec.key(p, s), sym) for t, s in zip(trees, sources) for p, sym in tc.events(t)]
        self.model = GraphPY(self.V, self.spec, seed=self.seed).fit(data, n_sweeps, n_keep)
        self.samples = self.model.samples
        return self

    def logp_sample(self, snap, tree, src=None):
        return float(sum(np.log(snap.vec(self.spec.key(p, src))[tc.V.index(s)]) for p, s in tc.events(tree)))

    def logp(self, tree, src=None):
        """log of the posterior predictive probability of a tree (mixture over
        the retained samples)."""
        v = np.array([self.logp_sample(s, tree, src) for s in self.samples])
        return float(np.logaddexp.reduce(v) - np.log(len(v)))

    def cond_dist(self, snap, src=None):
        """node_dist for hsr.treecheck: pos -> ([], 1, next-symbol distribution)."""
        return lambda pos: ([], 1.0, snap.vec(self.spec.key(pos, src)))

    def hyper_summary(self):
        m = self.model
        return {"d": {str(g): float(v) for g, v in m.d.items()},
                "theta": {str(g): float(v) for g, v in m.theta.items()},
                "lambda": {str(g): v.tolist() for g, v in m.lam.items()}}
