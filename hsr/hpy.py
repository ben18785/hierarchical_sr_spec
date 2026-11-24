"""Hierarchical Pitman-Yor language model (Teh 2006) over the tree events of
fnprior: the fully Bayesian counterpart of its Kneser-Ney smoothing.

Each event is (context..., symbol), with the context the two preceding
symbols in fnprior's tree coding (parent and grandparent for a first child;
parent and left sibling for a second child). For a context u = (a, b) the
next-symbol distribution is

    G_u ~ PY(d_|u|, theta_|u|, G_pi(u)),   pi(a, b) = (b,),  pi(b,) = (),
    G_() ~ PY(d_0, theta_0, Uniform(V)),

with V the symbols allowed in that role. Posterior inference is by Gibbs
sampling of the Chinese-restaurant seating (every training event is a
customer; a new table in restaurant u sends a customer to pi(u)), and the
discount and strength of each depth are updated by random-walk Metropolis on
the Pitman-Yor seating likelihood with priors d ~ Beta(1, 1), theta ~ Gamma(1, 1).
Predictive probabilities are averaged over the retained samples.

Imports only numpy and scipy, so it also runs in the workflow's fnprior
environment.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


class HPY:
    def __init__(self, vocab, depth=2, seed=0):
        self.V = sorted(vocab)
        self.base = 1.0 / len(self.V)
        self.depth = depth
        self.d = np.full(depth + 1, 0.5)
        self.theta = np.full(depth + 1, 1.0)
        self.rng = np.random.default_rng(seed)
        # restaurant u -> {w: [table sizes]}; c[u] customers, t[u] tables
        self.tables = {}
        self.c = {}
        self.t = {}
        self.cw = {}       # (u, w) -> customers
        self.tw = {}       # (u, w) -> tables

    # ----------------------------------------------------------- predictive
    def prob(self, u, w):
        """p(w | u) under the current seating and hyperparameters."""
        if u is None:
            return self.base
        parent = self.prob(u[1:] if len(u) else None, w)
        cu = self.c.get(u, 0)
        m = len(u)
        d, th = self.d[m], self.theta[m]
        if cu == 0:
            return parent
        return ((self.cw.get((u, w), 0) - d * self.tw.get((u, w), 0))
                + (th + d * self.t.get(u, 0)) * parent) / (th + cu)

    # -------------------------------------------------------------- seating
    def _add(self, u, w):
        if u is None:
            return
        m = len(u)
        d, th = self.d[m], self.theta[m]
        sizes = self.tables.setdefault(u, {}).setdefault(w, [])
        parent_p = self.prob(u[1:] if len(u) else None, w)
        wts = np.array([s - d for s in sizes] + [(th + d * self.t.get(u, 0)) * parent_p])
        k = int(self.rng.choice(len(wts), p=wts / wts.sum()))
        self.c[u] = self.c.get(u, 0) + 1
        self.cw[(u, w)] = self.cw.get((u, w), 0) + 1
        if k == len(sizes):
            sizes.append(1)
            self.t[u] = self.t.get(u, 0) + 1
            self.tw[(u, w)] = self.tw.get((u, w), 0) + 1
            self._add(u[1:] if len(u) else None, w)
        else:
            sizes[k] += 1

    def _remove(self, u, w):
        if u is None:
            return
        sizes = self.tables[u][w]
        k = int(self.rng.choice(len(sizes), p=np.array(sizes) / sum(sizes)))
        sizes[k] -= 1
        self.c[u] -= 1
        self.cw[(u, w)] -= 1
        if sizes[k] == 0:
            sizes.pop(k)
            self.t[u] -= 1
            self.tw[(u, w)] -= 1
            self._remove(u[1:] if len(u) else None, w)

    # --------------------------------------------------------- hyperparameters
    def _loglik_depth(self, m, d, th):
        """log P(seating at depth m | d, theta) (Pitman-Yor EPPF, product over restaurants)."""
        if not (0 < d < 1 and th > -d):
            return -np.inf
        tot = 0.0
        for u, ws in self.tables.items():
            if len(u) != m or self.c.get(u, 0) == 0:
                continue
            t, cu = self.t[u], self.c[u]
            # prod_{i=1}^{t-1} (theta + i d) / (theta + 1)_{c-1}
            tot += np.sum(np.log(th + d * np.arange(1, t))) - (gammaln(th + cu) - gammaln(th + 1))
            for sizes in ws.values():
                for s in sizes:
                    tot += gammaln(s - d) - gammaln(1 - d)
        return tot

    def _update_hyper(self, steps=5):
        for m in range(self.depth + 1):
            for _ in range(steps):
                d0, t0 = self.d[m], self.theta[m]
                d1 = d0 + self.rng.normal(0, 0.05)
                t1 = t0 * np.exp(self.rng.normal(0, 0.2))
                lp0 = self._loglik_depth(m, d0, t0) - t0
                lp1 = self._loglik_depth(m, d1, t1) - t1 + np.log(t1) - np.log(t0)   # Gamma(1,1) prior, log-scale proposal
                if np.log(self.rng.random()) < lp1 - lp0:
                    self.d[m], self.theta[m] = d1, t1

    # --------------------------------------------------------------- fitting
    def fit(self, events, n_sweeps=20, n_keep=10, hyper_every=1):
        """events: list of (context tuple of length depth, symbol). Returns self
        with ``samples``: list of (d, theta, cw, tw, c, t) snapshots."""
        events = [(tuple(u), w) for u, w in events]
        for u, w in events:
            self._add(u, w)
        self.samples = []
        for sweep in range(n_sweeps):
            for i in self.rng.permutation(len(events)):
                u, w = events[i]
                self._remove(u, w)
                self._add(u, w)
            if sweep % hyper_every == 0:
                self._update_hyper()
            if sweep >= n_sweeps - n_keep:
                self.samples.append((self.d.copy(), self.theta.copy(), dict(self.cw), dict(self.tw), dict(self.c),
                                     dict(self.t)))
        return self

    def predictive(self, u, w):
        """p(w | u) averaged over the retained samples."""
        u = tuple(u)
        cur = (self.d, self.theta, self.cw, self.tw, self.c, self.t)
        ps = []
        for d, th, cw, tw, c, t in self.samples:
            self.d, self.theta, self.cw, self.tw, self.c, self.t = d, th, cw, tw, c, t
            ps.append(self.prob(u, w))
        self.d, self.theta, self.cw, self.tw, self.c, self.t = cur
        return float(np.mean(ps))
