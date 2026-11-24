"""Behaviour classes: skeletons that the data cannot tell apart.

Two skeletons are *behaviourally equivalent* on a population if each can
reproduce the other's fitted curves, on the datasets where either is
competitive, to within ``nats`` of log likelihood. For dataset j with points
x_j and noise sd sigma_j, skeleton b reproduces a's fitted curve if

    min_phi  sum_i (f_b(x_ji; phi) - f_a(x_ji; phi_ja))^2 / (2 sigma_j^2)  <  nats,

where phi_ja is a's MAP fit to dataset j: the best version of b loses less
than ``nats`` of log likelihood against a's fit, so no dataset of this size
and noise could separate them. The test runs in both directions.

Classes are built greedily in order of posterior mass, with complete linkage
(a skeleton joins a class only if it is equivalent to every member), so that
chains of near-equivalences do not merge distinct behaviours.

The comparison is made on a dense grid spanning each dataset's x-range (not
only at the observed points, so forms that agree at the data but differ in a
gap between points are kept apart), with the squared discrepancy averaged over
the grid and multiplied by n_j, so the threshold still reads as nats for a
dataset of that size.

End behaviour. In-range equivalence says nothing about what a form does
beyond the data. Each (form, dataset) pair also gets an *end behaviour*: as
x -> infinity the fitted curve tends to 0, levels off at a finite non-zero
value, grows without bound, or does something else (oscillates, becomes
undefined); and as x -> 0 it tends to 0, to a finite non-zero value, or
diverges. It is a property of the form with the constants the data support,
so it is computed for draws from each dataset's Laplace posterior of the
constants: for c0*exp(c1*x) it depends on the sign of c1, which the data fix.
It is not something the data observe directly: it is what each candidate form
*assumes* outside the data, and reporting it shows where the data are silent.
Two forms are put in the same class only if they also agree on their most
probable end behaviour on every tested dataset.

Classes are used for *reporting* only: the posterior of a class is the sum of
its members' posteriors. The model and sampler are unchanged.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares


RIGHT = ("to_zero", "levels_off", "unbounded", "other")
LEFT = ("zero", "nonzero", "unbounded", "other")


def _limit_label(v, scale, labels):
    """v: (n_draws, m) values at m points approaching the limit point
    geometrically. A non-monotone tail is 'other' (e.g. oscillation). A
    monotone tail has converged if its last steps are at most half the
    earlier ones (or at rounding level); the limit is then zero or finite
    non-zero. A monotone tail that has not converged is unbounded."""
    zero, finite, unbounded, other = labels
    out = []
    tol = 1e-12 * scale
    for row in v:
        if np.isinf(row[-1]) or (np.isnan(row[-1]) and np.any(np.abs(row[np.isfinite(row)]) > 1e6 * scale)):
            out.append(unbounded)
            continue
        if not np.all(np.isfinite(row)):
            out.append(other)
            continue
        d = np.diff(row)
        if not (np.all(d >= -tol) or np.all(d <= tol)):
            out.append(other)
            continue
        late, early = abs(row[-1] - row[-5]), abs(row[-5] - row[-9])
        if late <= 0.5 * early or late < 1e-10 * scale:
            out.append(zero if abs(row[-1]) < 1e-3 * scale else finite)
        else:
            out.append(unbounded)
    return out


def end_behaviour(cs, phis, x_lo, x_hi, scale):
    """Lists of right (x -> inf) and left (x -> 0) labels, one per row of
    phis, from 25 points spaced geometrically from 1e3 to 1e9 times the
    largest x (and 1e-3 to 1e-9 times the smallest)."""
    phis = np.atleast_2d(np.asarray(phis, float))
    res = []
    for pts, labels in ((x_hi * np.logspace(3, 9, 25), RIGHT), (x_lo * np.logspace(-3, -9, 25), LEFT)):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = np.broadcast_to(pts[None, :, None], (len(phis), len(pts), 1))
            v = np.broadcast_to(np.asarray(cs.f(X, phis), float), (len(phis), len(pts)))
        res.append(_limit_label(v, scale, labels))
    return res[0], res[1]


class Equivalence:
    def __init__(self, skeletons, table, X, M=None, nats=1.0, max_datasets=8, compiled=None, seed=0,
                 n_grid=60, n_draws=40):
        self.skeletons = skeletons
        self.table = table
        self.X = np.asarray(X, float)
        if self.X.ndim == 2:
            self.X = self.X[..., None]
        self.M = np.ones(self.X.shape[:2], bool) if M is None else np.asarray(M, bool)
        self.nats = nats
        self.max_datasets = max_datasets
        self._compiled = compiled
        self.rng = np.random.default_rng(seed)
        self._memo = {}
        self._ends = {}
        self.n_draws = n_draws
        self._grids = []
        for j in range(self.X.shape[0]):
            xj = self.X[j][self.M[j], 0]
            self._grids.append(np.linspace(xj.min(), xj.max(), n_grid)[:, None])
        L = table.laplace
        self._rel = L - np.max(np.where(np.isfinite(L), L, -np.inf), axis=1, keepdims=True)

    def _cs(self, e):
        if self._compiled is not None:
            return self._compiled(self.skeletons[e])
        from .likelihood import CompiledSkeleton
        return CompiledSkeleton(self.skeletons[e])

    def _f(self, cs, x, phi):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(cs.f(x[None], np.asarray(phi, float)[None, :])[0], float)

    def _reproduces(self, a, b, j):
        """True if skeleton b reproduces a's fit to dataset j within self.nats."""
        phi_a = np.asarray(self.table.phi[a][j], float)
        if not np.all(np.isfinite(phi_a)):
            return None
        x = self._grids[j]
        n_j = int(self.M[j].sum())
        cs_a, cs_b = self._cs(a), self._cs(b)
        target = self._f(cs_a, x, phi_a)
        if not np.all(np.isfinite(target)):
            return None
        sigma = float(np.exp(self.table.log_sigma[j, a]))
        if not np.isfinite(sigma) or sigma <= 0:
            return None

        w = np.sqrt(n_j / len(x))          # sum over the grid scaled to n_j points

        def res(p):
            r = w * (self._f(cs_b, x, p) - target) / sigma
            return np.where(np.isfinite(r), r, 1e6)

        if cs_b.k == 0:
            return bool(0.5 * np.sum(res(np.zeros(0)) ** 2) < self.nats)
        phi_b = np.asarray(self.table.phi[b][j], float)
        starts = [phi_b if np.all(np.isfinite(phi_b)) else np.zeros(cs_b.k)]
        starts += [starts[0] + self.rng.normal(0, 0.5, cs_b.k) for _ in range(2)]
        best = np.inf
        for p0 in starts:
            try:
                sol = least_squares(res, p0, method="lm", max_nfev=200 * (cs_b.k + 1))
            except Exception:
                continue
            best = min(best, 0.5 * float(np.sum(sol.fun ** 2)))
            if best < self.nats:
                return True
        return False

    def end_labels(self, e, j):
        """(right, left) label distributions for form e fitted to dataset j,
        over draws from its Laplace posterior: two dicts label -> prob, or
        None if the fit failed."""
        key = (int(e), int(j))
        if key in self._ends:
            return self._ends[key]
        phi = np.asarray(self.table.phi[e][j], float)
        out = None
        if np.all(np.isfinite(phi)):
            cs = self._cs(e)
            k = len(phi)
            draws = phi[None, :]
            cov = np.asarray(self.table.cov[e][j], float)
            if k and cov.size and np.all(np.isfinite(cov)):
                try:
                    Lc = np.linalg.cholesky(cov[:k, :k] + 1e-12 * np.eye(k))
                    draws = np.vstack([phi, phi + self.rng.standard_normal((self.n_draws - 1, k)) @ Lc.T])
                except np.linalg.LinAlgError:
                    pass
            g = self._grids[j][:, 0]
            fit = self._f(cs, self._grids[j], phi)
            scale = float(np.sqrt(np.nanmean(fit ** 2))) if np.any(np.isfinite(fit)) else 1.0
            r, l = end_behaviour(cs, draws, g.min(), g.max(), max(scale, 1e-12))
            out = tuple({lab: c / len(labs) for lab, c in zip(*np.unique(labs, return_counts=True))}
                        for labs in (r, l))
        self._ends[key] = out
        return out

    def _same_ends(self, a, b, j):
        ea, eb = self.end_labels(a, j), self.end_labels(b, j)
        if ea is None or eb is None:
            return None
        mode = lambda d: max(d, key=d.get)
        return mode(ea[0]) == mode(eb[0]) and mode(ea[1]) == mode(eb[1])

    def end_summary(self, probs, min_p=0.01) -> dict:
        """Per dataset, the posterior over end behaviour, mixing over forms
        (weights = posterior form probabilities >= min_p, renormalised) and
        over each form's constants."""
        probs = np.asarray(probs)
        right, left = [], []
        for j in range(probs.shape[0]):
            idx = np.flatnonzero(probs[j] >= min_p)
            if len(idx) == 0:
                idx = np.array([int(np.argmax(probs[j]))])
            R, Lft, tot = {}, {}, 0.0
            for e in idx:
                el = self.end_labels(e, j)
                if el is None:
                    continue
                w = float(probs[j, e])
                tot += w
                for d, acc in zip(el, (R, Lft)):
                    for lab, p in d.items():
                        acc[lab] = acc.get(lab, 0.0) + w * p
            right.append({k: round(v / tot, 3) for k, v in R.items()} if tot else {})
            left.append({k: round(v / tot, 3) for k, v in Lft.items()} if tot else {})
        mean = lambda ds, labs: {lab: round(float(np.mean([d.get(lab, 0.0) for d in ds])), 3) for lab in labs}
        return {"end_right": right, "end_left": left,
                "end_right_mean": mean(right, RIGHT), "end_left_mean": mean(left, LEFT)}

    def equivalent(self, a, b) -> bool:
        a, b = int(a), int(b)
        if a == b:
            return True
        key = (min(a, b), max(a, b))
        if key in self._memo:
            return self._memo[key]
        # the datasets on which either skeleton is competitive
        score = np.maximum(self._rel[:, a], self._rel[:, b])
        js = [j for j in np.argsort(score)[::-1][:self.max_datasets] if np.isfinite(score[j])]
        ok, tested = True, 0
        for j in js:
            if self._same_ends(a, b, j) is False:
                ok = False
                break
            for u, v in ((a, b), (b, a)):
                r = self._reproduces(u, v, j)
                if r is None:
                    continue
                tested += 1
                if not r:
                    ok = False
                    break
            if not ok:
                break
        self._memo[key] = bool(ok and tested > 0)
        return self._memo[key]

    def classes(self, probs, min_mass=0.01, max_forms=60) -> dict:
        """{skeleton index: representative index} for the skeletons with
        posterior probability >= min_mass for some dataset (at most
        max_forms, by total mass). The representative is the member with
        the largest total mass."""
        probs = np.asarray(probs)
        mass = probs.sum(axis=0)
        cand = np.flatnonzero(probs.max(axis=0) >= min_mass)
        cand = cand[np.argsort(mass[cand])[::-1]][:max_forms]
        reps = []          # list of member lists; first member is the representative
        for e in cand:
            for members in reps:
                if all(self.equivalent(e, m) for m in members):
                    members.append(int(e))
                    break
            else:
                reps.append([int(e)])
        return {m: members[0] for members in reps for m in members}


def class_summary(probs, class_of, skeletons, z_samples=None, top=5, min_p=0.01, true_idx=None) -> dict:
    """Posterior summaries at the level of behaviour classes. Skeletons not
    in ``class_of`` are their own class."""
    probs = np.asarray(probs)
    J, C = probs.shape
    rep = np.arange(C)
    for e, r in class_of.items():
        rep[e] = r
    reps = np.unique(rep)
    Pc = np.zeros((J, len(reps)))
    col = {r: i for i, r in enumerate(reps)}
    np.add.at(Pc, (slice(None), np.array([col[r] for r in rep])), probs)   # sum members
    key = lambda r: skeletons[int(r)].key
    out = {}
    members = {}
    for e, r in class_of.items():
        members.setdefault(r, []).append(e)
    out["class_members"] = {key(r): [key(m) for m in sorted(ms, key=lambda m: -probs[:, m].sum())]
                            for r, ms in members.items() if len(ms) > 1}
    topc = np.argmax(Pc, axis=1)
    out["top_class_keys"] = [key(reps[i]) for i in topc]
    out["top_class_prob"] = [round(float(Pc[j, i]), 4) for j, i in enumerate(topc)]
    gp = []
    for j in range(J):
        idx = np.argsort(Pc[j])[::-1][:top]
        gp.append([[key(reps[i]), round(float(Pc[j, i]), 4)] for i in idx if Pc[j, i] >= min_p])
    out["group_class_posterior"] = gp
    w = Pc.mean(axis=0)
    order = np.argsort(w)[::-1][:15]
    out["class_pop_weight"] = {key(reps[i]): round(float(w[i]), 4) for i in order if w[i] > 1e-3}
    if z_samples is not None:
        zc = rep[np.asarray(z_samples)]                                  # (chains, draws, J)
        nc = np.array([[len(np.unique(d)) for d in ch] for ch in zc]).ravel()
        vals, cnt = np.unique(nc, return_counts=True)
        out["n_classes_posterior"] = {int(v): round(float(c / cnt.sum()), 4) for v, c in zip(vals, cnt)}
        flat = zc.reshape(-1, zc.shape[-1])
        out["class_coassignment"] = np.round((flat[:, :, None] == flat[:, None, :]).mean(axis=0), 3).tolist()
    if true_idx is not None:
        t = np.asarray(true_idx)
        ok = [(t[j] >= 0) and rep[t[j]] == reps[topc[j]] for j in range(J)]
        out["class_recovery"] = float(np.mean(ok))
    return out
