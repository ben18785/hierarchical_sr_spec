"""Themes and variations: a second level of pooling over skeletons.

Model (Overleaf note, Section 5.3). Datasets are partitioned into clusters
by a Chinese restaurant process CRP(gamma); each cluster b has a skeleton e_b
(distinct clusters have distinct skeletons) and a theme t_b. The themes of
the clusters are an exchangeable sequence from DP(gamma0, P0), with P0 the
grammar prior normalised over the candidate set, so

    P(t_b = t | other themes) ∝ m_t + gamma0 P0(t)     (m_t: other clusters with theme t)

and the cluster's skeleton is a variation on its theme,

    e_b | t_b ~ K(. | t_b),
    K(e | t) = rho                     if e = t   (t has at least one known edit)
             = (1 - rho) w_t(e)        if e in N(t)
             = 1                       if e = t and N(t) is empty.

N(t) are the known one-step edits of t (hsr.candidates.edit_graph), and
w_t(e) is either uniform, 1/|N(t)|, or proportional to the grammar prior at
its prior-mean theta, P(e)/sum_{e' in N(t)} P(e') (fixed for the run, so the
theta update stays exact). The
cluster's datasets then contribute the cluster marginal M_e(S) (independent or
pooled constants, hsr.pooled), exactly as in hsr.gibbs.

Collapsed Gibbs, all steps exact for this joint (theta and gamma0 updates
follow the same conventions as hsr.gibbs):
  1. z_j: an existing cluster e with weight n_e^{-j} exp(M_e(S+j) - M_e(S)); an
     unused skeleton e with weight gamma Q(e) exp(M_e({j})), where
     Q(e) = sum_t (m_t + gamma0 P0(t)) K(e | t) / (B + gamma0); a new cluster
     then draws its theme from P(t | e) ∝ (m_t + gamma0 P0(t)) K(e | t).
  2. block move: each cluster's (skeleton, theme) pair is redrawn with its
     members fixed, ∝ (m_t + gamma0 P0(t)) K(e | t) exp(M_e(S)).
  3. each cluster's theme alone, ∝ (m_t + gamma0 P0(t)) K(e_b | t).
  4. rho ~ Beta(a + #clusters on their theme, b + #clusters on an edit).
  5. theta from the rule counts of the distinct themes; gamma (clusters among
     datasets) and gamma0 (themes among clusters) by Escobar-West.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from scipy.special import logsumexp

from scipy.special import gammaln

from .gibbs import GibbsConfig, GibbsResult, _log_similarity, escobar_west_gamma
from .grammar import Grammar


@dataclass
class ThemeConfig:
    rho_a: float = 2.0      # Beta(2, 1): mild preference for a cluster sitting on its theme
    rho_b: float = 1.0
    rho_init: float = 0.5
    update_rho: bool = True
    gamma0_a: float = 1.0
    gamma0_b: float = 1.0
    gamma0_init: float = 1.0
    update_gamma0: bool = True


class EditKernel:
    """Sparse kernel K(e | t) built from (parent, child) edges over C candidates."""

    def __init__(self, n_cand: int, edges, log_w=None):
        """``log_w`` (C,), optional: log weights of the candidates as variations
        (the log grammar prior at a fixed theta). Each theme then shares its
        variation mass 1 - rho in proportion to exp(log_w) over N(t), not
        uniformly, so the grammar's ranking of variations is kept."""
        self.C = n_cand
        edges = sorted({(int(a), int(b)) for a, b in edges if a != b})
        rows = np.array([a for a, _ in edges], int)
        cols = np.array([b for _, b in edges], int)
        self.n_children = np.bincount(rows, minlength=n_cand).astype(float)
        self.has_children = self.n_children > 0
        if not len(rows):
            vals = np.zeros(0)
        elif log_w is None:
            vals = 1.0 / self.n_children[rows]
        else:
            lw = np.asarray(log_w, float)[cols]
            lw = np.where(np.isfinite(lw), lw, -np.inf)
            # normalise within each row (rows are sorted, so contiguous)
            starts = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
            lse = np.logaddexp.reduceat(lw, starts)
            vals = np.exp(lw - np.repeat(lse, np.diff(np.r_[starts, len(rows)])))
        self.weighted = log_w is not None
        # A[t, e] = weight of e among the variations N(t); rows sum to 1
        self.A = sps.csr_matrix((vals, (rows, cols)), shape=(n_cand, n_cand))
        self.At = self.A.T.tocsr()          # row e lists the parents of e
        self.n_edges = len(edges)

    def self_weight(self, rho):
        return np.where(self.has_children, rho, 1.0)

    def row(self, t, rho):
        """(indices, values) of K(. | t)."""
        a = self.A
        idx = a.indices[a.indptr[t]:a.indptr[t + 1]]
        val = (1 - rho) * a.data[a.indptr[t]:a.indptr[t + 1]]
        return np.concatenate([[t], idx]), np.concatenate([[rho if self.has_children[t] else 1.0], val])

    def parents(self, e, rho):
        """(theme indices, K(e | t)) for every theme that can produce e (including e itself)."""
        a = self.At
        idx = a.indices[a.indptr[e]:a.indptr[e + 1]]
        val = (1 - rho) * a.data[a.indptr[e]:a.indptr[e + 1]]
        return np.concatenate([[e], idx]), np.concatenate([[rho if self.has_children[e] else 1.0], val])

    def push(self, w, rho):
        """Q_unnorm(e) = sum_t w_t K(e | t) for a weight vector w over themes."""
        return w * self.self_weight(rho) + (1 - rho) * (self.At @ w)

    def push2(self, w, rho):
        """sum_t w_t K(e | t)^2."""
        if not hasattr(self, "_A2t"):
            self._A2t = self.A.multiply(self.A).T.tocsr()
        return w * self.self_weight(rho) ** 2 + (1 - rho) ** 2 * (self._A2t @ w)

    def pull(self, a, rho):
        """sum_e K(e | t) a(e) for every theme t."""
        return a * self.self_weight(rho) + (1 - rho) * (self.A @ a)

    def pairs(self, rho):
        """(themes, skeletons, K(e | t)) for every pair with K > 0."""
        if not hasattr(self, "_pairs"):
            coo = self.A.tocoo()
            self._pairs = (np.r_[np.arange(self.C), coo.row], np.r_[np.arange(self.C), coo.col],
                           np.r_[np.zeros(self.C), coo.data], np.r_[np.ones(self.C), np.zeros(len(coo.data))])
        t, e, aval, diag = self._pairs
        return t, e, np.where(diag > 0, self.self_weight(rho)[t], (1 - rho) * aval)


@dataclass
class ThemeResult(GibbsResult):
    theme_probs: np.ndarray = None      # (J, C) posterior P(theme of dataset j's cluster = t)
    n_themes: np.ndarray = None         # (n_chains, n_kept)
    rho: np.ndarray = None
    gamma0: np.ndarray = None

    @property
    def theme_top(self):
        return np.argmax(self.theme_probs, axis=1)


def _split_merge_themes(rng, z, n_e, members, Mc, theme, m, model, kernel, rho, gamma, gamma0, P0, logsim, n_prop):
    """Split-merge moves for the themes model, with the skeletons and themes of
    the clusters involved summed out. Allocation as in hsr.gibbs._split_merge.

    With w(t) = m_t + gamma0 P0(t) counting the B0 other clusters' themes, and
    a_S(e) = exp M_e(S) over the skeletons not used by other clusters, the
    weight of one cluster S (CRP(gamma) term, theme and skeleton summed) is
        gamma (|S|-1)! sum_e push(w)(e) a_S(e) / (B0 + gamma0)
    and of two clusters S1, S2 with distinct skeletons
        gamma^2 (|S1|-1)! (|S2|-1)! T / ((B0 + gamma0)(B0 + 1 + gamma0)),
        T = sum_{t1,t2,e1!=e2} w(t1) (w(t2) + [t2 = t1]) K(e1|t1) K(e2|t2) a1(e1) a2(e2)
          = (w.u1)(w.u2) + sum_t w u1 u2 - sum_e a1 a2 (push(w)^2 + push2(w)),
    with u = pull(a). After an accepted move the new skeletons and themes are
    drawn from their exact conditional."""
    J = len(z)
    C = len(n_e)
    for _ in range(n_prop):
        i, j = rng.choice(J, size=2, replace=False)
        ci, cj = int(z[i]), int(z[j])
        involved = [ci] if ci == cj else [ci, cj]
        old_themes = {c: theme[c] for c in involved}
        for c in involved:
            m[old_themes[c]] -= 1
        B0 = len(members) - len(involved)
        w = m + gamma0 * P0
        pw = kernel.push(w, rho)
        allowed = n_e == 0
        for c in involved:
            allowed[c] = True

        def scaled(T):
            M = np.where(allowed, model.marginal_all(sorted(T)), -np.inf)
            c0 = M.max()
            return np.exp(M - c0) if np.isfinite(c0) else np.zeros(C), c0

        if ci == cj:
            S = members[ci]
            Si, Sj, logq = {int(i)}, {int(j)}, 0.0
            for k in rng.permutation(sorted(S - {i, j})):
                p_i = 1.0 / (1.0 + np.exp(logsim[k, j] - logsim[k, i]))
                if rng.random() < p_i:
                    Si.add(int(k))
                    logq += np.log(p_i)
                else:
                    Sj.add(int(k))
                    logq += np.log1p(-p_i)
        else:
            Si, Sj = members[ci], members[cj]
            S = Si | Sj
            logq = 0.0
            for k in S - {i, j}:
                p_i = 1.0 / (1.0 + np.exp(logsim[k, j] - logsim[k, i]))
                logq += np.log(p_i) if k in Si else np.log1p(-p_i)
        a, ca = scaled(S)
        a1, c1 = scaled(Si)
        a2, c2 = scaled(Sj)
        W1 = float(pw @ a)
        u1, u2 = kernel.pull(a1, rho), kernel.pull(a2, rho)
        T = float((w @ u1) * (w @ u2) + np.sum(w * u1 * u2) - np.sum(a1 * a2 * (pw ** 2 + kernel.push2(w, rho))))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_split = (np.log(gamma) + gammaln(len(Si)) + gammaln(len(Sj)) + np.log(max(T, 0.0)) + c1 + c2
                         - np.log(B0 + 1 + gamma0))
            log_merge = gammaln(len(S)) + np.log(W1) + ca
        log_a = (log_split - log_merge - logq) if ci == cj else (log_merge - log_split + logq)
        if np.isnan(log_a) or np.log(rng.random()) >= log_a:
            for c in involved:
                m[old_themes[c]] += 1
            continue
        for c in involved:
            del members[c]
            Mc.pop(c, None)
            n_e[c] = 0
            theme.pop(c)
        if ci == cj:
            # (e1, t1) from its marginal, then (e2, t2) given it
            tt, ee, kv = kernel.pairs(rho)
            R = (w @ u2 + u2[tt]) - a2[ee] * (pw[ee] + kv)
            lw = np.log(np.maximum(w[tt] * kv * a1[ee] * np.maximum(R, 0.0), 1e-300))
            lw[(kv <= 0) | (a1[ee] <= 0) | (R <= 0)] = -np.inf
            k1 = int(rng.choice(len(lw), p=np.exp(lw - logsumexp(lw))))
            e1, t1 = int(ee[k1]), int(tt[k1])
            m[t1] += 1
            w2 = m + gamma0 * P0
            lw2 = np.log(np.maximum(kernel.push(w2, rho) * a2, 1e-300))
            lw2[(a2 <= 0)] = -np.inf
            lw2[e1] = -np.inf
            e2 = int(rng.choice(C, p=np.exp(lw2 - logsumexp(lw2))))
            idx, kv2 = kernel.parents(e2, rho)
            lt = np.log(w2[idx]) + np.log(kv2)
            t2 = int(idx[rng.choice(len(idx), p=np.exp(lt - logsumexp(lt)))])
            m[t2] += 1
            new = ((e1, t1, Si), (e2, t2, Sj))
        else:
            lwm = np.log(np.maximum(pw * a, 1e-300))
            lwm[a <= 0] = -np.inf
            e = int(rng.choice(C, p=np.exp(lwm - logsumexp(lwm))))
            idx, kv = kernel.parents(e, rho)
            lt = np.log(w[idx]) + np.log(kv)
            t = int(idx[rng.choice(len(idx), p=np.exp(lt - logsumexp(lt)))])
            m[t] += 1
            new = ((e, t, S),)
        for e, t, T_ in new:
            members[e] = set(T_)
            n_e[e] = len(T_)
            z[sorted(T_)] = e
            Mc[e] = model.marginal(e, sorted(T_))
            theme[e] = t


def _p0(log_p):
    return np.exp(log_p - logsumexp(log_p))


def run_chain_themes(L, counts, grammar: Grammar, kernel: EditKernel, cfg: GibbsConfig, tcfg: ThemeConfig,
                     seed: int, init="argmax", model=None, z_init=None):
    from .pooled import IndependentModel
    rng = np.random.default_rng(seed)
    model = IndependentModel(L) if model is None else model
    S1 = model.singletons()
    J, C = S1.shape
    theta = grammar.prior_mean()
    gamma, gamma0, rho = cfg.gamma_init, tcfg.gamma0_init, tcfg.rho_init
    log_p = grammar.log_prob(counts, theta)
    P0 = _p0(log_p)
    if z_init is not None:
        z = np.asarray(z_init, int).copy()
    elif init == "argmax":
        z = np.argmax(log_p[None, :] + S1, axis=1)
    else:
        lw = log_p[None, :] + S1
        z = np.array([rng.choice(C, p=np.exp(r - logsumexp(r))) for r in lw])
    members = {}
    for j, e in enumerate(z):
        members.setdefault(int(e), set()).add(j)
    n_e = np.bincount(z, minlength=C).astype(float)
    Mc = {e: model.marginal(e, sorted(S)) for e, S in members.items()}
    theme = {e: e for e in members}                   # cluster skeleton -> theme
    m = np.zeros(C)                                    # clusters per theme
    for t in theme.values():
        m[t] += 1

    def theme_weights():
        return m + gamma0 * P0

    def add_theme(t, d):
        m[t] += d

    def draw_theme(e):
        idx, kv = kernel.parents(e, rho)
        lw = np.log(theme_weights()[idx]) + np.log(kv)
        p = np.exp(lw - lw.max())
        return int(idx[rng.choice(len(idx), p=p / p.sum())])

    assign = np.zeros((J, C))
    theme_counts = np.zeros((J, C))
    pop = np.zeros(C)
    n_forms, n_themes, gammas, gamma0s, rhos, kept, log_liks = [], [], [], [], [], [], []
    thetas = {nt: [] for nt in grammar.nonterminals}
    for sweep in range(cfg.n_sweeps):
        # ---- 1. single-dataset moves
        for j in (rng.permutation(J) if cfg.single_site else ()):
            e0 = int(z[j])
            members[e0].discard(j)
            n_e[e0] -= 1
            if members[e0]:
                Mc[e0] = model.marginal(e0, sorted(members[e0]))
            else:
                del members[e0]
                Mc.pop(e0, None)
                add_theme(theme.pop(e0), -1)
            B = len(members)
            Q = kernel.push(theme_weights(), rho) / (B + gamma0)
            with np.errstate(divide="ignore"):
                lw = np.log(gamma) + np.log(Q) + S1[j]
            joint = {}
            for e, S in members.items():
                joint[e] = model.marginal(e, sorted(S | {j}))
                lw[e] = np.log(n_e[e]) + joint[e] - Mc[e]
            top = lw.max()
            if not np.isfinite(top):
                lw = np.where(np.isfinite(S1[j]), 0.0, -np.inf) if np.isfinite(S1[j]).any() else np.zeros(C)
                top = lw.max()
            p = np.exp(lw - top)
            e1 = int(rng.choice(C, p=p / p.sum()))
            z[j] = e1
            n_e[e1] += 1
            if e1 in members:
                members[e1].add(j)
                Mc[e1] = joint[e1]
            else:
                members[e1] = {j}
                Mc[e1] = S1[j, e1]
                t = draw_theme(e1)
                theme[e1] = t
                add_theme(t, +1)
        # ---- 1b. split-merge (hsr.gibbs._split_merge, with themes)
        if cfg.split_merge and J > 1:
            _split_merge_themes(rng, z, n_e, members, Mc, theme, m, model, kernel, rho, gamma, gamma0, P0,
                                _log_similarity(S1, np.log(np.maximum(P0, 1e-300))), cfg.split_merge)
        # ---- 2. block move: (skeleton, theme) of each cluster
        if cfg.relabel and sweep % cfg.relabel_every == 0:
            for e in rng.permutation(list(members)):
                e = int(e)
                S = sorted(members[e])
                add_theme(theme.pop(e), -1)
                w = theme_weights()
                with np.errstate(divide="ignore"):
                    lw = np.log(kernel.push(w, rho)) + model.marginal_all(S)
                other = n_e > 0
                other[e] = False
                lw[other] = -np.inf
                top = lw.max()
                if not np.isfinite(top):
                    new = e
                else:
                    p = np.exp(lw - top)
                    new = int(rng.choice(C, p=p / p.sum()))
                if new != e:
                    z[S] = new
                    n_e[new], n_e[e] = len(S), 0
                    members[new] = members.pop(e)
                    Mc.pop(e, None)
                    Mc[new] = model.marginal(new, S)
                t = draw_theme(new)
                theme[new] = t
                add_theme(t, +1)
        # ---- 3. themes alone
        for e in list(members):
            add_theme(theme[e], -1)
            t = draw_theme(e)
            theme[e] = t
            add_theme(t, +1)
        # ---- 4. rho
        if tcfg.update_rho:
            on = sum(1 for e, t in theme.items() if e == t and kernel.has_children[t])
            off = sum(1 for e, t in theme.items() if e != t)
            rho = rng.beta(tcfg.rho_a + on, tcfg.rho_b + off)
        # ---- 5. theta, gamma, gamma0
        used_themes = np.flatnonzero(m > 0)
        if cfg.update_theta:
            theta = grammar.sample_theta(rng, counts[used_themes].sum(axis=0))
            log_p = grammar.log_prob(counts, theta)
            P0 = _p0(log_p)
        B = len(members)
        if cfg.update_gamma:
            gamma = escobar_west_gamma(rng, gamma, B, J, cfg.gamma_a, cfg.gamma_b)
        if tcfg.update_gamma0 and B > 0:
            gamma0 = escobar_west_gamma(rng, gamma0, len(used_themes), B, tcfg.gamma0_a, tcfg.gamma0_b)
        if sweep >= cfg.burn_in and (sweep - cfg.burn_in) % cfg.thin == 0:
            assign[np.arange(J), z] += 1
            theme_counts[np.arange(J), [theme[int(e)] for e in z]] += 1
            Q = kernel.push(theme_weights(), rho) / (B + gamma0)
            pop += (n_e + gamma * Q) / (gamma + J)
            n_forms.append(B)
            n_themes.append(len(used_themes))
            log_liks.append(float(sum(Mc.values())))
            gammas.append(gamma)
            gamma0s.append(gamma0)
            rhos.append(rho)
            for nt in grammar.nonterminals:
                thetas[nt].append(theta[nt])
            kept.append(z.copy())
    nk = max(len(kept), 1)
    return {"assign_probs": assign / nk, "theme_probs": theme_counts / nk, "pop": pop / nk,
            "n_forms": np.array(n_forms), "n_themes": np.array(n_themes), "gamma": np.array(gammas),
            "gamma0": np.array(gamma0s), "rho": np.array(rhos),
            "theta": {nt: np.array(v) for nt, v in thetas.items()}, "z": np.array(kept),
            "log_lik": np.array(log_liks)}


def run_gibbs_themes(L, counts, grammar: Grammar, kernel: EditKernel, cfg: GibbsConfig | None = None,
                     tcfg: ThemeConfig | None = None, model=None, z_init=None) -> ThemeResult:
    cfg = cfg or GibbsConfig()
    tcfg = tcfg or ThemeConfig()
    if cfg.burn_in >= cfg.n_sweeps:
        raise ValueError("burn_in must be smaller than n_sweeps")
    chains = [run_chain_themes(L, counts, grammar, kernel, cfg, tcfg, seed=cfg.seed * 1000 + c + 17,
                               init="argmax" if c == 0 else cfg.init, model=model, z_init=z_init)
              for c in range(cfg.n_chains)]
    cap = np.stack([ch["assign_probs"] for ch in chains])
    return ThemeResult(
        assign_probs=cap.mean(axis=0), chain_assign_probs=cap,
        pop_weight=np.mean([ch["pop"] for ch in chains], axis=0),
        n_forms=np.stack([ch["n_forms"] for ch in chains]),
        gamma=np.stack([ch["gamma"] for ch in chains]),
        theta={nt: np.stack([ch["theta"][nt] for ch in chains]) for nt in grammar.nonterminals},
        z_samples=np.stack([ch["z"] for ch in chains]),
        log_lik=np.stack([ch["log_lik"] for ch in chains]),
        theme_probs=np.mean([ch["theme_probs"] for ch in chains], axis=0),
        n_themes=np.stack([ch["n_themes"] for ch in chains]),
        rho=np.stack([ch["rho"] for ch in chains]),
        gamma0=np.stack([ch["gamma0"] for ch in chains]),
    )
