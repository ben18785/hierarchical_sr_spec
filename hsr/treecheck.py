"""Properness checks and forward sampling for top-down tree priors.

A prior is given by ``node_dist(pos) -> (cached, p_fresh, sym)``: the
probability of reusing each cached whole subtree at that position (an
adaptor; empty for a symbol-level model), the probability of a fresh draw,
and the distribution of a fresh node's symbol over hsr.treecode.V (which
may sum to less than one, e.g. when a model puts mass on symbols outside V).
``key(pos)`` must identify positions whose node_dist is the same.

termination   P(the generated tree is finite): the minimal fixed point of
              the branching-process equations (iterated from zero), so mass
              lost to infinite trees (and to symbols outside V) shows up
              as 1 - P.
sample_tree   forward sampling (None when a size cap is hit).
"""
from __future__ import annotations

import numpy as np

from . import treecode as tc

NV = len(tc.V)
VI = {s: i for i, s in enumerate(tc.V)}


def reachable(key, max_keys=200000):
    """{key: representative position} reachable from the root."""
    seen = {key(tc.ROOT): tc.ROOT}
    todo = [tc.ROOT]
    while todo:
        pos = todo.pop()
        for s in tc.UNARY + tc.BINARY:
            kids = [tc.child_pos(pos, s, 0)]
            if s in tc.BINARY:
                kids += [tc.child_pos(pos, s, 1, l) for l in tc.V]
            for c in kids:
                k = key(c)
                if k not in seen:
                    seen[k] = c
                    todo.append(c)
                    if len(seen) > max_keys:
                        raise RuntimeError("too many positions")
    return seen


def termination(node_dist, key, tol=1e-12, max_iter=5000):
    """P(finite tree) and the per-position symbol sums (for normalisation checks)."""
    pos_of = reachable(key)
    keys = list(pos_of)
    ix = {k: i for i, k in enumerate(keys)}
    dist = [node_dist(pos_of[k]) for k in keys]
    sym = np.array([d[2] for d in dist])
    fresh = np.array([d[1] for d in dist])
    cached = np.zeros((len(keys), NV))
    for i, d in enumerate(dist):
        for T, p in d[0]:
            cached[i, VI[T[0]]] += p
    # child indices
    c0 = np.full((len(keys), NV), -1)
    c1 = np.full((len(keys), NV, NV), -1)
    for k, pos in pos_of.items():
        i = ix[k]
        for s in tc.UNARY + tc.BINARY:
            c0[i, VI[s]] = ix[key(tc.child_pos(pos, s, 0))]
            if s in tc.BINARY:
                for l in tc.V:
                    c1[i, VI[s], VI[l]] = ix[key(tc.child_pos(pos, s, 1, l))]
    leaf = np.array([tc.ARITY[s] == 0 for s in tc.V])
    un = [VI[s] for s in tc.UNARY]
    bi = [VI[s] for s in tc.BINARY]
    Q = np.zeros((len(keys), NV))          # Q[k, s] = P(node at k has root s and is finite)
    for it in range(max_iter):
        tot = Q.sum(1)
        fin = np.zeros((len(keys), NV))
        fin[:, leaf] = 1.0
        fin[:, un] = tot[c0[:, un]]
        for s in bi:
            left = Q[c0[:, s]]                           # (K, NV): left child root l, finite
            right = tot[c1[:, s, :]]                     # (K, NV): right child finite given l
            fin[:, s] = (left * right).sum(1)
        Qn = cached + fresh[:, None] * sym * fin
        if np.max(np.abs(Qn - Q)) < tol:
            Q = Qn
            break
        Q = Qn
    return {"p_finite": float(Q[ix[key(tc.ROOT)]].sum()), "iterations": it + 1, "n_positions": len(keys),
            "sym_sum_min": float(sym.sum(1).min()), "sym_sum_max": float(sym.sum(1).max()),
            "total_min": float((cached.sum(1) + fresh * sym.sum(1)).min()),
            "total_max": float((cached.sum(1) + fresh * sym.sum(1)).max())}


def sample_tree(node_dist, rng, cap=200):
    """One tree drawn from the prior; None if it exceeds ``cap`` nodes or
    draws a symbol outside V."""
    count = [0]

    class _Stop(Exception):
        pass

    def draw(pos):
        cached, fresh, sym = node_dist(pos)
        if cached:
            u = rng.random()
            acc = 0.0
            for T, p in cached:
                acc += p
                if u < acc:
                    count[0] += tc.size(T)
                    if count[0] > cap:
                        raise _Stop
                    return T
        tot = sym.sum()
        if rng.random() > tot:
            raise _Stop                  # mass outside V
        s = tc.V[int(rng.choice(NV, p=sym / tot))]
        count[0] += 1
        if count[0] > cap:
            raise _Stop
        if tc.ARITY[s] == 0:
            return (s,)
        left = draw(tc.child_pos(pos, s, 0))
        if tc.ARITY[s] == 1:
            return (s, left)
        return (s, left, draw(tc.child_pos(pos, s, 1, left[0])))

    try:
        return draw(tc.ROOT)
    except (_Stop, RecursionError):
        return None
