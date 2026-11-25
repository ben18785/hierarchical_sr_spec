"""Diagnostics of the gap between seen and novel test forms, shared by the
hold-out report (workflow/scripts/holdout_report.py) and the temporal report
(workflow/scripts/temporal_steps.py report).

Novel test forms (generalised keys absent from the training data) are larger
than seen ones, and every prior pays per node, so part of the seen/novel gap is
size. Per prior, over model-form pairs:

  by_size    mean log P of seen and novel pairs in node-count bins (SIZE_BINS)
  per_node / novel_penalty_fixed_size
             OLS fit log P = a + b * nodes + g * novel; b is the per-node
             slope, g the novelty penalty at fixed size
  vs_sym0    log P minus sym0's log P (sym0, the tree prior without context,
             depends only on size and symbol use), seen / novel
  redundant  novel forms whose constants are not identifiable (``redundant``:
             the Jacobian in the constants is rank deficient, e.g.
             c0*x1 + c1*x1), the signature of a form the canonicaliser failed
             to simplify; ``raw_novel_nonredundant`` is the novel mean without
             them
"""
import numpy as np

SIZE_BINS = [(1, 5), (6, 9), (10, 14), (15, 20), (21, 10_000)]


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def mean(a):
    """Mean of the finite entries of ``a`` (None if there are none)."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None


def ols_novel(y, nodes, novel):
    """(per-node slope, novelty penalty at fixed size) from
    y = a + b * nodes + g * novel over finite y; (None, None) if either group
    is empty or there are fewer than four points."""
    y, nodes, novel = np.asarray(y, float), np.asarray(nodes, float), np.asarray(novel, bool)
    ok = np.isfinite(y) & np.isfinite(nodes)
    if ok.sum() < 4 or novel[ok].all() or not novel[ok].any():
        return None, None
    X = np.column_stack([np.ones(ok.sum()), nodes[ok], novel[ok].astype(float)])
    b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
    return float(b[1]), float(b[2])


def redundant(key, n_x=40, n_draws=3, seed=0, tol=1e-8):
    """True if the constants of form ``key`` are not identifiable: the
    Jacobian of f(x; c) with respect to c is rank deficient at every one of
    ``n_draws`` random constant vectors (generic points), on a grid of x."""
    import sympy as sp
    from hsr.candidates import skeleton_from_key
    from hsr.canonical import consts_of
    try:
        expr = skeleton_from_key(key).expr
    except Exception:
        return False
    cs = sorted(consts_of(expr), key=str)
    if len(cs) < 2:
        return False
    x = [v for v in expr.free_symbols if str(v) == "x1"]
    if not x:
        return False
    try:
        J = sp.lambdify([x[0]] + cs, [sp.diff(expr, c) for c in cs], "numpy")
    except Exception:
        return False
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.2, 3.0, n_x)
    deficient = 0
    with np.errstate(all="ignore"):
        for _ in range(n_draws):
            cv = rng.uniform(0.5, 2.0, len(cs))
            try:
                M = np.array([np.broadcast_to(np.asarray(col, float), xs.shape) for col in J(xs, *cv)]).T
            except Exception:
                return False
            M = M[np.all(np.isfinite(M), axis=1)]
            if M.shape[0] < len(cs):
                return False
            sv = np.linalg.svd(M / (np.linalg.norm(M, axis=0) + 1e-300), compute_uv=False)
            deficient += sv[-1] < tol * sv[0]
    return bool(deficient == n_draws)


class RedundancyCache:
    """``redundant`` memoised by key (the same novel forms recur across runs)."""

    def __init__(self):
        self.memo = {}

    def __call__(self, key):
        if key not in self.memo:
            self.memo[key] = redundant(key)
        return self.memo[key]


def gap_stats(raw, nodes, seen, red=None, sym0=None, by_size=True):
    """Seen/novel gap diagnostics of one prior on one set of model-form pairs.

    raw    log P per pair (-inf for zero probability; left out of means)
    nodes  node count per pair (nan if unknown)
    seen   bool per pair (form in the training data)
    red    bool per pair (redundant constants), optional
    sym0   sym0's log P per pair, optional
    """
    raw, nodes, seen = np.asarray(raw, float), np.asarray(nodes, float), np.asarray(seen, bool)
    red = np.zeros_like(seen) if red is None else np.asarray(red, bool)
    slope, pen = ols_novel(raw, nodes, ~seen)
    o = {"raw_seen": mean(raw[seen]), "raw_novel": mean(raw[~seen]),
         "per_node": slope, "novel_penalty_fixed_size": pen,
         "raw_novel_nonredundant": mean(raw[~seen & ~red])}
    o["gap"] = None if o["raw_seen"] is None or o["raw_novel"] is None else o["raw_seen"] - o["raw_novel"]
    o["nodes_seen"], o["nodes_novel"] = mean(nodes[seen]), mean(nodes[~seen])
    if sym0 is not None:
        ex = raw - np.asarray(sym0, float)
        o["vs_sym0_seen"], o["vs_sym0_novel"] = mean(ex[seen]), mean(ex[~seen])
    if by_size:
        o["by_size"] = {}
        for lo, hi in SIZE_BINS:
            m = (nodes >= lo) & (nodes <= hi)
            o["by_size"][f"{lo}-{hi}"] = {"n_seen": int((m & seen).sum()), "n_novel": int((m & ~seen).sum()),
                                         "seen": mean(raw[m & seen]), "novel": mean(raw[m & ~seen])}
    return o


def by_size_table(rows):
    """Markdown lines: ``rows`` is [(prior, by_size dict)]; the first row's
    counts give the pairs line (counts do not depend on the prior)."""
    bins = list(rows[0][1])
    L = ["| prior | " + " | ".join(f"{b} nodes" for b in bins) + " |", "|---|" + "---|" * len(bins)]
    bs = rows[0][1]
    L.append("| pairs (seen / novel) | " + " | ".join(f"{bs[b]['n_seen']} / {bs[b]['n_novel']}" for b in bins) + " |")
    for p, bs in rows:
        L.append(f"| {p} | " + " | ".join(f"{fmt(bs[b]['seen'])} / {fmt(bs[b]['novel'])}" for b in bins) + " |")
    return L


FIXED_SIZE_NOTE = ("per node: OLS slope of log P on node count; novel penalty: OLS coefficient of the novel "
                   "indicator at fixed node count (log P = a + b nodes + g novel, over model-form pairs); vs sym0: "
                   "log P minus sym0's log P, seen / novel; novel (non-redundant): mean log P of novel pairs whose "
                   "constants are identifiable.")


def fixed_size_table(rows, label="weighting"):
    """Markdown lines: ``rows`` is [(prior, label value, gap_stats dict)]."""
    L = [f"| prior | {label} | seen | novel | gap | per node | novel penalty | vs sym0 seen / novel | "
         "novel (non-redundant) |", "|---|---|---|---|---|---|---|---|---|"]
    for p, w, a in rows:
        gap = None if a["raw_seen"] is None or a["raw_novel"] is None else a["raw_seen"] - a["raw_novel"]
        L.append(f"| {p} | {w} | {fmt(a['raw_seen'])} | {fmt(a['raw_novel'])} | {fmt(gap)} | "
                 f"{fmt(a['per_node'])} | {fmt(a['novel_penalty_fixed_size'])} | {fmt(a.get('vs_sym0_seen'))} / "
                 f"{fmt(a.get('vs_sym0_novel'))} | {fmt(a['raw_novel_nonredundant'])} |")
    return L
