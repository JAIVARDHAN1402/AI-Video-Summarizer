"""Kernel Temporal Segmentation (Potapov et al., ECCV 2014).

Splits a feature sequence into contiguous shots by finding the change points
that minimise within-segment kernel variance. Every SumMe/TVSum evaluation
uses this to turn per-frame importance into shot-level scores before the
knapsack summary, so the F1 numbers stay comparable to published work.

Same objective as Danila Potapov's reference ``cpd_nonlin`` / ``cpd_auto``,
with the change-point dynamic program vectorised over NumPy so a
2 fps-downsampled video (a few hundred frames) segments in well under a second.
"""

from __future__ import annotations

import numpy as np


def gaussian_kernel(x: np.ndarray) -> np.ndarray:
    """RBF Gram matrix with the median-distance heuristic bandwidth."""
    x = x.astype(np.float64)
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    pos = d2[d2 > 0]
    med = np.median(pos) if pos.size else 1.0
    return np.exp(-d2 / (2.0 * med + 1e-12))


def _segment_costs(K: np.ndarray) -> np.ndarray:
    """cost[a, b] = within-segment kernel variance of frames [a, b), for b > a.

        cost(a, b) = sum_t K[t,t]  -  (1/(b-a)) * sum_{i,j in [a,b)} K[i,j]
    """
    n = K.shape[0]
    diag = np.diag(K).astype(np.float64)
    csum_diag = np.concatenate([[0.0], np.cumsum(diag)])          # (n+1,)
    # C[i, j] = sum of K over rows [0, i) x cols [0, j)           # (n+1, n+1)
    row = np.vstack([np.zeros((1, n)), np.cumsum(K, axis=0)])
    C = np.hstack([np.zeros((n + 1, 1)), np.cumsum(row, axis=1)])
    Cdiag = np.diag(C).copy()                                     # C[b, b]

    cost = np.zeros((n + 1, n + 1), dtype=np.float64)
    for a in range(n):
        b = np.arange(a + 1, n + 1)
        cd = csum_diag[b] - csum_diag[a]
        block = Cdiag[b] - C[a, b] - C[b, a] + C[a, a]
        cost[a, b] = cd - block / (b - a)
    return cost


def _dp(cost: np.ndarray, n: int, max_ncp: int, lmin: int) -> tuple[np.ndarray, np.ndarray]:
    """Change-point DP up to ``max_ncp`` splits.

    ``I[k, l]`` is the best cost of splitting the first ``l`` frames into ``k``
    segments, so one pass yields the optimum for *every* k — there is no reason
    to re-run it per candidate count.
    """
    INF = 1e18
    I = np.full((max_ncp + 1, n + 1), INF)
    I[0, 0] = 0.0
    prev = np.zeros((max_ncp + 1, n + 1), dtype=np.int64)

    for k in range(1, max_ncp + 1):
        for l in range(k * lmin, n + 1):
            lo, hi = (k - 1) * lmin, l - lmin + 1
            if hi <= lo:
                continue
            cand = I[k - 1, lo:hi] + cost[lo:hi, l]
            m = int(np.argmin(cand))
            if cand[m] < I[k, l]:
                I[k, l] = cand[m]
                prev[k, l] = lo + m
    return I, prev


def _backtrack(prev: np.ndarray, n: int, ncp: int) -> np.ndarray:
    cps: list[int] = []
    cur = n
    for k in range(ncp, 0, -1):
        t = int(prev[k, cur])
        if t > 0:
            cps.append(t)
        cur = t
    return np.array(sorted(cps), dtype=np.int64)


def cpd_nonlin(K: np.ndarray, ncp: int, lmin: int = 1) -> tuple[np.ndarray, float]:
    """Best ``ncp`` change points for kernel matrix ``K``; returns (cps, cost)."""
    n = K.shape[0]
    I, prev = _dp(_segment_costs(K), n, ncp, lmin)
    return _backtrack(prev, n, ncp), float(I[ncp, n])


def cpd_auto(
    features: np.ndarray, max_ncp: int, vmax: float = 1.0, lmin: int = 1,
) -> np.ndarray:
    """Pick the number of change points automatically (penalised objective).

    Builds the kernel and segment costs once and runs the DP once; the
    penalised score for every candidate count is then just a read off the
    table.
    """
    n = features.shape[0]
    if n < 3:
        return np.array([], dtype=np.int64)
    max_ncp = int(min(max_ncp, n - 1))
    if max_ncp < 1:
        return np.array([], dtype=np.int64)

    K = gaussian_kernel(features)
    I, prev = _dp(_segment_costs(K), n, max_ncp, lmin)

    ncps = np.arange(1, max_ncp + 1, dtype=np.float64)
    costs = I[1:, n]
    penalty = (ncps / (2.0 * n)) * (np.log(np.maximum(n / ncps, 1.0)) + 1.0)
    scores = np.where(costs >= 1e17, np.inf, costs / n + vmax * penalty)
    best_ncp = int(np.argmin(scores)) + 1
    return _backtrack(prev, n, best_ncp)


def segment_bounds(n_frames: int, change_points: np.ndarray) -> list[tuple[int, int]]:
    """Change-point indices -> [start, end) shot ranges covering every frame."""
    edges = sorted(set([0, *[int(c) for c in change_points if 0 < c < n_frames], n_frames]))
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
