#!/usr/bin/env python3
"""
Written 2026-10-21 as a proposal; run for the first time 2026-10-23 (see RESULTS.md).

Hausdorff-moment positivity test of lattice two-point functions:
is C(t_s + k) the moment sequence of a positive measure on [-1, 1]
(hypothesis for M5 > 1), or on [0, 1] (the usual Chebyshev assumption)?

Criterion (Schmuedgen, arXiv:2008.12698, Thm 3.1, even case m = 2n):
  s_0..s_2n is a truncated [a,b] moment sequence  <=>
    H0   = (s_{i+j})_{i,j=0..n}                              >= 0   and
    Hloc = ((a+b) s_{i+j+1} - a b s_{i+j} - s_{i+j+2})_{0..n-1} >= 0
  [-1,1]:  Hloc = s_{i+j} - s_{i+j+2}
  [ 0,1]:  Hloc = s_{i+j+1} - s_{i+j+2}

Test both parities of t_s: lambda^{2m} mu stays positive, lambda^{odd} mu does not.
With link-reflection positivity only, the positive measure sits in C(1 + k).

Data layout (from BetterData/ConfigCreator.py; CHECK before use):
  corr_Z2_l48t96_...  : rows = Ncfg(88) * Nsrc(48) * 96,  cols: t g g 0 0 0 Re Im
  corr_Z2_l64t128_... : rows = Ncfg(40) * Nsrc(64) * 128
"""
import argparse
import os
import numpy as np

ENS = {
    "48": dict(stem="corr_Z2_l48t96_Ls12_m0.00078_LL_m0.0362_LL_{g}_{g}", ncfg=88, nsrc=48, nt=96),
    "64": dict(stem="corr_Z2_l64t128_Ls12_m0.000678_LL_m0.02661_LL_{g}_{g}", ncfg=40, nsrc=64, nt=128),
}
CHANNELS = {"V1": [1, 2, 4], "A1": [14, 13, 11], "V0": [8], "A0": [7]}


def load(path, ncfg, nsrc, nt):
    re = np.loadtxt(path, usecols=(6,))
    return re.reshape(ncfg, nsrc, nt).mean(axis=1)  # (ncfg, nt)


def jackknife(samples):
    n = samples.shape[0]
    tot = samples.sum(axis=0)
    return (tot[None, ...] - samples) / (n - 1)


def hankels(s, n, interval):
    H0 = np.array([[s[i + j] for j in range(n + 1)] for i in range(n + 1)])
    if interval == "-11":
        loc = lambda k: s[k] - s[k + 2]
    elif interval == "01":
        loc = lambda k: s[k + 1] - s[k + 2]
    else:
        raise ValueError(interval)
    H1 = np.array([[loc(i + j) for j in range(n)] for i in range(n)])
    return H0, H1


def unit_diag(H):
    d = np.sqrt(np.abs(np.diag(H)))
    d[d == 0] = 1.0
    return H / np.outer(d, d)


def ldl_pivots(H):
    """Pivots of unpivoted LDL^T; PSD (and nonsingular) <=> all pivots > 0."""
    A = H.astype(float).copy()
    m = A.shape[0]
    piv = np.empty(m)
    for k in range(m):
        piv[k] = A[k, k]
        if piv[k] == 0:
            piv[k:] = np.nan
            break
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:]) / piv[k]
    return piv


def stats(C, ts, n, interval):
    """C: (ncfg, nt) mean-sign-fixed correlator. Returns (min-eig, pivots) mean and jk error for both matrices."""
    def measure(c):
        s = c[ts:ts + 2 * n + 1] / c[ts]
        out = []
        for H in hankels(s, n, interval):
            Hn = unit_diag(H)
            out.append(np.concatenate([[np.linalg.eigvalsh(Hn).min()], ldl_pivots(Hn)]))
        return out

    central = measure(C.mean(axis=0))
    jk = [measure(c) for c in jackknife(C)]
    nj = len(jk)
    res = []
    for which in range(2):
        arr = np.array([j[which] for j in jk])
        err = np.sqrt((nj - 1) / nj * ((arr - arr.mean(axis=0)) ** 2).sum(axis=0))
        res.append((central[which], err))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.getcwd(), "RBCQCD"))
    ap.add_argument("--ens", choices=ENS, default="48")
    ap.add_argument("--chan", choices=CHANNELS, default="V1")
    ap.add_argument("--tmax-start", type=int, default=8)
    ap.add_argument("--nmax", type=int, default=4)
    a = ap.parse_args()
    e = ENS[a.ens]

    C = np.mean([load(os.path.join(a.dir, e["stem"].format(g=g)), e["ncfg"], e["nsrc"], e["nt"])
                 for g in CHANNELS[a.chan]], axis=0)
    sign = np.sign(C.mean(axis=0)[e["nt"] // 8])  # fix sign so the plateau region is positive
    C = sign * C
    # NOTE: around-the-world terms (lambda^{Nt-t}) are ignored; keep ts + 2n << Nt/2.

    for interval in ("-11", "01"):
        print(f"### support [{'-1' if interval == '-11' else '0'},1]   ens={a.ens} chan={a.chan}")
        for ts in range(1, a.tmax_start + 1):
            for n in range(1, a.nmax + 1):
                (m0, e0), (m1, e1) = stats(C, ts, n, interval)
                print(f"ts={ts} ({'odd' if ts % 2 else 'even'}) n={n}  "
                      f"H0 mineig={m0[0]:+.3e}({e0[0]:.1e})  "
                      f"Hloc mineig={m1[0]:+.3e}({e1[0]:.1e})  "
                      f"Hloc pivots={np.array2string(m1[1:], precision=2)} +- {np.array2string(e1[1:], precision=1)}")


if __name__ == "__main__":
    main()
