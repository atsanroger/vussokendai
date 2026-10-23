#!/usr/bin/env python3
"""
Full scan of the Hausdorff positivity test on the RBC/UKQCD 48I and 64I u-sbar correlators.
Builds on hankel_test_proposal.py (same criterion, Schmuedgen arXiv:2008.12698 Thm 3.1/3.2).

For every ensemble / channel (or single gamma) / support / start time ts / order n it reports
the smallest eigenvalue of the unit-diagonal Hankel matrices and its jackknife significance
z = mineig / sigma.  A matrix with z < -ZCUT is called a VIOLATION (excludes the hypothesis);
z > +ZCUT is a clear PASS; otherwise INCONCLUSIVE.  A pass is only compatibility, never a proof.

Row layout of the files is assumed (ncfg, nsrc, nt); the autocorrelation along row order is
at noise level, so the jackknife error does not depend on that assumption (checked 2026-10-23).
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hankel_test_proposal import ENS, CHANNELS, hankels, unit_diag, jackknife  # noqa: E402

ZCUT = 3.0
GAMMAS = [1, 2, 4, 8, 7, 11, 13, 14]


def load_gamma(d, e, g):
    path = os.path.join(d, e["stem"].format(g=g))
    re = np.loadtxt(path, usecols=(6,))
    return re.reshape(e["ncfg"], e["nsrc"], e["nt"]).mean(axis=1)


def mineigs(c, ts, n, interval):
    s = c[ts:ts + 2 * n + 1] / c[ts]
    return [np.linalg.eigvalsh(unit_diag(H)).min() for H in hankels(s, n, interval)]


def zscan(C, ts, n, interval):
    cen = mineigs(C.mean(axis=0), ts, n, interval)
    jk = np.array([mineigs(c, ts, n, interval) for c in jackknife(C)])
    nj = len(jk)
    err = np.sqrt((nj - 1) / nj * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
    return cen, err


def verdict(m, s):
    z = m / s if s > 0 else np.inf * np.sign(m)
    if z < -ZCUT:
        return "VIOL", z
    if z > ZCUT:
        return "pass", z
    return "inc", z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.getcwd(), "RBCQCD"))
    ap.add_argument("--tmax-start", type=int, default=8)
    ap.add_argument("--nmax", type=int, default=3)
    ap.add_argument("--save-rows", action="store_true", help="write all per-test numbers to hankel_scan_rows.npy next to this script")
    ap.add_argument("--ncfg48", type=int, default=None, help="override number of configs (nsrc follows from the row count)")
    ap.add_argument("--ncfg64", type=int, default=None)
    ap.add_argument("--ens", choices=list(ENS), default=None, help="only this ensemble")
    a = ap.parse_args()
    for k, nc in (("48", a.ncfg48), ("64", a.ncfg64)):
        if nc:
            e = ENS[k]
            rows_ = e["ncfg"] * e["nsrc"]
            assert rows_ % nc == 0, f"{rows_} rows of blocks not divisible by {nc}"
            e["nsrc"], e["ncfg"] = rows_ // nc, nc

    labels = list(CHANNELS) + [f"g{g}" for g in GAMMAS]
    rows = []
    for ens, e in ENS.items():
        if a.ens and ens != a.ens:
            continue
        cache = {g: load_gamma(a.dir, e, g) for g in GAMMAS}
        for lab in labels:
            gs = CHANNELS[lab] if lab in CHANNELS else [int(lab[1:])]
            C = np.mean([cache[g] for g in gs], axis=0)
            C = np.sign(C.mean(axis=0)[e["nt"] // 8]) * C
            for interval in ("-11", "01"):
                for ts in range(1, a.tmax_start + 1):
                    for n in range(1, a.nmax + 1):
                        if ts + 2 * n >= e["nt"] // 2:
                            continue
                        cen, err = zscan(C, ts, n, interval)
                        v0, z0 = verdict(cen[0], err[0])
                        v1, z1 = verdict(cen[1], err[1])
                        rows.append((ens, lab, interval, ts, n, v0, z0, v1, z1, cen[0], cen[1]))

    # summary: per ens/chan/support, count verdicts for the localizing matrix, split by ts parity
    print("ens chan  support | Hloc: pass/inc/VIOL  odd-ts | pass/inc/VIOL  even-ts || H0: pass/inc/VIOL")
    out = []
    for ens in [x for x in ENS if not a.ens or x == a.ens]:
        for lab in labels:
            for interval in ("-11", "01"):
                sel = [r for r in rows if r[0] == ens and r[1] == lab and r[2] == interval and r[4] >= 2]
                def cnt(rs, k):
                    return [sum(1 for r in rs if r[k] == v) for v in ("pass", "inc", "VIOL")]
                odd = cnt([r for r in sel if r[3] % 2 == 1], 7)
                even = cnt([r for r in sel if r[3] % 2 == 0], 7)
                h0 = cnt(sel, 5)
                line = f"{ens:>3} {lab:>5}  {'[-1,1]' if interval == '-11' else ' [0,1]'} | {odd[0]:2d}/{odd[1]:2d}/{odd[2]:2d}   | {even[0]:2d}/{even[1]:2d}/{even[2]:2d}   || {h0[0]:2d}/{h0[1]:2d}/{h0[2]:2d}"
                print(line)
                out.append((ens, lab, interval, odd, even, h0))
    if a.save_rows:
        np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hankel_scan_rows.npy"),
                np.array(rows, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
