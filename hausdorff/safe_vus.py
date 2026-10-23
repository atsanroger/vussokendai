#!/usr/bin/env python3
"""
Certified bounds on the inclusive tau -> X_s nu kernel integral from RBC/UKQCD u-sbar correlators,
and a comparison of the old [0,1] Chebyshev assumption with the [-1,1] one.

Model.  With lambda = exp(-E) and a measure sigma >= 0:
    C(t) = int dsigma(lambda) lambda^(t-1),   t = 1, 2, ...        (C(0) is dropped: contact terms)
 old assumption   : supp sigma in (0, exp(-E_min)]            (all states physical, lambda > 0)
 new assumption   : supp sigma in [-1, exp(-E_min)]            (lambda < 0 allowed, kernel = 0 there)
 E_min = lightest u-sbar state = kaon mass M0 (lattice units).
The "link-RP only" form lambda^(t-1) is the one for which odd start times pass (hankel_scan.py).

Inclusive integral per channel:  Sigma_lat = int dsigma(lambda) kappa_J(lambda)/lambda,
    kappa_J = (2/E) K_J(y),  y = (a^-1 E)^2 / m_tau^2,  K_0 = (1-y)^2,  K_1 = (1-y)^2 (1+2y)  (y < 1, else 0)
    Sigma_phys = (a^-1)^2 Sigma_lat   [GeV^2];   C_ren = Z^2 C_file / L^3.
Calibration (checked by test_mock): a single state C = (m f^2/2) e^(-m t) gives Sigma_phys = f_phys^2 K(m_phys^2).

Bounds: LP on a fine lambda grid (the dual is the SOS/SDP certificate), each moment constraint
|sum - C(t)| <= z sigma_t.  Minimal feasible z ("z*") is reported: z* >> 3 means the data are inconsistent
with the assumed support at that number of moments.
"""
import os
import sys
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hankel_test_proposal import ENS as _ENS  # noqa: E402

MTAU = 1.77686          # GeV
GF = 1.1663787e-5       # GeV^-2
SEW = 1.0201
GAMMA_TAU = 2.2670e-12  # GeV, hbar / tau lifetime
PREF = GF ** 2 * MTAU ** 3 / (16 * np.pi)
# Inclusive strange branching fraction B_us = 2.908 +- 0.048 %  (HFLAV 2023, arXiv:2411.18639, Sec. 11.6.1, eq. 246;
# table value B(tau -> X_s nu) = 0.0291(5), unitarity-constrained 0.0292(4)).
BS = 0.02908
BS_ERR = 0.00048

PHYS = {
    "48": dict(Ina=1.7295, Z=0.71076, M0=0.28816, L=48),
    "64": dict(Ina=2.3586, Z=0.74293, M0=0.21539, L=64),
}
# channel -> (gamma list, J)   [A_T, V_T carry J=0; A_S, V_S carry J=1]
CHANNELS = {"AT": ([7], 0), "VT": ([8], 0), "AS": ([11, 13, 14], 1), "VS": ([1, 2, 4], 1)}


def kernel_lat(E, J, Ina):
    y = (Ina * E) ** 2 / MTAU ** 2
    K = np.where(y < 1, (1 - y) ** 2 * ((1 + 2 * y) if J == 1 else 1.0), 0.0)
    return 2.0 / E * K


def load_channel(ens, gam, d):
    e = _ENS[ens]
    cs = []
    for g in gam:
        p = os.path.join(d, e["stem"].format(g=g))
        cs.append(np.loadtxt(p, usecols=(6,)).reshape(e["ncfg"], e["nsrc"], e["nt"]).mean(axis=1))
    C = np.mean(cs, axis=0)
    C = np.sign(C.mean(axis=0)[e["nt"] // 8]) * C
    ph = PHYS[ens]
    return C * ph["Z"] ** 2 / ph["L"] ** 3          # renormalised, lattice units, shape (ncfg, nt)


def jk_stats(C, tmax):
    n = C.shape[0]
    jk = (C.sum(axis=0)[None, :] - C) / (n - 1)
    mean = C.mean(axis=0)
    err = np.sqrt((n - 1) / n * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
    return mean[: tmax + 1], err[: tmax + 1], jk[:, : tmax + 1]


def make_grid(Emin, support, nE=3000, Emax=9.0, nneg=600):
    E = np.concatenate([np.linspace(Emin, 3.0, nE), np.linspace(3.0, Emax, nE // 3)[1:], [14.0, 40.0]])
    lam = np.concatenate([np.exp(-E), [0.0]])       # lambda = 0 (E = infinity)
    if support == "pm1":
        neg = -np.concatenate([np.linspace(1.0, 0.05, nneg), np.geomspace(0.05, 1e-4, nneg // 2)[1:]])
        lam = np.concatenate([lam, neg])
    return lam


def moment_matrix(lam, T):
    # row k = t-1: lambda^(t-1), t = 1..T ; 0**0 = 1
    return np.vstack([np.power(lam, k) if k else np.ones_like(lam) for k in range(T)])


def _solve(c, A, b, bnd):
    for m in ("highs", "highs-ipm", "highs-ds"):
        r = linprog(c, A_ub=A, b_ub=b, bounds=bnd, method=m)
        if r.status == 0:
            return r
    return r


def feasibility_zstar(C, sig, lam, T):
    """Smallest z such that a positive measure on lam reproduces C(1..T) within z sigma."""
    c1 = C[1]
    M = moment_matrix(lam, T) * c1                  # variables u = w / C(1)
    A = np.hstack([M / sig[1:T + 1, None], -np.ones((T, 1))])
    A2 = np.hstack([-M / sig[1:T + 1, None], -np.ones((T, 1))])
    b = C[1:T + 1] / sig[1:T + 1]
    res = _solve(np.r_[np.zeros(len(lam)), 1.0], np.vstack([A, A2]), np.r_[b, -b],
                 [(0, None)] * len(lam) + [(0, None)])
    return res.x[-1] if res.status == 0 else np.nan


def bounds(C, sig, lam, T, z, kappa):
    """(lower, upper) of sum_i w_i kappa_i / lambda_i subject to the moment boxes; (nan, nan) if infeasible."""
    c1 = C[1]
    obj = np.zeros_like(lam)
    pos = lam > 0
    obj[pos] = kappa[pos] / lam[pos] * c1           # variables u = w / C(1)
    M = moment_matrix(lam, T) * c1
    sg = sig[1:T + 1]
    A = np.vstack([M / sg[:, None], -M / sg[:, None]])
    b = np.r_[(C[1:T + 1] + z * sg) / sg, -(C[1:T + 1] - z * sg) / sg]
    bnd = [(0, None)] * len(lam)
    hi = _solve(-obj, A, b, bnd)
    lo = _solve(obj, A, b, bnd)
    if hi.status != 0 or lo.status != 0:
        return np.nan, np.nan
    return lo.fun, -hi.fun


def certified_bounds(C, sig, lam, T, z, kappa_fn, ndense=200000):
    """Rigorous version of bounds().  The LP duals give a polynomial p(lambda) = sum_k p_k lambda^k, p_k = (y+_k - y-_k)/sigma_k,
    with  p >= g  on the grid (g = kappa/lambda for lambda > 0, 0 for lambda < 0).  If p >= g on the whole continuous
    support, then for every positive measure sigma whose moments m_k = int lambda^k dsigma lie in the boxes
        int g dsigma <= int p dsigma = sum_k p_k m_k <= sum_k p_k C(k+1) + z sum_k |p_k| sigma_{k+1}.
    p is checked on a dense grid of the support and lifted by the largest violation delta (p_0 += delta).
    The lower bound uses the mirror image (q <= g).  kappa_fn(E) returns kappa for lattice energy E.
    Returns dict(lo_cert, hi_cert, lo_grid, hi_grid, delta_lo, delta_hi)."""
    c1 = C[1]
    g = np.zeros_like(lam); pos = lam > 0
    g[pos] = kappa_fn(-np.log(lam[pos])) / lam[pos]
    M = moment_matrix(lam, T) * c1
    sg = sig[1:T + 1]
    A = np.vstack([M / sg[:, None], -M / sg[:, None]])
    b = np.r_[(C[1:T + 1] + z * sg) / sg, -(C[1:T + 1] - z * sg) / sg]
    bnd = [(0, None)] * len(lam)
    lp = lam[pos]
    parts = [np.linspace(lp.min(), lp.max(), ndense), [0.0]]
    if lam.min() < 0:
        parts.append(np.linspace(-1.0, -1e-6, ndense // 4))
    lamd = np.concatenate(parts)
    gd = np.zeros_like(lamd)
    pd = lamd > 0
    gd[pd] = kappa_fn(-np.log(lamd[pd])) / lamd[pd]
    res = {}
    for name, sign in (("hi", 1.0), ("lo", -1.0)):
        r = _solve(-sign * g * c1, A, b, bnd)           # hi: min -g.u ; lo: min +g.u
        if r.status != 0:
            return None
        y = -r.ineqlin.marginals                         # >= 0, one per constraint row
        pk = (y[:T] - y[T:]) / sg
        if sign > 0:
            poly = pk.copy()
            delta = max(0.0, float(np.max(gd - np.polyval(poly[::-1], lamd))))
            poly[0] += delta
            res["hi_cert"] = float(np.dot(poly, C[1:T + 1]) + z * np.dot(np.abs(poly), sg))
            res["hi_grid"] = -r.fun
            res["delta_hi"] = delta
        else:
            poly = -pk
            delta = max(0.0, float(np.max(np.polyval(poly[::-1], lamd) - gd)))
            poly[0] -= delta
            res["lo_cert"] = float(np.dot(poly, C[1:T + 1]) - z * np.dot(np.abs(poly), sg))
            res["lo_grid"] = r.fun
            res["delta_lo"] = delta
    return res


def cheb_coeffs(n, shifted):
    """Monomial coefficients of T_n (or shifted T*_n = T_n(2x-1)) on [-1,1] ([0,1])."""
    t = np.polynomial.chebyshev.cheb2poly([0] * n + [1])
    if not shifted:
        return t
    p = np.polynomial.Polynomial(t)
    return p(np.polynomial.Polynomial([-1, 2])).coef


def cheb_diag(C, nmax=12):
    """<T*_n> (old, support [0,1]) and <T_n> (new, support [-1,1]) of the normalised measure; jackknife errors.
    s_k = C(k+1)/C(1) = <lambda^k>, k = 0..nmax."""
    n = C.shape[0]
    jk = (C.sum(axis=0)[None, :] - C) / (n - 1)

    def vals(c):
        s = c[1:nmax + 2] / c[1]
        return [np.array([np.dot(cheb_coeffs(j, sh)[: len(s)], s[: j + 1]) for j in range(1, nmax + 1)]) for sh in (True, False)]

    cen = vals(C.mean(axis=0))
    arr = [np.array([vals(c)[i] for c in jk]) for i in (0, 1)]
    err = [np.sqrt((n - 1) / n * ((a - a.mean(axis=0)) ** 2).sum(axis=0)) for a in arr]
    return cen, err


def test_mock():
    """Single state; exact moments, tiny errors: the LP interval must contain f^2 K(m)."""
    ens = "48"; ph = PHYS[ens]
    m_phys, f_phys = 0.4937, 0.1564
    m, f = m_phys / ph["Ina"], f_phys / ph["Ina"]
    T = 10
    t = np.arange(T + 2)
    C = m * f ** 2 / 2 * np.exp(-m * t)
    sig = C * 1e-3
    exact = f_phys ** 2 * (1 - (m_phys / MTAU) ** 2) ** 2
    for sup in ("pos", "pm1"):
        lam = make_grid(m * 0.97, sup)
        kap = np.zeros_like(lam); p = lam > 0
        kap[p] = kernel_lat(-np.log(lam[p]), 0, ph["Ina"])
        lo, hi = bounds(C, sig, lam, T, 3.0, kap)
        print(f"mock single kaon, support {sup}: Sigma_phys in [{lo * ph['Ina'] ** 2:.5f}, {hi * ph['Ina'] ** 2:.5f}]  exact {exact:.5f}")


if __name__ == "__main__":
    test_mock()
