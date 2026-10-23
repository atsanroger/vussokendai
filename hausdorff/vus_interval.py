"""Interval for |Vus| = sqrt( B_s Gamma_tau / (Pref S_EW Sigma_tot) ),  Sigma_tot = sum of the four channel sums.
Bounds of independent channels add (each channel has its own measure), which is conservative.
B_s = 0.02908(48) from HFLAV 2023 (see safe_vus.BS); override with BS=... and BS_ERR=..."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_vus as S
D = os.path.join(os.getcwd(), "RBCQCD")
Bs = float(os.environ.get("BS", S.BS)); BsE = float(os.environ.get("BS_ERR", S.BS_ERR))
c = Bs * S.GAMMA_TAU / (S.PREF * S.SEW)
print(f"B_s = {Bs} +- {BsE},  c = B_s Gamma_tau/(Pref SEW) = {c:.4e} GeV^2")
print(f"expected Sigma_tot for Vus=0.2243: {c/0.2243**2:.4f} GeV^2\n")
out = {}
for ens in ("48", "64"):
    ph = S.PHYS[ens]
    for sup in ("pos", "pm1"):
        lam = S.make_grid(ph["M0"], sup)
        for T in (14, 18, 22):
            for z in (3.0, 2.0):
                lo_t = hi_t = 0.0; ok = True; zs = []
                for ch, (gam, J) in S.CHANNELS.items():
                    C = S.load_channel(ens, gam, D); mean, err, _ = S.jk_stats(C, T + 1)
                    kap = np.zeros_like(lam); p = lam > 0; kap[p] = S.kernel_lat(-np.log(lam[p]), J, ph["Ina"])
                    zs.append(S.feasibility_zstar(mean, err, lam, T))
                    lo, hi = S.bounds(mean, err, lam, T, z, kap)
                    if np.isnan(lo): ok = False; break
                    lo_t += lo * ph["Ina"] ** 2; hi_t += hi * ph["Ina"] ** 2
                lab = "new [-1,1]" if sup == "pm1" else "old [0,1] "
                if ok:
                    vlo, vhi = np.sqrt(c / hi_t), np.sqrt(c / lo_t)
                    print(f"{ens}I {lab} T={T} z={z:.0f}: Sigma_tot in [{lo_t:.4f}, {hi_t:.4f}]  ->  |Vus| in [{vlo:.4f}, {vhi:.4f}]   (width {100*(vhi-vlo)/(vhi+vlo):.1f}%)")
                    out[(ens, sup, T, z)] = (vlo, vhi)
                else:
                    print(f"{ens}I {lab} T={T} z={z:.0f}: INFEASIBLE  (z* per channel AT,VT,AS,VS = {np.round(zs,2)})")
print()
for T in (14, 18, 22):
    for z in (3.0, 2.0):
        a, b = out.get(("48", "pm1", T, z)), out.get(("64", "pm1", T, z))
        if a and b:
            lo, hi = max(a[0], b[0]), min(a[1], b[1])
            print(f"T={T} z={z:.0f}: new-method intervals of 48I and 64I {'overlap in [%.4f, %.4f]' % (lo, hi) if lo < hi else 'DO NOT OVERLAP'}")
