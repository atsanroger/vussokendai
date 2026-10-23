import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_vus as S
D = os.path.join(os.getcwd(), "RBCQCD")
TS = [10, 14, 18, 22, 26, 30]
for ens in ("48", "64"):
    ph = S.PHYS[ens]; lam = S.make_grid(ph["M0"], "pm1")
    for ch in ("AS", "VS"):
        gam, J = S.CHANNELS[ch]
        C = S.load_channel(ens, gam, D); mean, err, _ = S.jk_stats(C, max(TS) + 1)
        kap = np.zeros_like(lam); p = lam > 0; kap[p] = S.kernel_lat(-np.log(lam[p]), J, ph["Ina"])
        print(f"{ens}I {ch} new [-1,1]   rel.err C(t): t=10 {err[10]/mean[10]:.1e}  t=20 {err[20]/mean[20]:.1e}  t=30 {err[30]/mean[30]:.1e}")
        for T in TS:
            zs = S.feasibility_zstar(mean, err, lam, T)
            row = f"   T={T:2d} z*={zs:5.2f} |"
            for z in (3.0, 2.0, 1.0):
                lo, hi = S.bounds(mean, err, lam, T, z, kap)
                row += f"  z={z:.0f}: [{lo*ph['Ina']**2:.5f},{hi*ph['Ina']**2:.5f}] w={((hi-lo)/(hi+lo)*100):4.1f}%"
            print(row)
