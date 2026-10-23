import os, sys, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_vus as S
D = os.path.join(os.getcwd(), "RBCQCD")
TS = [4, 6, 8, 10, 12, 14]
Z = 3.0
res = {}
for ens in ("48", "64"):
    ph = S.PHYS[ens]
    print(f"\n===== {ens}I   (Emin = M0 = {ph['M0']}, a^-1 = {ph['Ina']})   bounds on Sigma_phys [GeV^2] at z={Z}; z* = minimal feasible z")
    grids = {sup: S.make_grid(ph["M0"], sup) for sup in ("pos", "pm1")}
    for ch, (gam, J) in S.CHANNELS.items():
        C = S.load_channel(ens, gam, D)
        mean, err, _ = S.jk_stats(C, max(TS) + 1)
        print(f"-- {ens}I {ch} (J={J})")
        print("   T   |   z*(old [0,1])  z*(new [-1,1]) |  old [0,1] bounds          |  new [-1,1] bounds")
        for T in TS:
            line = f"   {T:2d}  |"
            zs = []
            bs = []
            for sup in ("pos", "pm1"):
                lam = grids[sup]
                kap = np.zeros_like(lam); p = lam > 0
                kap[p] = S.kernel_lat(-np.log(lam[p]), J, ph["Ina"])
                zs.append(S.feasibility_zstar(mean, err, lam, T))
                lo, hi = S.bounds(mean, err, lam, T, Z, kap)
                bs.append((lo * ph["Ina"] ** 2, hi * ph["Ina"] ** 2))
                res[(ens, ch, T, sup)] = (zs[-1], bs[-1])
            f = lambda b: ("   infeasible            " if np.isnan(b[0]) else f"[{b[0]:.5f}, {b[1]:.5f}]")
            print(f"{line}    {zs[0]:7.2f}        {zs[1]:7.2f}     | {f(bs[0])}  | {f(bs[1])}")
