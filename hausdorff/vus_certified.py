"""Certified (dual-polynomial verified) Sigma bounds per channel and the resulting |Vus| interval, new [-1,1] support."""
import os, sys, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_vus as S
D = os.path.join(os.getcwd(), "RBCQCD") if os.path.isdir(os.path.join(os.getcwd(), "RBCQCD")) else os.path.expanduser("~/latticeQCD/BetterData/RBCQCD")
Bs = float(os.environ.get("BS", S.BS_PLACEHOLDER)); T = int(os.environ.get("TMOM", 18)); Z = float(os.environ.get("ZSIG", 3.0))
c = Bs * S.GAMMA_TAU / (S.PREF * S.SEW)
tab = {}
for ens in ("48", "64"):
    ph = S.PHYS[ens]; f2 = ph["Ina"] ** 2; lam = S.make_grid(ph["M0"], "pm1")
    lo_t = hi_t = lo_g = hi_g = 0.0
    for ch, (gam, J) in S.CHANNELS.items():
        C = S.load_channel(ens, gam, D); mean, err, _ = S.jk_stats(C, T + 1)
        r = S.certified_bounds(mean, err, lam, T, Z, lambda E, J=J: S.kernel_lat(E, J, ph["Ina"]))
        tab[(ens, ch)] = {k: (v * f2 if k.endswith(("cert", "grid")) else v) for k, v in r.items()}
        lo_t += r["lo_cert"] * f2; hi_t += r["hi_cert"] * f2; lo_g += r["lo_grid"] * f2; hi_g += r["hi_grid"] * f2
        print(f"{ens}I {ch}: grid [{r['lo_grid']*f2:.5f},{r['hi_grid']*f2:.5f}]  certified [{r['lo_cert']*f2:.5f},{r['hi_cert']*f2:.5f}]  (lift lo {r['delta_lo']:.1e}, hi {r['delta_hi']:.1e})")
    print(f"{ens}I total  grid [{lo_g:.4f},{hi_g:.4f}]  certified [{lo_t:.4f},{hi_t:.4f}] -> |Vus| in [{np.sqrt(c/hi_t):.4f}, {np.sqrt(c/lo_t):.4f}]  (B_s={Bs}, T={T}, z={Z})\n")
    tab[(ens, "tot")] = dict(lo=lo_t, hi=hi_t, vlo=float(np.sqrt(c / hi_t)), vhi=float(np.sqrt(c / lo_t)))
json.dump({f"{k[0]}_{k[1]}": v for k, v in tab.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vus_certified.json"), "w"), indent=1, default=float)
