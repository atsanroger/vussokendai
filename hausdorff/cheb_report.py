import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_vus as S
D = os.path.join(os.getcwd(), "RBCQCD")
NMAX = 12
print("<T*_n> (old, [0,1])  vs  <T_n> (new, [-1,1]) of the normalised measure, s_k = C(k+1)/C(1); |value|<=1 required.")
print("'OUT' = |value| - 1 > 1 sigma ; value(err) shown; per channel and ensemble\n")
for ens in ("48", "64"):
    for ch, (gam, J) in S.CHANNELS.items():
        C = S.load_channel(ens, gam, D)
        (old, new), (eo, en) = S.cheb_diag(C, NMAX)
        rows = []
        for tag, v, e in (("old [0,1] ", old, eo), ("new [-1,1]", new, en)):
            first = next((j + 1 for j in range(NMAX) if abs(v[j]) - 1 > e[j]), None)
            nout = sum(1 for j in range(NMAX) if abs(v[j]) - 1 > e[j])
            maxabs = max(abs(v))
            rows.append(f"   {tag}: first OUT at n={first}, #OUT(n<=12)={nout}, max|value|={maxabs:.2f}")
        print(f"{ens}I {ch}")
        print("\n".join(rows))
        print("   n:          " + " ".join(f"{j:>7d}" for j in range(1, NMAX + 1)))
        print("   old value:  " + " ".join(f"{x:>7.2f}" for x in old))
        print("   new value:  " + " ".join(f"{x:>7.2f}" for x in new))
        print("   old error:  " + " ".join(f"{x:>7.2f}" for x in eo))
        print("   new error:  " + " ".join(f"{x:>7.2f}" for x in en))
