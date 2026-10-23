import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_vus as S
# mock single kaon (as test_mock) -> certified interval must contain the exact value
ph = S.PHYS["48"]; m_phys, f_phys = 0.4937, 0.1564
m, f = m_phys / ph["Ina"], f_phys / ph["Ina"]; T = 10
t = np.arange(T + 2); C = m * f ** 2 / 2 * np.exp(-m * t); sig = C * 1e-3
exact = f_phys ** 2 * (1 - (m_phys / S.MTAU) ** 2) ** 2
for sup in ("pos", "pm1"):
    lam = S.make_grid(m * 0.97, sup)
    r = S.certified_bounds(C, sig, lam, T, 3.0, lambda E: S.kernel_lat(E, 0, ph["Ina"]))
    f2 = ph["Ina"] ** 2
    print(f"{sup}: grid [{r['lo_grid']*f2:.5f},{r['hi_grid']*f2:.5f}]  certified [{r['lo_cert']*f2:.5f},{r['hi_cert']*f2:.5f}]  "
          f"lift delta lo/hi = {r['delta_lo']:.2e}/{r['delta_hi']:.2e}   exact {exact:.5f}")
