# Certified bounds on the inclusive tau -> X_s nu rate, and old vs new method (2026-10-23)

Code: `safe_vus.py` (library), `cheb_report.py`, `bounds_report.py`, `bounds_T.py`, `vus_interval.py`.
Data: `RBCQCD/corr_Z2_*`, all 8 gamma channels, combined as A_T (g7), V_T (g8), A_S (g11,13,14), V_S (g1,2,4).
Run from the directory that contains `RBCQCD/`.

## Setup (what is assumed)
- C(t) = int dsigma(lambda) lambda^(t-1), t >= 1, lambda = exp(-E), sigma >= 0 (the odd-start-time form that passed
  `hankel_scan.py`). C(0) is dropped (contact terms).
- **old**: supp sigma in (0, exp(-M0)]. **new**: supp sigma in [-1, exp(-M0)], kernel = 0 on lambda < 0.
- E >= M0: the kaon is the lightest u-sbar state (M0 = 0.28816 on 48I, 0.21539 on 64I, from the TauDecay.pdf table).
- Kernel (no regulator needed once E >= M0): K_0 = (1-y)^2, K_1 = (1-y)^2 (1+2y), y = (a^-1 E)^2 / m_tau^2;
  Sigma_phys = (a^-1)^2 int dsigma (2/E) K_J / lambda;  C_ren = Z^2 C_file / L^3, Z_A = Z_V from the table.
- Bounds: LP on a fine lambda grid (its dual is the SOS/SDP certificate); every moment is boxed to +-z sigma_jackknife
  (Bonferroni, covariance not used).  z* = smallest z for which the LP is feasible.
- Calibration: a single state C = (m f^2/2) e^(-m t) with f_K = 0.1564 GeV gives [0.02075, 0.02089] around the exact
  f^2 (1 - m^2/m_tau^2)^2 = 0.02083 (`python safe_vus.py`).
- |Vus| = sqrt( B_s Gamma_tau / (Pref S_EW Sigma_tot) ), Pref = G_F^2 m_tau^3 / (16 pi) (eq. 44 of TauDecay.pdf).
  B_s = 0.02908(48) is B_us of HFLAV 2023 (arXiv:2411.18639, Sec. 11.6.1, eq. 246; the fit table lists B(tau -> X_s nu) = 0.0291(5),
  0.0292(4) unitarity-constrained). |Vus| scales as sqrt(B_s); override with `BS=... BS_ERR=... python vus_interval.py`.

## 1. Do the Chebyshev coefficients leave [-1, 1]?  (`cheb_report.py`, n <= 12)
`<T*_n>` (old, shifted, support [0,1]) vs `<T_n>` (new, support [-1,1]) of the normalised measure s_k = C(k+1)/C(1).

| | old [0,1] | new [-1,1] |
|---|---|---|
| 48I, 64I, all of A_T V_T A_S V_S | first |value| > 1 at n = 5..8, significantly (e.g. 48I A_T n=6: 2.16 +- 0.04); up to 10^2..10^3 at n = 12 | never (max |value| = 0.98, error <= 0.02) |

Remark: the new values follow cos(n pi/2) because the measure sits near lambda = 0; staying inside +-1 is therefore a
necessary and weak test. The LP below is the stronger one.

## 2. Feasibility and bounds (`bounds_report.py`, z = 3)
- **J = 0 channels (A_T, V_T): the old support is infeasible at 3 sigma.** z*(old) = 3.3 (A_T) and 4.8-5.2 (V_T) on 48I,
  9.1 and 18.7 on 64I, for every T >= 6. The new support is feasible (z* <= 0.1).
- Sigma_AT in [0.02048, 0.02119] GeV^2 (48I), [0.02048, 0.02126] (64I); kaon pole alone is 0.02084.
- J = 1 channels are feasible for both supports but wide (best at T >= 18, z = 3):
  V_S [0.0377, 0.0591] (48I), [0.0367, 0.0580] (64I);  A_S [0.0102, 0.0289] (48I), [0.0062, 0.0276] (64I).
  More moments do not help beyond T ~ 18 (relative error of C(t) > 25%).

## 3. |Vus| (B_s = 0.02908(48)), sum of the four channels (`vus_interval.py`, `vus_certified.py`)
| ensemble | method | T | z | Sigma_tot [GeV^2] | |Vus| |
|---|---|---|---|---|---|
| 48I | old [0,1] | 14-22 | 2, 3 | infeasible | - |
| 64I | old [0,1] | 14-22 | 2, 3 | infeasible | - |
| 48I | new [-1,1] | 18 | 3 | [0.0687, 0.1105] | [0.1962, 0.2489] |
| 64I | new [-1,1] | 18 | 3 | [0.0636, 0.1089] | [0.1977, 0.2588] |
| both | new, overlap | 18 | 3 | | **[0.1977, 0.2488]** |
| both | new, overlap, B_s +-1 sigma added | 18 | 3 | | [0.1961, 0.2509] |
| both | new, overlap | 18 | 2 | | [0.1998, 0.2351] |

Expected Sigma_tot for |Vus| = 0.2243 is 0.0846 GeV^2, inside both intervals.
For context (not an input): HFLAV 2023 quotes |Vus| = 0.2184(21) (OPE) and 0.2189(19) (lattice inclusive); both lie inside.
For comparison the old direct-fit values in `old_code_check/` run from 0.13 to 0.34 and do not converge with T.

## What this does and does not show
- The [0,1] assumption is **not** compatible with the J = 0 data; the [-1,1] one is. That is what makes the old
  method unsafe, and it is independent of any |Vus| number.
- The new interval is wide (about +-11%), not competitive with other determinations. It is a guarantee, not a precision result.
- It holds only if (i) sigma >= 0 on [-1, 1] really is true for this action (checked, not proven), (ii) E >= M0,
  (iii) B_s, Z, a^-1 are right (B_s now from HFLAV). Not included: continuum limit (one lattice spacing each), errors of a^-1 and Z
  (about 0.2% on |Vus| from a^-1), the full covariance (boxes instead), lattice artefacts of the local currents.
- The J = 1 axial channel limits the width; the 48I and 64I intervals overlap but are not combined.
