# Hausdorff positivity scan of the RBC/UKQCD ūs correlators (2026-10-23)

Question: is the lattice two-point function C(t_s + k) a moment sequence of a positive measure
supported on [-1, 1] (hypothesis for M5 > 1) or only on [0, 1] (the assumption behind the
Chebyshev method in `AllClass.py`)?  Criterion: Schmüdgen, arXiv:2008.12698, Thm 3.1.
Code: `hankel_test_proposal.py` (single run) and `hankel_scan.py` (full scan).

## What was run
- Data: `RBCQCD/corr_Z2_l48t96_*` (88 cfg x 48 src) and `corr_Z2_l64t128_*` (40 cfg x 64 src),
  ūs local-local, real part. Channels V1 (g 1,2,4), A1 (14,13,11), V0 (8), A0 (7) and each single gamma.
- Start times t_s = 1..8 (both parities), orders n = 1..3, jackknife over configurations.
- A matrix is a VIOLATION if min-eigenvalue / sigma < -3, a clear PASS if > +3, else inconclusive.
- Row order of the files (cfg, src, t) is an assumption; the autocorrelation along row order is
  at noise level (|ac| < 0.05 for lags 1..100), so the jackknife error does not depend on it.
- The 7 files that were damaged on the old SSD were replaced by md5-verified copies first;
  no file ends in zero padding (scan of 1834 files).

## Result (localizing matrix only, n >= 2; counts pass / inconclusive / VIOLATION)

| channel | ens | support | odd t_s | even t_s |
|---|---|---|---|---|
| V1 (spatial vector) | 48 | [-1,1] | 6/2/0 | 6/2/0 |
| V1 | 48 | [0,1]  | 6/2/0 | 4/3/1 |
| A1 (spatial axial)  | 48 | [-1,1] | 5/3/0 | 4/4/0 |
| A1 | 48 | [0,1]  | 3/5/0 | 3/5/0 |
| V1 | 64 | [-1,1] | 7/1/0 | 6/2/0 |
| V1 | 64 | [0,1]  | 6/2/0 | 6/2/0 |
| A1 | 64 | [-1,1] | 6/2/0 | 6/2/0 |
| A1 | 64 | [0,1]  | 4/3/1 | 5/3/0 |
| V0 (temporal vector) | 48 | [-1,1] | 5/3/0 | 2/5/1 |
| V0 | 48 | [0,1]  | 1/5/2 | 3/5/0 |
| V0 | 64 | [-1,1] | 5/3/0 | 2/3/3 |
| V0 | 64 | [0,1]  | 0/5/3 | 3/5/0 |
| A0 (temporal axial)  | 48 | [-1,1] | 5/3/0 | 2/5/1 |
| A0 | 48 | [0,1]  | 2/5/1 | 3/5/0 |
| A0 | 64 | [-1,1] | 5/3/0 | 3/3/2 |
| A0 | 64 | [0,1]  | 3/3/2 | 3/5/0 |

Full table: run `python hankel_scan.py`.

## Reading (what the data support, and what they do not)
1. **No [-1,1] violation at odd t_s in any channel, on either ensemble.** The hypothesis is compatible
   with the data. Compatibility is not a proof.
2. **J = 0 channels (V0, A0) show the pattern expected for states with lambda < 0 and link-RP only:**
   odd t_s passes [-1,1] but fails [0,1]; even t_s fails [-1,1] (2-3 violations at 64I).
   Caveat: V0 is the timelike component of a non-conserved local current for m_u != m_s; it is
   the least clean channel, so contact terms may mimic this. Do not read it as proof of M5 > 1 effects.
3. **J = 1 channels (V1, A1), which dominate the tau kernel, do not discriminate yet:**
   almost everything is pass or inconclusive for both supports. A1 at 64I has one [0,1] violation
   at odd t_s (z < -3) and none for [-1,1], which is only weak evidence.
4. Many correlated tests were run (the same data over several t_s and n), so isolated 3-sigma
   violations are expected by chance; the V0/A0 pattern is repeated in both ensembles but the
   tests are not independent.
5. Sensitivity to the config x source split of 64I: 40x64 and 80x32 give identical verdicts in all
   listed rows; 20x128 changes one cell (A0, even t_s: 3/3/2 -> 4/2/2).
6. n = 4 and t_s >= 5 are at the noise floor (pivot errors larger than their values); not used.

## Not done yet (needed before any |Vus| claim)
- Free-field exact test of M5 = 1.8 Möbius (no noise; a single negative pivot is decisive).
- Control with M5 = 1.0 on the same gauge fields (Bridge data is not on this machine).
- Valence M5 (1.6 vs 1.8), valence action (Ls = 12 vs sea Ls = 24) and the number of
  configurations (88/80/40) must be confirmed.
- SDP bounds of the tau kernel on [-1,1]; the old `AllClass.py` uses the [0,1] Chebyshev coefficients.
- Check the |Vus| normalization in `AllClass.py` (see `hausdorff/OLD_CODE_NOTES.md`).
