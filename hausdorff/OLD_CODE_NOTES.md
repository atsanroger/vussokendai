# Notes on the old `AllClass.py` (read 2026-10-23, commit 5cac0ac; NOT executed)

These are points found by reading the code. None of them has been tested; each one needs a
check by the author before |Vus| is trusted. "Old version" may already have fixed some.

## What it does
- Reads `RBCQCD/corr_Z2_*` (same files as the scan), averages over sources, jackknifes over configs,
  forms Cbar(t) = C(t + 2 t0) / C(2 t0), then either a Chebyshev fit (`--Fit`) or a direct sum
  with shifted-Chebyshev coefficients, and contracts with the tanh-smeared kernel coefficients
  (`Particle.ang0` = (1-x)^2, `Particle.ang1` = (1-x)^2 (1+2x), x = s/m_tau^2).
- Inputs in `my_header.py`: a^-1 = 1.7295(38) GeV (48I) and 2.3586(7) GeV (64I), Z_A = 0.71076 / 0.74293,
  f_K, m_K, PDG V_us values, S_EW = 1.0201, G_F, m_tau. The Chebyshev basis is the [0,1] one
  (`cof_chebshev`, T*_n(2 lambda - 1)); `--EO Even/Odd` selects the parity of the start time.

## Points to check
1. **Kernel assignment in `RBCQCD.vus`.** `Axial_T` and `Axial_S` use `spectral(1, ...)` (J=1 kernel),
   `Vector_T` and `Vector_S` use `spectral(0, ...)` (J=0 kernel). At zero spatial momentum the
   temporal components (A_0, V_0) carry J=0 and the spatial ones (A_i, V_i) carry J=1, so the
   expected assignment is by time/space: A_T -> J=0, V_S -> J=1.
   The module-level mock code of the same file already uses that assignment (kaon, which sits in
   A_T, with `S = 0`; K*, which sits in V_S, with `S = 1`), so `vus()` disagrees with the rest of the file.
   Run with the original code (patched only for cupy, `old_code_check/`), 48I and 64I, Q0 = 0.05, `--Fit`:
   the two assignments differ by 5-10% in Vus at the same T, e.g. 64I T=6: 0.211 (as written) vs 0.188
   (time/space); 48I T=6: 0.300 vs 0.269.
   **Neither assignment converges with T** (values jump up and down between T=2 and T=8), and the
   quoted error (about 0.0006) does not include the truncation error, so no Vus can be read from it.
2. **`--t0` is ignored in `vus`.** `spectral` calls `self.directsum_chebshev(T, t0, **kwargs)`, but the
   signature is `(self, T, ND=1, t0=0.5, ...)`, so `t0` lands in `ND` and the real t0 stays 0.5.
   The same slip is in the `chebshev_fit_base(T, t0, ...)` call.
3. **No experimental strange branching fraction in the |Vus| formula.**
   `V_us = sqrt(1 / (Sum * Prefactor * SEW * a^-3 / (totaldecaywidth * Z_A^2)))`, where
   `totaldecaywidth = Gamma_e / 0.1779` is the total tau width. A determination from
   Gamma(tau -> X_s nu) = |V_us|^2 * (lattice sum) needs B_s (about 0.029) in the numerator.
   Check that `Sum` already contains it, or the formula is incomplete.
4. **Module-level code runs on import.** The bottom of the file performs fits and plots, and
   rebinds the name `Fit` (first the `--Fit` flag, then an lsqfit result). `vus` passes `Use_Fit = Fit`
   and tests `Use_Fit == True`, so after the rebinding the fit path is never taken.
5. **Configuration count.** 327,680 rows of 64I are 80 cfg x 32 src (`SixFour`, `Ncfg = 80` here) or
   40 cfg x 64 src (what `hankel_test_proposal.py` assumed); both reproduce the row count and the
   file does not say which. For the Hankel scan the choice does not matter (identical verdicts for
   40x64 and 80x32; one cell differs at 20x128), but the quoted statistics should be fixed from the
   run records. `jacknifed_correlation` infers the number of sources from the row count.
6. **Valence M5.** `M5 = 1.8` is hard-coded for both ensembles; `postive.tex` lists 1.6 for this valence set.
7. **Parity**: the code takes `data[::2]` / `data[1::2]` for Even/Odd, i.e. start times 0 or 1 after the
   `2 t0` roll; the new scan finds the parity matters for the J=0 channels.

## What the new work adds
`hankel_scan.py` / `RESULTS.md` test whether the [-1,1] support is compatible with the data, which is the
prerequisite for replacing the [0,1] Chebyshev basis by a [-1,1] one and for the SDP bounds.
