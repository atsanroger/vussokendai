# Inclusive reconstruction at M5 > 1: literature check and old-data inventory

2026-10-21. Research notes only. Nothing in RutticeQCD was changed and no data was modified.
Companion file: `hankel_test_proposal.py` (a proposal; **it has not been run or tested**).

Labels used below:
- **[checked]**: I read the arXiv abstract page or the arXiv API record, or the full text.
- **[memory]**: a citation detail I could not confirm online this session. Check it before citing.

---

## 0. TL;DR

1. **Syritsyn–Negele is a proceedings paper, not JHEP.** It is arXiv:0710.0425, PoS(LATTICE 2007)078.
   - What it shows: for Shamir DWF with M > 1, the one-body operator in the *t*-direction transfer matrix (built following Lüscher) has **real, negative eigenvalues**. These give the (−1)^t terms.
   - What it does *not* do: it never discusses the sign of the spectral weights.
   - One technical line matters here. The eigenvectors are orthogonal with respect to the weight matrix diag(B, B), and B becomes **indefinite** exactly when M > 1. My reading is that the naive Fock-space metric is then not positive. So positivity of the weights is **not** guaranteed by that construction.
   - I found **no published proof** of time-direction reflection positivity for DWF/Möbius at M5 > 1, and no disproof either.
   - Möbius (b, c) is not addressed. With c ≠ 0 the temporal hopping also enters the s-off-diagonal blocks, so the Syritsyn–Negele construction does not carry over verbatim.
   - **Verdict:** your hypothesis ("T Hermitian, weights ≥ 0, support on [−1, 1]") is plausible but unproven. It is directly testable on data with the Hankel conditions in §2.
2. **There is a parity subtlety that decides which moments to test.**
   - If the theory has *link* reflection positivity but not *site* reflection positivity (the classic situation for staggered fermions, and plausibly for Wilson-type kernels beyond the positive-T regime), the positive measure is in **C(1+k)**: C(t) = Σ|c_n|² λ_n^{t−1}.
   - In that case the coefficient of λ^t for a λ < 0 state is **negative**: a fit of the form B(−1)^t e^{−Dt} gives B < 0.
   - Your own fits (`BetterData/fit_m_osc`, `fit_mobius`) have negative central values of C0 (for example −0.00059(79) and −0.49(96)). These are statistically consistent with zero, so they prove nothing either way.
   - **Test both parities of the starting time.** A positive measure on [−1, 1] stays positive under even shifts λ^{2m}μ, but not under odd ones.
   - The thesis normalisation C̄(t) = C(t+2t₀)/C(2t₀) with t₀ = 0.5 is an *odd* shift. Bailas–Hashimoto–Ishikawa use t₀ = 1, an even shift.
3. **The Hausdorff [−1, 1] criterion is exact and verified** (Schmüdgen, arXiv:2008.12698, Thm 3.1/3.2; book GTM 277, Ch. 10).
   - For an even number of moments 2n: the two conditions are H = (s_{i+j})_{0..n} ⪰ 0 and (s_{i+j} − s_{i+j+2})_{0..n−1} ⪰ 0. Positive-semidefiniteness of these two matrices is necessary and sufficient. There is no extra rank condition on a compact interval.
   - For an odd number 2n+1: (s_{i+j} + s_{i+j+1}) ⪰ 0 and (s_{i+j} − s_{i+j+1}) ⪰ 0.
4. **Every inclusive method I checked assumes λ ∈ (0, 1]** (equivalently, energies E ≥ E₀ with positive spectral density): Chebyshev (Bailas–Hashimoto–Ishikawa, Gambino–Hashimoto), HLT, Backus–Gilbert, and Lanczos.
   - Bailas et al. say explicitly that many lattice actions do not satisfy reflection positivity. They assume they are close enough to the continuum.
   - **Nobody uses Hausdorff on [−1, 1] for inclusive kernels.** The closest is Abbott–Jay–Oare (arXiv:2508.01377). They map staggered λ < 0 states onto the *Hamburger* problem (support on all of ℝ) and give rigorous Weyl-ball bounds.
5. **Data: yes, you have M5 > 1 correlators.**
   - **RBC/UKQCD 48I and 64I** ūs local-local vector and axial two-point functions, all 8 gammas (V_x, V_y, V_z, V_t, A_x, A_y, A_z, A_t), raw per configuration and per source. They are in `~/BetterData/RBCQCD/` and several identical copies.
   - Also **15 JLQCD M5 = 1.0 Möbius ensembles** (V_i, A_4, P, S channels, 16 GB).
   - Your own Bridge++ M5 = 1.0 / 1.8 scans from the thesis.
6. **Disk warning (outside scope, important).** `dmesg` shows `I/O error, dev sdd` read errors on the WSL virtual disk.
   - 4 of the 8 files in `~/Data/Lattice/48c/` cannot be read.
   - Identical good copies exist elsewhere (md5 matched), so nothing is lost yet. **Back up `~/Data` and `~/BetterData`** (and check the host SSD) soon.

---

## 1. Literature: can inclusive reconstruction work at M5 = 1.8?

### 1.1 Syritsyn & Negele, and follow-ups

- **S. Syritsyn, J. W. Negele, "Oscillatory terms in the domain wall transfer matrix"**, arXiv:0710.0425, PoS(LATTICE 2007)078 **[checked, full text read]**.
  - Shamir DWF in temporal gauge.
  - Free-field geometries: 1 space + s + t, and 3 space + s + t. Interacting test: HYP-smeared MILC lattices at M = 1.7.
  - They construct the *t*-direction transfer matrix following Lüscher (CMP 54 (1977) 283), for operators placed on the s-boundaries.
  - Key statements (paraphrased):
    - The free B = D + 1 − M − Σ cos k_i − cos k_s is ≥ 0 for M ≤ 1. For M > 1 it is negative for some k.
    - The transfer-matrix eigenfunctions are orthogonal with weight diag(B, B), and all eigenvalues are real.
    - Negative η gives terms that flip sign under a one-step evolution.
    - The negative modes sit at the cutoff scale and are "innocuous".
    - Remedy: fit with an extra Ã(−1)^t e^{−m̃t} term. They also advise choosing M per action so that these modes do not appear.
  - **Not discussed:** the sign of the spectral weights, reflection positivity, Möbius.
  - My inference (flagged as inference): an indefinite B means the "weight matrix" metric is indefinite for M > 1. So the Lüscher-style Hilbert space is not manifestly positive, and positive weights are *not* established by this construction. It also does not show that they fail.
- **R. S. Sufian, M. J. Glatzmaier, Y.-B. Yang, "Unphysical Poles of Domain Wall Fermions at finite Ls"**, arXiv:1603.01591 (no journal reference listed) **[checked abstract]**.
  - Free 4D propagator poles for Shamir, Boriçi and Möbius.
  - Oscillations come from unphysical poles. Their remedy is M·a₅ < 1 (Shamir/Boriçi) or M(b₅ − c₅) < 1 (Möbius).
  - Note: RBC 48I/64I have b − c = 1, so this condition fails at M5 = 1.8.
- **M. Tomii, "Unphysical poles and dispersion relations for Möbius domain-wall fermions in free field theory at finite Ls"**, PRD 96 (2017) 074504, arXiv:1706.03099 **[checked]**.
  - There are Ls − 1 extra poles. Small b − c raises their mass, which is how (b, c) changes the picture.
  - Not addressed: weights or positivity.
- **W.-L. Chen, "Unphysical pole in lattice QCD"**, PhD thesis, SOKENDAI, degree 2025-09-26 (甲第2603号) **[checked, full text read]**. Repository record: https://ir.soken.ac.jp/records/2000386
  - Free-field residue analysis: r(t) = C_phys/C at M5 = 1.0 and 1.8 for Shamir, Möbius (1.5, 0.5) and Boriçi, Ls = 8.
  - Uses the shifted-Chebyshev bound |⟨T*_N⟩| ≤ 1 as a detector of λ < 0.
  - Data: JLQCD and quenched configurations at M5 = 1.8, with 0, 3 and 6 stout steps.
  - **Relevant to your new idea:** the thesis tests the [0,1] Hausdorff hypothesis in Chebyshev form. It does *not* test whether the [−1, 1] hypothesis holds. The odd normalisation (2t₀ = 1) matters (see §0.2).
  - Context: Boyle et al.'s RBC/UKQCD inclusive-τ |Vus| work (arXiv:1803.07228, PRL 121 (2018) 202003, **[checked]**) uses dispersive weights on lattice HVPs, not spectral reconstruction. That approach needs no spectral positivity, which is presumably why M5 = 1.8 was never an issue there.

### 1.2 Reflection positivity and the transfer matrix (background)

- M. Lüscher, CMP 54 (1977) 283: a self-adjoint, strictly positive transfer matrix for Wilson lattice gauge theory with Wilson fermions. Citation confirmed through Syritsyn–Negele's reference list. The exact κ range in the paper itself is **[memory]**: I recall positivity for κ < 1/6, but did not verify it.
- P. Menotti, A. Pelissetto, "General proof of Osterwalder–Schrader positivity for the Wilson action", CMP 113 (1987) 369–373.
  - Extends Osterwalder–Seiler to site reflections.
  - A secondary source says the proof holds for K < 1/6; I could not open the PDF (**[partly checked]**).
- K. Osterwalder, E. Seiler, Ann. Phys. 110 (1978) 440: link-reflection positivity for lattice gauge theory with Wilson fermions. Volume and page are **[memory]**.
- Y. Kikukawa, K. Usui, "Reflection positivity of free overlap fermions", PRD 82 (2010) 114503, arXiv:1005.3751 **[checked]**: link-reflection positivity for free overlap fermions.
- Staggered fermions are the textbook case of "Hermitian, not positive" (λ < 0, opposite-parity states with (−1)^t). Hackett–Wagman (arXiv:2412.04444) and Abbott–Jay–Oare (arXiv:2508.01377) both note this. Abbott–Jay–Oare map it to the Hamburger moment problem.
- Logic for the user's case:
  - Link reflection positivity (OS w.r.t. planes between time slices) alone gives a positive Hilbert space and a Hermitian contraction T, with ⟨F|T^{t−1}|F⟩ = C(t).
  - Site reflection positivity in addition gives T ≥ 0.
  - So "λ < 0 but weights ≥ 0" is exactly the link-RP-only scenario, with the odd offset of §0.2.
  - Whether 5d DWF/Möbius at M5 > 1 (with Pauli–Villars fields and c ≠ 0) has link RP is, as far as I could find, **open**.

### 1.3 Moment-problem conditions on [−1, 1] (exact)

Source: K. Schmüdgen, "Ten Lectures on the Moment Problem", arXiv:2008.12698, Lecture 3, Thms 3.1–3.2 **[checked, text read]**. These follow the book: Schmüdgen, *The Moment Problem*, GTM 277, Springer 2017, Ch. 10.

For [a, b], even case m = 2n, the sequence s₀..s₂ₙ is a truncated [a,b] moment sequence of a positive Radon measure **iff**:

- (s_{i+j})_{i,j=0..n} ⪰ 0, and
- ((a+b)s_{i+j+1} − ab·s_{i+j} − s_{i+j+2})_{i,j=0..n−1} ⪰ 0.

Odd case m = 2n+1: (s_{i+j+1} − a s_{i+j})_{0..n} ⪰ 0 and (b s_{i+j} − s_{i+j+1})_{0..n} ⪰ 0.

With [a, b] = [−1, 1]:

| case | condition 1 | condition 2 |
|---|---|---|
| even 2n | (s_{i+j})_{0..n} ⪰ 0 | (s_{i+j} − s_{i+j+2})_{0..n−1} ⪰ 0 |
| odd 2n+1 | (s_{i+j} + s_{i+j+1})_{0..n} ⪰ 0 | (s_{i+j} − s_{i+j+1})_{0..n} ⪰ 0 |

For comparison, the old [0,1] assumption (even case) is (s_{i+j}) ⪰ 0 and (s_{i+j+1} − s_{i+j+2}) ⪰ 0.

Further results from the same lecture:

- Interior points of the moment cone ⇔ both matrices are positive *definite*.
- Each interior point has exactly two **principal** representing measures (upper/lower; these are the Gauss–Radau-type quadratures).
- For every ξ ∈ (a, b) there is exactly one **canonical** measure with an atom at ξ. This is what drives the Chebyshev–Markov–Stieltjes extremal bounds.
- Uniqueness and boundary points are covered there too.

### 1.4 Rigorous bounds from moments

- **Golub & Meurant**, *Matrices, Moments and Quadrature with Applications*, Princeton University Press 2010 **[checked existence and contents: Ch. 7, bounds for bilinear forms]**.
  - For f whose derivatives have constant sign on [a, b], Gauss gives one side and Gauss–Radau or Gauss–Lobatto gives the other.
  - The original paper, Golub & Meurant 1994, "Matrices, moments and quadrature" (Pitman Res. Notes Math. 303), is **[memory]**.
  - **Caveat:** inclusive kernels (θ-function phase space, smeared step) do *not* have constant-sign derivatives. Golub–Meurant does not apply directly.
- **Chebyshev–Markov–Stieltjes (Markov–Krein) inequalities.**
  - Given moments up to degree 2n of a positive measure, the distribution function μ([a, x]) is bracketed by partial sums of the Gauss (canonical) quadrature weights at nodes below x.
  - The sharp extremal values come from the canonical measure with an atom at x (Schmüdgen Thm 3.9; Krein–Nudel'man 1977; Karlin–Studden 1966).
  - Standard texts are Akhiezer (1965), Szegő's *Orthogonal Polynomials*, and Freud's *Orthogonal Polynomials*. The specific theorem numbers are **[memory]**.
  - History: arXiv:math/9307220 **[checked]**. A differential version: Gilboa–Peled, J. Approx. Theory (2015) (journal details **[memory]**).
- **General kernels.**
  - The sharp bounds on ∫K dμ over all positive μ on [−1, 1] matching the data are a linear program over the moment cone. For polynomial or piecewise-polynomial K the dual is an SDP (Markov–Lukács certificates).
  - Recent lattice versions of exactly this, but with energy support ℝ⁺, i.e. λ ∈ (0, 1]:
    - Abbott, Fields, Jay, Oare, Saccardi, "The Causal Bootstrap", arXiv:2605.20509 (May 2026) **[checked]**.
    - Mutzel, Tilloy, "Certified spectral functions from lattice Monte Carlo data", arXiv:2606.09791 (June 2026) **[checked]**; uses reflection positivity explicitly.
    - Abbott, Jay, Oare, arXiv:2508.01377 (Aug 2025) **[checked]**: Hausdorff on [0,1], Stieltjes for thermal states, Hamburger for staggered λ < 0. Rigorous Weyl-ball bounds. It also warns about "bleed-in" from λ < 0 and λ > 1 states.
  - Switching any of these to Hausdorff on [−1, 1] is mechanical: change the localizing matrix to (1 − λ²).
- **Nevanlinna in the λ variable.**
  - For positive μ on [−1, 1], F(w) = ∫dμ(λ)/(w − λ) = Σ_k s_k w^{−k−1} is a Markov/Nevanlinna function off [−1, 1]. So Nevanlinna–Pick and Padé bounds work naturally there.
  - Only the *energy-variable* formulation assumes λ > 0.
  - This is standard moment theory (Markov's theorem on Padé approximants of Markov functions). I did not find a lattice paper that uses it this way.

### 1.5 Current inclusive methods: does each assume λ ≥ 0? has anyone used [−1, 1]?

| Method | Key papers (all **[checked]** via arXiv API) | Assumes λ ∈ (0, 1]? | Uses [−1, 1]? |
|---|---|---|---|
| Chebyshev (shifted T*_n(2λ−1)) | Hashimoto, PTEP 2017, arXiv:1703.01881; Bailas–Hashimoto–Ishikawa, PTEP 2020, arXiv:2001.11779; Gambino–Hashimoto, PRL 125 (2020) 032001, arXiv:2005.13730; Gambino et al., arXiv:2203.11762; Barone et al., JHEP 07 (2023) 145, arXiv:2305.14092; Barone et al., arXiv:2312.17401; Kellermann et al., arXiv:2504.03358 | **Yes, explicitly** ("eigenvalues of ẑ constrained to lie between 0 and 1"; fits impose \|T̄_j\| ≤ 1). Uses JLQCD Möbius data (M5 not stated in 2001.11779) | No. The thesis uses \|⟨T*⟩\| > 1 only as a *detector* |
| Backus–Gilbert | Hansen–Meyer–Robaina, PRD 96 (2017) 094513, arXiv:1704.08993 | Yes (E ≥ E₀ basis e^{−tE}) | No |
| HLT | Hansen–Lupo–Tantalo, PRD 99 (2019) 094508, arXiv:1903.06476; τ: Evangelista et al., arXiv:2308.03125; Alexandrou et al., arXiv:2403.05404; staggered: Blum, Jay, Jin, Kronfeld, Stewart, arXiv:2411.14300 | Yes. The A[g] functional controls only E ≥ E₀, so λ < 0 contributions Σ g_t λ^t are uncontrolled. The staggered paper discusses "strategies for opposite-parity effects" (abstract only read) | No |
| Nevanlinna / conformal maps | Fei–Yeh–Gull, PRL 126 (2021) 056402, arXiv:2010.04572; Bergamaschi–Jay–Oare, arXiv:2305.16190 | Yes (positive spectral density in energy) | No |
| Lanczos / moment bounds | Wagman, arXiv:2406.20009 (assumes T Hermitian and positive; negative Ritz values treated as spurious); Hackett–Wagman, PRD 112 (2025) 014514, arXiv:2412.04444; Abbott–Jay–Oare, arXiv:2508.01377 | Mostly yes. Abbott–Jay–Oare treat λ < 0 by Hamburger | Hamburger (ℝ), not Hausdorff [−1, 1] |
| Review | Hashimoto, "Smeared spectral functions from lattice QCD: opportunities and challenges", arXiv:2609.02239 | (did not read past the abstract) | — |

**Bottom line:** the [−1, 1] Hausdorff + Chebyshev-on-[−1, 1] + moment-bound programme looks new for inclusive decays. It rests on an unproven positivity property that you can test directly.

### 1.6 Practical remarks on the [−1, 1] Chebyshev idea

- **Smoothness.** K(λ) is zero for λ < e^{−a m_τ}, so the continuation by 0 to [−1, 0) is smooth at λ = 0. The roughness is at threshold (the τ kernel vanishes like (1 − s/m_τ²)², so it is C¹) and is the same as before.
- **Resolution cost.** T_N(λ) on [−1, 1] has half the λ resolution of T*_N(2λ−1) on [0, 1]. Expect roughly 2× the polynomial order for the same truncation error.
  - Alternative: approximate K only on [0, 1], keep |p| bounded on [−1, 0), and bound the λ < 0 piece by max|p| · μ([−1, 0)). Bound μ([−1, 0)) from the moments by CMS/canonical measures.
- **Finite T.** C(t) on a periodic lattice is a thermal trace, and around-the-world terms ∝ λ_m^{N_t − t} mix signs for λ < 0.
  - At 48I, e^{−a m_π N_t} ≈ e^{−0.080·96} ≈ 5×10⁻⁴, which is not negligible at the precision of Hankel determinants.
  - Use t ≪ N_t/2, or subtract the backward ground state first.
- **Data alone cannot prove positivity, but it can refute it.** A significantly negative Hankel pivot falsifies the hypothesis for that operator. Passing is consistency only.
  - The decisive theory test is free-field: compute the exact free Möbius (b=1.5, c=0.5, Ls=12, M5=1.8) V–V correlator, where there is no noise, and apply §1.3. The thesis already has the free 4D propagator machinery (thesis Ch. 3; your `postive.tex` contour-integral code).

---

## 2. Old data on this machine (read only; nothing changed)

### 2.1 M5 > 1: RBC/UKQCD 48I / 64I ūs currents (main candidate for the Hankel test)

- **Path (good copy):** `~/BetterData/RBCQCD/` (16 files, 275 MB).
  - Byte-identical copies (md5 match): `~/Data/`, `~/Data/Lattice/Data/`, `~/Data/Lattice/Data/Second/`, `~/Data/Lattice/64c/`.
  - Also `~/Data/Lattice/48c/` for gammas 2, 11, 13 and 14 only. **Gammas 1, 4, 7 and 8 there hit I/O errors.**
  - OneDrive `Desktop\Data\Lattice` and `Desktop\LatticeTau\Data\Lattice` (280 MB each) look like copies. I did not hash them.
- **Files:** `corr_Z2_l48t96_Ls12_m0.00078_LL_m0.0362_LL_{g}_{g}` and `corr_Z2_l64t128_Ls12_m0.000678_LL_m0.02661_LL_{g}_{g}`, with g ∈ {1, 2, 4, 8, 7, 11, 13, 14}. All dated 2015-12-20.
  - Gamma-bitmask reading (inferred, matches `ConfigCreator.py` and your notes): 1, 2, 4, 8 = γx, γy, γz, γt (vector); 14, 13, 11, 7 = A_x, A_y, A_z, A_t (∝ γ_μγ5).
  - Flavour: light (0.00078 / 0.000678) × strange (0.0362 / 0.02661), i.e. the ūs channel. Local-local currents with Z2 noise sources.
- **Format:** ASCII, 8 columns `t Γsrc Γsnk 0 0 0 Re Im`, blocks of N_t rows.
  - 48I: 405,504 rows = 88 cfg × 48 sources × 96.
  - 64I: 327,680 = 40 cfg × 64 sources × 128.
  - The cfg × src split comes from `BetterData/ConfigCreator.py`. Your `finalv120230501.tex` table says 40/88 instead, and arXiv:1411.7017 analyzed 80 (48I) and 40 (64I) configs. Confirm before jackknifing.
- **Ensembles** (arXiv:1411.7017, PRD 93 (2016) 074505, **[checked]**): 48I is 48³×96, Ls(sea) = 24, β = 2.13, a⁻¹ = 1.730 GeV; 64I is 64³×128, Ls = 12, β = 2.25, a⁻¹ = 2.359 GeV. Both Iwasaki, Möbius with b+c = 2 and b−c = 1 (so b = 1.5, c = 0.5).
  - Data reference in your `tau.bib`: Boyle et al., PRD 93 (2016) 054502, arXiv:1511.01950 **[checked]**.
- **Uncertainty on M5.**
  - The thesis states M5 = 1.8 as the RBC/UKQCD choice. I could not find M5 in the parts of 1411.7017 I read.
  - Your own `OneDrive\Desktop\LatticeTau\postive.tex` table lists **M5 = 1.6, Ls = 12, b = 1.5, c = 0.5** for this valence data.
  - The valence Ls = 12 on 48I (sea Ls = 24) suggests a different valence action (zMöbius?).
  - **Either way M5 > 1.** Confirm the valence action with the data providers.
- Partially quenched 48I: `~/Data/Lattice/48c-pq-run0/{1040,1360,1480,1960,2120}/`, 5 cfgs, 256 `.cor` files each.
  - Masses (0.0006979, 0.0358) and (0.00078, 0.0362), both orderings, all 8×8 Γ combinations of {G0..G3, G0G5..G3G5}.
  - Format: 96 rows, `t Re Im`. 11 MB.
- 64I PQ: `~/Data/Lattice/64c-pq/{temporal,time_only}_{unitary,pqMunitary}_{v,a}_ls/{1380,1820,1900,2660,2700}`, 2 columns, 196 KB.
- Derived products: `~/Data/Lattice/Data/{4811,…,645454}` (source-averaged, 3 columns), `COV*`, `JACKMEAN*`; `~/Data/*.csv` (`Average_*`, `AB48*.csv`, …); `~/BetterData/CovofFourEight*/CovofSixFour*.csv`; experimental τ data in `~/Data/Experiment/` (Belle Kπ, …).

### 2.2 M5 = 1.0: JLQCD Möbius DWF

- `~/BetterData/data_and_fits-002/` (16 GB, dated 2016–2017). 15 ensembles `SymDW_sHtTanh_b2.0_smr3_{L}x{T}x{Ls}_b{β}_M1.00_mud{…}_ms{…}`:
  - β = 4.17: 32³×64×12 (7 ensembles; mud 0.0035–0.019; ms 0.030 / 0.040) and 48³×96×12 (mud 0.0035).
  - β = 4.35: 48³×96×8 (6 ensembles; mud 0.0042 / 0.0080 / 0.0120; ms 0.0180 / 0.0250).
  - β = 4.47: 64³×128×8 (mud 0.0030, ms 0.0150).
- Each ensemble has `data/` (~927 files) plus many fit directories.
  - Channels: PP, A4A4, A4P, PA4, SS, V1V1, V2V2, V3V3, vectorave. Flavours: ud-ud, ud-s, s-s, heavy-{ud, s, heavy}. Smearing combos 0-0, 0-1, 1-0, 1-1.
  - Format: CSV `t, value`, concatenated configs. 48×96: 4800 rows = 50 cfg × 96; 32×64: 6400 = 100 × 64; 64×128: 12800 = 100 × 128.
  - **No spatial axial A_iA_i**, so the J = 1 axial part of the τ kernel is not available here.
- Same ensembles, merged Z2-noise data and fits: `~/BetterData/drive-download-20230531T042218Z-001/` (merged data 443 MB, fits 363 MB). Also `raw_data_one_ensemble/` (247 `hadronCorrel_*` text files) and the IroIro `python_analysis/`.
- `~/BetterData/Ryan/data2Pt_Ryan_mud{0.0042,0.0080,0.0120}.h5` and `~/Data/data2Pt_Ryan_NewSmearing14.14.h5`: HDF5, groups LL/LS/SL/SS with Re/Im of shape (400, 96). Probably the β = 4.35 48³×96 ensembles; the channel is unlabelled.

### 2.3 Your thesis's own M5 scans (Bridge++)

- `~/BetterData/BRIDGE/` (`M51.0Mobius.csv`, `M51.8Mobius.csv`, `M51.0Borici.csv`, `M51.8Borici.csv`, `M51.8Mobius{JLQCD,Quench}SM{0,3,6}[B].csv`, `PSPS.csv`, `A4PS.csv`), `~/BetterData/20250504/` (`M510Mobius`, `M518Mobius`, `M518b15c05Rindou`), `~/BetterData/Aurona/` (`M510BSC*`, `M516BSC*`), `~/BetterData/Peagsus/` (Bridge++ logs, 32³×64, M0 = 1.8), `~/Bridge/Neff/eigentest/`.
- Mostly pseudoscalar correlators, 32³×64 JLQCD configs or 16³×32 quenched.
- Format: `t Re Im` per config, with `# Conf` headers in some files.
- These directly compare **M5 = 1.0 vs 1.8 on the same gauge fields**. They are ideal for a controlled Hankel comparison (P–P is reflection-positive in the same sense as V–V).

### 2.4 Windows side

- `C:\Users\atsan\OneDrive\Desktop\LatticeTau\`:
  - Drafts: `finalv120230501.tex/pdf` ("The tau inclusive hadronic decay from lattice QCD": Chebyshev, exclusive mock checks, Vus) and `postive.tex/pdf` ("Detecting the unphysical contribution of domain-wall fermion by Chebyshev approximation", with a section "Positivity check by ⟨T̄_i⟩").
  - `tau decay*.tex`, Mathematica notebooks (`ChebAppTestPlot.nb`, `KernelApproximation.nb`, …), `tau.bib`, and a copy of the data (280 MB).
- `C:\Users\atsan\OneDrive\Desktop\Data\`: another copy plus analysis scripts.
- Not PhD data: `C:\Users\atsan\ruttice_ckpt` (3 × 1.7 GB RutticeQCD checkpoints), `wsl_offload` (Nsight profiles), `JOB` (PDFs).
- `Downloads` is 41 GB of mixed files. I did not search it in depth, and my whole-home keyword search was stopped before it finished, so the Windows side is not exhaustive.
- WSL dirs I did not open in depth: `~/hashimoto` (an autotools tree, 84 MB), `~/Bridge` (apart from `Neff/eigentest`), `~/RutticeQCD*` (skipped `run/` and `target*/` as asked).

---

## 3. Proposed Hankel positivity test (NOT run)

Data: `~/BetterData/RBCQCD/`, 48I and 64I, channels V(J=1) = mean of g = 1, 2, 4; A(J=1) = mean of g = 14, 13, 11; V(J=0) = 8; A(J=0) = 7.
Repeat on the `BRIDGE/` M5 = 1.0 vs 1.8 pairs as a control.

1. Average over sources, then jackknife over configurations (bin if autocorrelated).
   - Fix the overall sign so that C > 0 in the plateau region.
   - Use t ≤ N_t/2 − margin only.
2. For each start t_s ∈ {1, 2, …, 8} (**odd and even separately**, see §0.2) and n = 1..4, set s_k = C(t_s + k)/C(t_s), k = 0..2n.
3. Build three families of matrices:
   - **H0** = (s_{i+j})_{0..n}. Positivity of the measure; necessary for any support.
   - **H[−1,1]** = (s_{i+j} − s_{i+j+2})_{0..n−1}. Support in [−1, 1].
   - **H[0,1]** = (s_{i+j+1} − s_{i+j+2})_{0..n−1}. This is the old λ ≥ 0 assumption. Also check (s_{i+j+1})_{0..n−1} ⪰ 0, which together with H0 is the Stieltjes condition for support in [0, ∞).
4. Rescale each matrix to unit diagonal (a congruence, so PSD status is unchanged). Report the minimal eigenvalue and the LDLᵀ pivots with jackknife errors. Watch the jackknife bias near zero: quote the pivot on the mean ± jackknife σ.
5. Expected pattern if your hypothesis holds:
   - H0 and H[−1,1] ⪰ 0 for one parity of t_s at all n.
   - H[0,1] (and possibly H0 at the other parity) significantly violated at small t_s, n ≥ 2.
   - At M5 = 1.0 everything passes for both parities.
6. Free-field exact check (strongly recommended before trusting data): free Möbius V–V at the 48I/64I parameters, M5 = 1.8. No noise, so any negative pivot is decisive.
7. If the test passes, the next step is the [−1, 1] SDP bound on ∫K dμ: the §1.4 dual with localizing polynomial (1 − λ²).

The skeleton is in `hankel_test_proposal.py` (paths and the cfg × src split are parameters; please check them).
