# M5 > 1 時的 inclusive 重建：文獻查核與舊資料盤點（繁體中文版）

2026-10-21。純研究筆記：沒有改動 RutticeQCD，也沒有修改任何資料。
英文原版：`REPORT.md`。附檔 `hankel_test_proposal.py` 只是提案，**沒有執行，也沒有測試**。

標記說明：
- **[已查證]**：讀過 arXiv 摘要頁、arXiv API 紀錄或全文。
- **[憑記憶]**：這次無法在線上確認的引用細節，引用前請再核對。

---

## 0. 重點摘要

1. **Syritsyn–Negele 是會議論文，不是 JHEP。** 它是 arXiv:0710.0425，PoS(LATTICE 2007)078。
   - 它證明的事：Shamir DWF 在 M > 1 時，依 Lüscher 方法構造的 *t* 方向 transfer matrix，其單體算符有**實數的負特徵值**，這就是 (−1)^t 項的來源。
   - 它*沒有*做的事：完全沒有討論譜權重（spectral weights）的正負號。
   - 有一個技術細節很關鍵。特徵向量是對權重矩陣 diag(B, B) 正交，而 B 恰好在 M > 1 時變成**不定號（indefinite）**。我的解讀是：這時 Fock 空間的度規不再是正定的，所以這個構造**並不保證**權重非負。
   - 我沒有找到任何已發表的證明，說 M5 > 1 的 DWF／Möbius 在時間方向具有 reflection positivity，也沒有找到反證。
   - 論文沒有處理 Möbius 的 (b, c)。c ≠ 0 時，時間方向的 hopping 也會出現在 s 方向的非對角區塊，所以 Syritsyn–Negele 的構造不能直接套用。
   - **結論：**你的假設（T Hermitian、權重 ≥ 0、支撐在 [−1, 1]）合理但未經證明。可以直接用 §1.3 的 Hankel 條件在資料上檢驗。
2. **有一個奇偶性（parity）的細節，決定要檢驗哪一組 moment。**
   - 如果理論只有 *link* reflection positivity，沒有 *site* reflection positivity（staggered 費米子的典型情況；超出 T 正定範圍的 Wilson 型 kernel 也可能如此），正測度落在 **C(1+k)**：C(t) = Σ|c_n|² λ_n^{t−1}。
   - 這時 λ < 0 的態，其 λ^t 的係數是**負的**：擬合 B(−1)^t e^{−Dt} 會得到 B < 0。
   - 你自己的擬合（`BetterData/fit_m_osc`、`fit_mobius`）中，C0 的中心值是負的（例如 −0.00059(79)、−0.49(96)），但統計上與零相容，正反都證明不了什麼。
   - **兩種起始時間的奇偶都要測。**[−1, 1] 上的正測度乘上 λ^{2m}（偶數平移）仍是正的，乘上奇數次方就不是。
   - 論文中的歸一化 C̄(t) = C(t+2t₀)/C(2t₀) 取 t₀ = 0.5，是*奇數*平移。Bailas–Hashimoto–Ishikawa 用 t₀ = 1，是偶數平移。
3. **[−1, 1] 上的 Hausdorff 判準是精確的，並已查證**（Schmüdgen，arXiv:2008.12698，定理 3.1／3.2；專書 GTM 277 第 10 章）。
   - 偶數個 moment（2n）：兩個條件是 H = (s_{i+j})_{0..n} ⪰ 0 與 (s_{i+j} − s_{i+j+2})_{0..n−1} ⪰ 0。這兩個矩陣半正定是充分必要條件。在緊緻區間上不需要額外的秩條件。
   - 奇數個 moment（2n+1）：(s_{i+j} + s_{i+j+1}) ⪰ 0 與 (s_{i+j} − s_{i+j+1}) ⪰ 0。
4. **我查過的所有 inclusive 方法都假設 λ ∈ (0, 1]**（等價於能量 E ≥ E₀ 且譜密度為正）：Chebyshev（Bailas–Hashimoto–Ishikawa、Gambino–Hashimoto）、HLT、Backus–Gilbert、Lanczos。
   - Bailas 等人明確寫到許多格點作用量不滿足 reflection positivity，並假設他們離連續極限夠近。
   - **沒有人把 [−1, 1] 上的 Hausdorff 問題用在 inclusive kernel 上。**最接近的是 Abbott–Jay–Oare（arXiv:2508.01377）。他們把 staggered 的 λ < 0 態對應到 *Hamburger* 問題（支撐為整條實數線），並給出嚴格的 Weyl-ball 界限。
5. **資料：有，你手上有 M5 > 1 的關聯函數。**
   - **RBC/UKQCD 48I 與 64I** 的 ūs local-local 向量流與軸向流兩點函數，8 個 gamma 都有（V_x、V_y、V_z、V_t、A_x、A_y、A_z、A_t），逐組態、逐源的原始資料。位於 `~/BetterData/RBCQCD/`，另有數份相同的副本。
   - 另有 **15 個 JLQCD M5 = 1.0 Möbius ensemble**（V_i、A_4、P、S 通道，16 GB）。
   - 以及你論文中用 Bridge++ 做的 M5 掃描。
6. **磁碟警告（超出範圍但很重要）。**`dmesg` 顯示 WSL 虛擬磁碟有 `I/O error, dev sdd` 讀取錯誤。
   - `~/Data/Lattice/48c/` 裡 8 個檔案中有 4 個讀不出來。
   - 其他位置有內容相同的完好副本（md5 一致），所以目前沒有遺失。**請盡快備份 `~/Data` 與 `~/BetterData`**，也檢查一下主機的 SSD。

---

## 1. 文獻：M5 = 1.8 時 inclusive 重建是否可行？

### 1.1 Syritsyn & Negele 及後續工作

- **S. Syritsyn, J. W. Negele, "Oscillatory terms in the domain wall transfer matrix"**，arXiv:0710.0425，PoS(LATTICE 2007)078 **[已查證，讀過全文]**。
  - 時間規範下的 Shamir DWF。
  - 自由場幾何：1 空間維 + s + t，以及 3 空間維 + s + t。有交互作用的測試：M = 1.7 的 HYP-smeared MILC 組態。
  - 依 Lüscher（CMP 54 (1977) 283）的方法構造 *t* 方向的 transfer matrix，算符放在 s 方向的邊界上。
  - 主要論述（意譯）：
    - 自由場的 B = D + 1 − M − Σ cos k_i − cos k_s 在 M ≤ 1 時 ≥ 0；M > 1 時對某些 k 為負。
    - Transfer matrix 的特徵函數對權重 diag(B, B) 正交，所有特徵值都是實數。
    - 負的 η 給出每走一步就變號的項。
    - 這些負模態位在截斷尺度，是「無害的」。
    - 處理方式：擬合時加一項 Ã(−1)^t e^{−m̃t}。他們也建議依作用量選擇 M，讓這些模態不出現。
  - **沒有討論：**譜權重的正負、reflection positivity、Möbius。
  - 我的推論（標明為推論）：M > 1 時 B 不定號，代表「權重矩陣」度規不定號。因此 Lüscher 式的 Hilbert 空間不再明顯為正，權重非負並*未*被這個構造保證；但它也沒有證明權重會變負。
- **R. S. Sufian, M. J. Glatzmaier, Y.-B. Yang, "Unphysical Poles of Domain Wall Fermions at finite Ls"**，arXiv:1603.01591（未列期刊資訊）**[已查證摘要]**。
  - 分析 Shamir、Boriçi、Möbius 的自由 4D 傳播子極點。
  - 振盪來自非物理極點。他們的處方是 M·a₅ < 1（Shamir／Boriçi）或 M(b₅ − c₅) < 1（Möbius）。
  - 注意：RBC 48I/64I 的 b − c = 1，所以在 M5 = 1.8 時這個條件不成立。
- **M. Tomii, "Unphysical poles and dispersion relations for Möbius domain-wall fermions in free field theory at finite Ls"**，PRD 96 (2017) 074504，arXiv:1706.03099 **[已查證]**。
  - 有 Ls − 1 個多餘的極點。b − c 小時這些極點的質量變大，這就是 (b, c) 改變圖像的方式。
  - 沒有處理：權重或正定性。
- **W.-L. Chen, "Unphysical pole in lattice QCD"**，SOKENDAI 博士論文，2025-09-26 授予（甲第2603號）**[已查證，讀過全文]**。典藏紀錄：https://ir.soken.ac.jp/records/2000386
  - 自由場留數分析：在 M5 = 1.0 與 1.8、Ls = 8 下，對 Shamir、Möbius (1.5, 0.5)、Boriçi 計算 r(t) = C_phys/C。
  - 用 shifted Chebyshev 的界 |⟨T*_N⟩| ≤ 1 作為 λ < 0 的偵測器。
  - 資料：M5 = 1.8 的 JLQCD 與 quenched 組態，stout smearing 0、3、6 步。
  - **與你的新想法的關係：**論文是以 Chebyshev 形式檢驗 [0, 1] 的 Hausdorff 假設，*並沒有*檢驗 [−1, 1] 假設是否成立。奇數歸一化（2t₀ = 1）的問題見 §0.2。
  - 背景：Boyle 等人的 RBC/UKQCD inclusive τ |Vus| 工作（arXiv:1803.07228，PRL 121 (2018) 202003，**[已查證]**）是對格點 HVP 使用色散權重，不是譜重建。這個做法完全不需要譜的正定性，大概就是當時 M5 = 1.8 從來不成問題的原因。

### 1.2 Reflection positivity 與 transfer matrix（背景）

- M. Lüscher, CMP 54 (1977) 283：為帶 Wilson 費米子的 Wilson 格點規範理論構造自伴且嚴格正定的 transfer matrix。透過 Syritsyn–Negele 的參考文獻確認了這個引用。原文中確切的 κ 範圍是 **[憑記憶]**：我記得是 κ < 1/6 時正定，但沒有查證。
- P. Menotti, A. Pelissetto, "General proof of Osterwalder–Schrader positivity for the Wilson action"，CMP 113 (1987) 369–373。
  - 把 Osterwalder–Seiler 推廣到 site reflection。
  - 二手資料說證明在 K < 1/6 時成立；我打不開 PDF（**[部分查證]**）。
- K. Osterwalder, E. Seiler, Ann. Phys. 110 (1978) 440：帶 Wilson 費米子的格點規範理論的 link-reflection positivity。卷號與頁碼為 **[憑記憶]**。
- Y. Kikukawa, K. Usui, "Reflection positivity of free overlap fermions"，PRD 82 (2010) 114503，arXiv:1005.3751 **[已查證]**：自由 overlap 費米子的 link-reflection positivity。
- Staggered 費米子是「Hermitian 但不正定」的教科書例子（λ < 0，帶 (−1)^t 的相反宇稱態）。Hackett–Wagman（arXiv:2412.04444）與 Abbott–Jay–Oare（arXiv:2508.01377）都提到這點；Abbott–Jay–Oare 把它對應到 Hamburger moment 問題。
- 套用到你的情況的邏輯：
  - 只有 link reflection positivity（對時間切片之間的平面做 OS 反射）時，得到正定的 Hilbert 空間與 Hermitian 的壓縮算子 T，且 ⟨F|T^{t−1}|F⟩ = C(t)。
  - 再加上 site reflection positivity，才有 T ≥ 0。
  - 所以「λ < 0 但權重 ≥ 0」恰好就是只有 link-RP 的情況，並帶有 §0.2 所說的奇數偏移。
  - 5d DWF／Möbius 在 M5 > 1（含 Pauli–Villars 場與 c ≠ 0）是否具有 link RP，就我所查，是**未解問題**。

### 1.3 [−1, 1] 上的 moment 問題條件（精確）

來源：K. Schmüdgen, "Ten Lectures on the Moment Problem"，arXiv:2008.12698，第 3 講，定理 3.1–3.2 **[已查證，讀過內文]**。內容依循專書：Schmüdgen, *The Moment Problem*, GTM 277, Springer 2017, 第 10 章。

對區間 [a, b]，偶數情形 m = 2n：序列 s₀..s₂ₙ 是某個正 Radon 測度的截斷 [a,b] moment 序列，**若且唯若**：

- (s_{i+j})_{i,j=0..n} ⪰ 0，且
- ((a+b)s_{i+j+1} − ab·s_{i+j} − s_{i+j+2})_{i,j=0..n−1} ⪰ 0。

奇數情形 m = 2n+1：(s_{i+j+1} − a s_{i+j})_{0..n} ⪰ 0 且 (b s_{i+j} − s_{i+j+1})_{0..n} ⪰ 0。

代入 [a, b] = [−1, 1]：

| 情形 | 條件 1 | 條件 2 |
|---|---|---|
| 偶數 2n | (s_{i+j})_{0..n} ⪰ 0 | (s_{i+j} − s_{i+j+2})_{0..n−1} ⪰ 0 |
| 奇數 2n+1 | (s_{i+j} + s_{i+j+1})_{0..n} ⪰ 0 | (s_{i+j} − s_{i+j+1})_{0..n} ⪰ 0 |

作為對照，舊的 [0, 1] 假設（偶數情形）是 (s_{i+j}) ⪰ 0 與 (s_{i+j+1} − s_{i+j+2}) ⪰ 0。

同一講的其他結果：

- 位於 moment cone 內部 ⇔ 兩個矩陣都*正定*。
- 每個內部點恰有兩個 **principal** 表示測度（上／下；就是 Gauss–Radau 型求積）。
- 對每個 ξ ∈ (a, b)，恰有一個在 ξ 有原子的 **canonical** 測度。Chebyshev–Markov–Stieltjes 的極值界限就是由它決定的。
- 該講也涵蓋唯一性與邊界點。

### 1.4 由 moment 得到的嚴格界限

- **Golub & Meurant**，*Matrices, Moments and Quadrature with Applications*，Princeton University Press 2010 **[已查證書目與目錄：第 7 章，雙線性型的界]**。
  - 若 f 在 [a, b] 上各階導數符號固定，Gauss 給出一側的界，Gauss–Radau 或 Gauss–Lobatto 給出另一側。
  - 原始論文 Golub & Meurant 1994 "Matrices, moments and quadrature"（Pitman Res. Notes Math. 303）為 **[憑記憶]**。
  - **注意：**inclusive kernel（θ 函數相空間、smeared 階梯函數）的導數符號*不*固定，Golub–Meurant 不能直接套用。
- **Chebyshev–Markov–Stieltjes（Markov–Krein）不等式。**
  - 給定正測度到 2n 次的 moment，分布函數 μ([a, x]) 會被 x 以下節點的 Gauss（canonical）求積權重部分和夾住。
  - 最佳的極值由在 x 有原子的 canonical 測度達成（Schmüdgen 定理 3.9；Krein–Nudel'man 1977；Karlin–Studden 1966）。
  - 標準教科書有 Akhiezer (1965)、Szegő *Orthogonal Polynomials*、Freud *Orthogonal Polynomials*。具體定理編號為 **[憑記憶]**。
  - 歷史回顧：arXiv:math/9307220 **[已查證]**。微分形式版本：Gilboa–Peled，J. Approx. Theory (2015)（期刊細節 **[憑記憶]**）。
- **一般 kernel。**
  - 在所有符合資料、支撐在 [−1, 1] 的正測度 μ 上，∫K dμ 的最佳上下界是 moment cone 上的線性規劃。K 為多項式或分段多項式時，其對偶問題是 SDP（Markov–Lukács 證書）。
  - 近期格點上正是這種做法的版本，但支撐取能量的 ℝ⁺，也就是 λ ∈ (0, 1]：
    - Abbott, Fields, Jay, Oare, Saccardi, "The Causal Bootstrap"，arXiv:2605.20509（2026 年 5 月）**[已查證]**。
    - Mutzel, Tilloy, "Certified spectral functions from lattice Monte Carlo data"，arXiv:2606.09791（2026 年 6 月）**[已查證]**；明確使用 reflection positivity。
    - Abbott, Jay, Oare，arXiv:2508.01377（2025 年 8 月）**[已查證]**：[0,1] 上的 Hausdorff、熱態用 Stieltjes、staggered 的 λ < 0 用 Hamburger。給出嚴格的 Weyl-ball 界限，並提醒來自 λ < 0 與 λ > 1 態的 "bleed-in"。
  - 把上述任何一個改成 [−1, 1] 上的 Hausdorff 都只是機械性的修改：把 localizing 矩陣換成 (1 − λ²)。
- **以 λ 為變數的 Nevanlinna。**
  - 對 [−1, 1] 上的正測度 μ，F(w) = ∫dμ(λ)/(w − λ) = Σ_k s_k w^{−k−1} 在 [−1, 1] 以外是 Markov／Nevanlinna 函數，所以 Nevanlinna–Pick 與 Padé 界限在那裡自然適用。
  - 只有*以能量為變數*的表述才需要假設 λ > 0。
  - 這是標準的 moment 理論（Markov 關於 Markov 函數 Padé 近似的定理）。我沒有找到以這種方式使用它的格點論文。

### 1.5 現有 inclusive 方法：是否假設 λ ≥ 0？有人用過 [−1, 1] 嗎？

| 方法 | 主要文獻（皆經 arXiv API **[已查證]**） | 假設 λ ∈ (0, 1]？ | 用過 [−1, 1]？ |
|---|---|---|---|
| Chebyshev（shifted T*_n(2λ−1)） | Hashimoto, PTEP 2017, arXiv:1703.01881；Bailas–Hashimoto–Ishikawa, PTEP 2020, arXiv:2001.11779；Gambino–Hashimoto, PRL 125 (2020) 032001, arXiv:2005.13730；Gambino et al., arXiv:2203.11762；Barone et al., JHEP 07 (2023) 145, arXiv:2305.14092；Barone et al., arXiv:2312.17401；Kellermann et al., arXiv:2504.03358 | **是，明文寫出**（「ẑ 的特徵值被限制在 0 與 1 之間」；擬合時要求 \|T̄_j\| ≤ 1）。使用 JLQCD Möbius 資料（2001.11779 未註明 M5） | 沒有。論文只把 \|⟨T*⟩\| > 1 當作*偵測器* |
| Backus–Gilbert | Hansen–Meyer–Robaina, PRD 96 (2017) 094513, arXiv:1704.08993 | 是（E ≥ E₀ 的 e^{−tE} 基底） | 沒有 |
| HLT | Hansen–Lupo–Tantalo, PRD 99 (2019) 094508, arXiv:1903.06476；τ：Evangelista et al., arXiv:2308.03125；Alexandrou et al., arXiv:2403.05404；staggered：Blum, Jay, Jin, Kronfeld, Stewart, arXiv:2411.14300 | 是。A[g] 泛函只控制 E ≥ E₀，λ < 0 的貢獻 Σ g_t λ^t 不受控制。staggered 那篇討論了「處理相反宇稱效應的策略」（只讀了摘要） | 沒有 |
| Nevanlinna／保角映射 | Fei–Yeh–Gull, PRL 126 (2021) 056402, arXiv:2010.04572；Bergamaschi–Jay–Oare, arXiv:2305.16190 | 是（能量上的譜密度為正） | 沒有 |
| Lanczos／moment 界限 | Wagman, arXiv:2406.20009（假設 T Hermitian 且正定；負的 Ritz 值當作偽值剔除）；Hackett–Wagman, PRD 112 (2025) 014514, arXiv:2412.04444；Abbott–Jay–Oare, arXiv:2508.01377 | 大多是。Abbott–Jay–Oare 用 Hamburger 處理 λ < 0 | 用 Hamburger（ℝ），不是 [−1, 1] 的 Hausdorff |
| 回顧 | Hashimoto, "Smeared spectral functions from lattice QCD: opportunities and challenges", arXiv:2609.02239 | （只讀了摘要） | — |

**總結：**「[−1, 1] 上的 Hausdorff + [−1, 1] 上的 Chebyshev + moment 界限」這套方案，對 inclusive 衰變來說看起來是新的。它建立在一個未經證明、但可以直接檢驗的正定性性質上。

### 1.6 關於 [−1, 1] Chebyshev 想法的實務備註

- **光滑性。**K(λ) 在 λ < e^{−a m_τ} 時為零，所以在 [−1, 0) 補 0 的延拓在 λ = 0 處是光滑的。不光滑之處在閾值（τ kernel 以 (1 − s/m_τ²)² 的方式趨於零，因此是 C¹），這點與原本相同。
- **解析度的代價。**[−1, 1] 上的 T_N(λ) 在 λ 方向的解析度只有 [0, 1] 上 T*_N(2λ−1) 的一半。要達到相同的截斷誤差，大約需要 2 倍的多項式階數。
  - 另一種做法：只在 [0, 1] 上逼近 K，讓 |p| 在 [−1, 0) 上有界，再用 max|p| · μ([−1, 0)) 界住 λ < 0 的部分。μ([−1, 0)) 本身可以由 moment 透過 CMS／canonical 測度來界定。
- **有限 T。**週期格子上的 C(t) 是熱跡（thermal trace），繞一圈的項 ∝ λ_m^{N_t − t} 在 λ < 0 時會混入不同正負號。
  - 在 48I，e^{−a m_π N_t} ≈ e^{−0.080·96} ≈ 5×10⁻⁴，以 Hankel 行列式的精度來看不能忽略。
  - 取 t ≪ N_t/2，或先扣掉反向傳播的基態。
- **資料本身無法證明正定性，但可以否定它。**一個顯著為負的 Hankel pivot 就足以否定該算符的假設；通過檢驗只代表相容。
  - 決定性的理論檢驗是自由場：計算精確的自由 Möbius（b=1.5、c=0.5、Ls=12、M5=1.8）V–V 關聯函數，沒有雜訊，再套用 §1.3。論文已經有自由 4D 傳播子的工具（論文第 3 章；你 `postive.tex` 中的圍道積分程式）。

---

## 2. 本機上的舊資料（唯讀；沒有改動任何東西）

### 2.1 M5 > 1：RBC/UKQCD 48I / 64I 的 ūs 流（Hankel 檢驗的主要候選）

- **路徑（完好副本）：**`~/BetterData/RBCQCD/`（16 個檔案，275 MB）。
  - 位元組完全相同的副本（md5 一致）：`~/Data/`、`~/Data/Lattice/Data/`、`~/Data/Lattice/Data/Second/`、`~/Data/Lattice/64c/`。
  - `~/Data/Lattice/48c/` 只有 gamma 2、11、13、14 完好。**那裡的 gamma 1、4、7、8 會出現 I/O 錯誤。**
  - OneDrive 的 `Desktop\Data\Lattice` 與 `Desktop\LatticeTau\Data\Lattice`（各 280 MB）看起來也是副本，我沒有計算雜湊值。
- **檔案：**`corr_Z2_l48t96_Ls12_m0.00078_LL_m0.0362_LL_{g}_{g}` 與 `corr_Z2_l64t128_Ls12_m0.000678_LL_m0.02661_LL_{g}_{g}`，g ∈ {1, 2, 4, 8, 7, 11, 13, 14}，日期皆為 2015-12-20。
  - Gamma bitmask 的解讀（推論，與 `ConfigCreator.py` 和你的筆記一致）：1、2、4、8 = γx、γy、γz、γt（向量）；14、13、11、7 = A_x、A_y、A_z、A_t（∝ γ_μγ5）。
  - 味：輕夸克（0.00078／0.000678）× 奇異夸克（0.0362／0.02661），即 ūs 通道。Local-local 流，Z2 雜訊源。
- **格式：**ASCII，8 欄 `t Γsrc Γsnk 0 0 0 Re Im`，每 N_t 行為一個區塊。
  - 48I：405,504 行 = 88 組態 × 48 源 × 96。
  - 64I：327,680 行 = 40 組態 × 64 源 × 128。
  - 組態 × 源的拆分來自 `BetterData/ConfigCreator.py`。但你的 `finalv120230501.tex` 表格寫的是 40/88，而 arXiv:1411.7017 分析的是 80（48I）與 40（64I）個組態。做 jackknife 前請先確認。
- **Ensemble**（arXiv:1411.7017，PRD 93 (2016) 074505，**[已查證]**）：48I 為 48³×96、Ls(sea) = 24、β = 2.13、a⁻¹ = 1.730 GeV；64I 為 64³×128、Ls = 12、β = 2.25、a⁻¹ = 2.359 GeV。兩者都是 Iwasaki，Möbius 且 b+c = 2、b−c = 1（即 b = 1.5、c = 0.5）。
  - 你 `tau.bib` 中引用的資料來源：Boyle et al., PRD 93 (2016) 054502, arXiv:1511.01950 **[已查證]**。
- **M5 的不確定性。**
  - 論文寫 M5 = 1.8 是 RBC/UKQCD 的選擇。我在讀過的 1411.7017 段落中找不到 M5 的數值。
  - 你自己的 `OneDrive\Desktop\LatticeTau\postive.tex` 表格中，這份價夸克資料列的是 **M5 = 1.6、Ls = 12、b = 1.5、c = 0.5**。
  - 48I 的價夸克 Ls = 12（海夸克 Ls = 24），暗示價夸克用的是不同的作用量（zMöbius？）。
  - **無論如何 M5 > 1。**請向資料提供者確認價夸克作用量。
- 部分淬火（partially quenched）48I：`~/Data/Lattice/48c-pq-run0/{1040,1360,1480,1960,2120}/`，5 個組態，每個 256 個 `.cor` 檔。
  - 質量 (0.0006979, 0.0358) 與 (0.00078, 0.0362)，兩種排列順序，{G0..G3, G0G5..G3G5} 的所有 8×8 Γ 組合。
  - 格式：96 行，`t Re Im`。11 MB。
- 64I 部分淬火：`~/Data/Lattice/64c-pq/{temporal,time_only}_{unitary,pqMunitary}_{v,a}_ls/{1380,1820,1900,2660,2700}`，2 欄，196 KB。
- 衍生產物：`~/Data/Lattice/Data/{4811,…,645454}`（源平均後，3 欄）、`COV*`、`JACKMEAN*`；`~/Data/*.csv`（`Average_*`、`AB48*.csv` 等）；`~/BetterData/CovofFourEight*/CovofSixFour*.csv`；實驗 τ 資料在 `~/Data/Experiment/`（Belle Kπ 等）。

### 2.2 M5 = 1.0：JLQCD Möbius DWF

- `~/BetterData/data_and_fits-002/`（16 GB，日期 2016–2017）。15 個 ensemble，命名為 `SymDW_sHtTanh_b2.0_smr3_{L}x{T}x{Ls}_b{β}_M1.00_mud{…}_ms{…}`：
  - β = 4.17：32³×64×12（7 個；mud 0.0035–0.019；ms 0.030／0.040）與 48³×96×12（mud 0.0035）。
  - β = 4.35：48³×96×8（6 個；mud 0.0042／0.0080／0.0120；ms 0.0180／0.0250）。
  - β = 4.47：64³×128×8（mud 0.0030、ms 0.0150）。
- 每個 ensemble 有 `data/`（約 927 個檔案）以及許多擬合資料夾。
  - 通道：PP、A4A4、A4P、PA4、SS、V1V1、V2V2、V3V3、vectorave。味：ud-ud、ud-s、s-s、heavy-{ud, s, heavy}。smearing 組合 0-0、0-1、1-0、1-1。
  - 格式：CSV `t, value`，各組態依序串接。48×96：4800 行 = 50 組態 × 96；32×64：6400 = 100 × 64；64×128：12800 = 100 × 128。
  - **沒有空間方向的軸向流 A_iA_i**，所以這裡做不出 τ kernel 中 J = 1 的軸向部分。
- 同一批 ensemble 的合併 Z2 雜訊資料與擬合：`~/BetterData/drive-download-20230531T042218Z-001/`（合併資料 443 MB、擬合 363 MB）。另有 `raw_data_one_ensemble/`（247 個 `hadronCorrel_*` 文字檔）與 IroIro 的 `python_analysis/`。
- `~/BetterData/Ryan/data2Pt_Ryan_mud{0.0042,0.0080,0.0120}.h5` 與 `~/Data/data2Pt_Ryan_NewSmearing14.14.h5`：HDF5，群組 LL/LS/SL/SS，各含形狀 (400, 96) 的 Re/Im。可能是 β = 4.35 的 48³×96 ensemble；通道沒有標示。

### 2.3 你論文中自己做的 M5 掃描（Bridge++）

- `~/BetterData/BRIDGE/`（`M51.0Mobius.csv`、`M51.8Mobius.csv`、`M51.0Borici.csv`、`M51.8Borici.csv`、`M51.8Mobius{JLQCD,Quench}SM{0,3,6}[B].csv`、`PSPS.csv`、`A4PS.csv`）、`~/BetterData/20250504/`（`M510Mobius`、`M518Mobius`、`M518b15c05Rindou`）、`~/BetterData/Aurona/`（`M510BSC*`、`M516BSC*`）、`~/BetterData/Peagsus/`（Bridge++ 輸出紀錄，32³×64，M0 = 1.8）、`~/Bridge/Neff/eigentest/`。
- 大多是贗純量關聯函數，組態為 32³×64 JLQCD 或 16³×32 quenched。
- 格式：每個組態 `t Re Im`，部分檔案有 `# Conf` 標頭。
- 這些資料直接**在相同規範場上比較 M5 = 1.0 與 1.8**，非常適合做對照的 Hankel 比較（P–P 與 V–V 在同樣意義下具有 reflection positivity）。

### 2.4 Windows 端

- `C:\Users\atsan\OneDrive\Desktop\LatticeTau\`：
  - 草稿：`finalv120230501.tex/pdf`（"The tau inclusive hadronic decay from lattice QCD"：Chebyshev、exclusive mock 檢驗、Vus）與 `postive.tex/pdf`（"Detecting the unphysical contribution of domain-wall fermion by Chebyshev approximation"，含一節 "Positivity check by ⟨T̄_i⟩"）。
  - `tau decay*.tex`、Mathematica 筆記本（`ChebAppTestPlot.nb`、`KernelApproximation.nb` 等）、`tau.bib`，以及一份資料副本（280 MB）。
- `C:\Users\atsan\OneDrive\Desktop\Data\`：另一份副本與分析程式。
- 與博士研究無關：`C:\Users\atsan\ruttice_ckpt`（3 個 1.7 GB 的 RutticeQCD checkpoint）、`wsl_offload`（Nsight 效能分析檔）、`JOB`（PDF）。
- `Downloads` 有 41 GB 的雜項檔案，我沒有深入搜尋；整個家目錄的關鍵字搜尋也在完成前被我停止了，所以 Windows 端的盤點並不完整。
- 沒有深入打開的 WSL 資料夾：`~/hashimoto`（autotools 原始碼樹，84 MB）、`~/Bridge`（`Neff/eigentest` 以外）、`~/RutticeQCD*`（依指示略過 `run/` 與 `target*/`）。

---

## 3. 提議的 Hankel 正定性檢驗（尚未執行）

資料：`~/BetterData/RBCQCD/` 的 48I 與 64I，通道 V(J=1) = g = 1、2、4 的平均；A(J=1) = g = 14、13、11 的平均；V(J=0) = 8；A(J=0) = 7。
再用 `BRIDGE/` 中 M5 = 1.0 與 1.8 的配對做為對照組。

1. 先對源取平均，再對組態做 jackknife（有自相關就先分 bin）。
   - 固定整體正負號，使 C 在平台區為正。
   - 只使用 t ≤ N_t/2 − 餘裕。
2. 對每個起始時間 t_s ∈ {1, 2, …, 8}（**奇偶分開**，見 §0.2）與 n = 1..4，令 s_k = C(t_s + k)/C(t_s)，k = 0..2n。
3. 建立三組矩陣：
   - **H0** = (s_{i+j})_{0..n}：測度為正，任何支撐都必須滿足。
   - **H[−1,1]** = (s_{i+j} − s_{i+j+2})_{0..n−1}：支撐在 [−1, 1]。
   - **H[0,1]** = (s_{i+j+1} − s_{i+j+2})_{0..n−1}：舊的 λ ≥ 0 假設。另外也檢查 (s_{i+j+1})_{0..n−1} ⪰ 0，它與 H0 一起構成支撐在 [0, ∞) 的 Stieltjes 條件。
4. 把每個矩陣縮放成對角線為 1（這是合同變換，不改變半正定性）。回報最小特徵值與 LDLᵀ pivot，附 jackknife 誤差。注意 jackknife 在零附近有偏差：用平均值上的 pivot ± jackknife σ 來報告。
5. 若你的假設成立，預期的模式是：
   - 對某一種 t_s 奇偶，H0 與 H[−1,1] 在所有 n 都 ⪰ 0。
   - H[0,1]（可能還有另一種奇偶的 H0）在 t_s 小、n ≥ 2 時顯著不成立。
   - M5 = 1.0 時，兩種奇偶全部通過。
6. 自由場精確檢驗（強烈建議在相信資料之前先做）：在 48I/64I 參數、M5 = 1.8 下計算自由 Möbius 的 V–V。沒有雜訊，任何負 pivot 都具決定性。
7. 若檢驗通過，下一步是 ∫K dμ 的 [−1, 1] SDP 界限：即 §1.4 的對偶問題，localizing 多項式取 (1 − λ²)。

程式骨架在 `hankel_test_proposal.py`（路徑與組態 × 源的拆分都是參數，請先核對）。
