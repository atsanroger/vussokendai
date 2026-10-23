"""Figures for the note.  Run from the directory that contains RBCQCD/:  python hausdorff/paper/make_figs.py"""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import safe_vus as S
D = os.path.join(os.getcwd(), "RBCQCD")
OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)
NEW, OLD = "#2a78d6", "#eb6834"                    # categorical slots 1 and 2 (validated)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "mathtext.fontset": "cm",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "legend.frameon": False})
TITLE = {"AT": r"$A_T$  ($J=0$)", "VT": r"$V_T$  ($J=0$)", "AS": r"$A_S$  ($J=1$)", "VS": r"$V_S$  ($J=1$)"}

# ---------- Figure 1: Chebyshev coefficients
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
n = np.arange(1, 13)
for ax, ch in zip(axs.flat, S.CHANNELS):
    C = S.load_channel("48", S.CHANNELS[ch][0], D)
    (old, new), (eo, en) = S.cheb_diag(C, 12)
    ax.axhspan(-1, 1, color=GRID, alpha=0.8, lw=0)
    ax.errorbar(n, old, eo, color=OLD, marker="o", ms=5, lw=1.6, capsize=2, label=r"old: $\langle T^*_n\rangle$, support $[0,1]$")
    ax.errorbar(n, new, en, color=NEW, marker="s", ms=5, lw=1.6, ls="--", capsize=2, label=r"new: $\langle T_n\rangle$, support $[-1,1]$")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title(TITLE[ch], fontsize=9, loc="left")
    ax.set_xticks(range(1, 13, 2))
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.8, bbox_to_anchor=(0.5, 1.0))
for ax in axs[1]: ax.set_xlabel(r"$n$")
for ax in axs[:, 0]: ax.set_ylabel("coefficient (symlog)")
fig.text(0.5, 0.005, r"48I. Grey band: $|\langle T\rangle|\leq 1$, required for a positive normalised measure on the stated support.", ha="center", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.94)); fig.savefig(os.path.join(OUT, "fig_cheb.pdf")); plt.close(fig)

# ---------- Figures 2 and 3: bounds versus T
TS = list(range(4, 24, 2)); Z = 3.0
import pickle
CACHE = os.path.join(OUT, ".bounds_cache.pkl")
res = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
for ens in (() if res else ("48", "64")):
    ph = S.PHYS[ens]; f2 = ph["Ina"] ** 2
    grids = {s: S.make_grid(ph["M0"], s) for s in ("pos", "pm1")}
    for ch, (gam, J) in S.CHANNELS.items():
        C = S.load_channel(ens, gam, D); mean, err, _ = S.jk_stats(C, max(TS) + 1)
        for sup, lam in grids.items():
            kap = np.zeros_like(lam); p = lam > 0; kap[p] = S.kernel_lat(-np.log(lam[p]), J, ph["Ina"])
            for T in TS:
                lo, hi = S.bounds(mean, err, lam, T, Z, kap)
                res[(ens, ch, sup, T)] = (lo * f2, hi * f2)
if not os.path.exists(CACHE):
    pickle.dump(res, open(CACHE, "wb"))
kaon_only = 0.1564 ** 2 * (1 - (0.4937 / S.MTAU) ** 2) ** 2
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
for ax, ch in zip(axs.flat, S.CHANNELS):
    nb = np.array([res[("48", ch, "pm1", T)] for T in TS]); ob = np.array([res[("48", ch, "pos", T)] for T in TS])
    ax.fill_between(TS, nb[:, 0], nb[:, 1], color=NEW, alpha=0.28, lw=0, label=r"new, $[-1,1]$")
    ax.plot(TS, nb[:, 0], color=NEW, lw=1.6); ax.plot(TS, nb[:, 1], color=NEW, lw=1.6)
    ok = ~np.isnan(ob[:, 0])
    if ok.any():
        ax.plot(np.array(TS)[ok], ob[ok, 0], color=OLD, lw=1.6, ls="--", marker="o", ms=4, label=r"old, $[0,1]$")
        ax.plot(np.array(TS)[ok], ob[ok, 1], color=OLD, lw=1.6, ls="--", marker="o", ms=4)
    if (~ok).any():
        t1 = np.array(TS)[~ok][0]
        ax.annotate("old: infeasible\nfor $T\\geq %d$" % t1, xy=(0.42, 0.62), xycoords="axes fraction", color=INK, fontsize=7.5)
    if ch == "AT":
        ax.axhline(kaon_only, color=MUTED, lw=1.0, ls=":"); ax.annotate("kaon pole only", xy=(TS[-1], kaon_only), xytext=(-2, 4),
                   textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
        ax.set_ylim(0.012, 0.03)
    ax.set_title(TITLE[ch], fontsize=9, loc="left")
h, l = axs[1, 1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.8, bbox_to_anchor=(0.5, 1.0))
for ax in axs[1]: ax.set_xlabel(r"number of moments $T$")
for ax in axs[:, 0]: ax.set_ylabel(r"$\Sigma$ bounds [GeV$^2$]")
fig.text(0.5, 0.005, r"48I, $z=3$.  Bounds on $\Sigma=\int d\sigma\,\kappa_J/\lambda$ for each channel.", ha="center", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.94)); fig.savefig(os.path.join(OUT, "fig_bounds.pdf")); plt.close(fig)

# ---------- Figure 3: |Vus|
c = S.BS_PLACEHOLDER * S.GAMMA_TAU / (S.PREF * S.SEW)
old_code = {"48": {"orig": [0.5353, 0.18812, 0.27506, 0.27590, 0.2999, 0.2575, 0.342],
                   "phys": [0.3717, 0.17497, 0.24573, 0.25168, 0.26853, 0.2423, 0.316]},
            "64": {"orig": [0.29142, 0.129816, 0.17663, 0.18668, 0.21144, 0.17006, 0.2527],
                   "phys": [0.22237, 0.120132, 0.159473, 0.170091, 0.18803, 0.16189, 0.2375]}}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, ens in zip(axs, ("48", "64")):
    lo = np.array([sum(res[(ens, ch, "pm1", T)][0] for ch in S.CHANNELS) for T in TS])
    hi = np.array([sum(res[(ens, ch, "pm1", T)][1] for ch in S.CHANNELS) for T in TS])
    ax.fill_between(TS, np.sqrt(c / hi), np.sqrt(c / lo), color=NEW, alpha=0.28, lw=0, label=r"new: certified interval")
    ax.plot(TS, np.sqrt(c / hi), color=NEW, lw=1.6); ax.plot(TS, np.sqrt(c / lo), color=NEW, lw=1.6)
    Tx = np.arange(2, 9)
    ax.plot(Tx, old_code[ens]["orig"], color=OLD, marker="o", ms=5, lw=1.2, ls="--", label="old code, as written")
    ax.plot(Tx, old_code[ens]["phys"], color=OLD, marker="s", ms=5, lw=1.2, ls=":", mfc="none", label="old code, time/space kernels")
    ax.axhline(0.2243, color=MUTED, lw=1.0, ls="-."); ax.set_title(f"{ens}I", fontsize=9, loc="left")
    ax.set_xlabel(r"number of moments $T$"); ax.set_ylim(0.10, 0.40)
axs[0].set_ylabel(r"$|V_{us}|$")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.4, bbox_to_anchor=(0.5, 1.0))
axs[1].annotate(r"PDG $0.2243$", xy=(22, 0.2243), xytext=(0, -9), textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
fig.text(0.5, 0.005, r"$B_s=0.0293$ is a placeholder; $|V_{us}|\propto\sqrt{B_s}$. New interval: $z=3$.", ha="center", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.9)); fig.savefig(os.path.join(OUT, "fig_vus.pdf")); plt.close(fig)
print("figures written to", OUT)
