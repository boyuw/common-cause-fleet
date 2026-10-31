"""Figures for the paper.  Reads results/summary.json only.

  python3 make_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
if not FIGS.is_dir():
    FIGS.mkdir(parents=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C = {"same_model": "#9E2A2B", "same_family": "#D98C4A", "cross_family": "#2E5E8C"}
LBL = {"same_model": "same model", "same_family": "same family,\ndifferent size",
       "cross_family": "cross-family"}
S = json.loads((RESULTS / "summary.json").read_text())


def _num(x, default=np.nan):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Figure 1: failure correlation by relatedness, overall and by stratum
# ---------------------------------------------------------------------------
def fig_phi():
    levels = ["same_model", "same_family", "cross_family"]
    levels = [l for l in levels if l in S["phi"] and np.isfinite(_num(S["phi"][l]["phi_pooled"]))]
    if not levels:
        print("fig_phi: no data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.25),
                             gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.55))

    # (a) grouped bars: all / easy / medium / hard
    ax = axes[0]
    cats = ["all", "easy", "medium", "hard"]
    alpha = {"all": 1.0, "easy": 0.62, "medium": 0.45, "hard": 0.28}
    width = 0.19
    for gi, lev in enumerate(levels):
        for ci, cat in enumerate(cats):
            if cat == "all":
                v = _num(S["phi"][lev]["phi_pooled"])
                lo, hi = (_num(x) for x in S["phi"][lev]["phi_pooled_ci"])
            else:
                st = S["phi"][lev]["by_stratum"].get(cat)
                if not st:
                    continue
                v = _num(st["phi"])
                lo, hi = (_num(x) for x in S["phi"][lev]["strata_ci"][cat])
            x = gi + (ci - 1.5) * width
            ax.bar(x, v, width * 0.9, color=C[lev], alpha=alpha[cat],
                   edgecolor="white", linewidth=0.4, zorder=3)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([x, x], [max(lo, 0), hi], color="0.25", lw=0.8, zorder=4)
            if cat == "all" and np.isfinite(hi):
                ax.annotate(f"{v:.2f}", (x, hi + 0.03), ha="center", fontsize=7.2,
                            color="0.1")
    ax.axhline(0, color="0.55", lw=0.7, zorder=1)
    ax.axhline(0.30, color="0.35", lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([LBL[l] for l in levels], fontsize=7.5)
    ax.set_ylabel("failure correlation $\\varphi$")
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlim(-0.55, len(levels) - 0.45)
    ax.set_title("(a) pairwise failure correlation", fontsize=8.5, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.35", alpha=alpha[c]) for c in cats]
    ax.legend(handles, ["all", "easy", "medium", "hard"], fontsize=6.3, ncol=2,
              loc="upper right", handlelength=0.9, columnspacing=0.8,
              borderaxespad=0.15, title="difficulty", title_fontsize=6.3)

    # (b) decomposition: raw vs difficulty-only vs MH-adjusted
    ax = axes[1]
    ys = np.arange(len(levels))[::-1]
    for y, lev in zip(ys, levels):
        raw = _num(S["phi"][lev]["phi_pooled"])
        rlo, rhi = (_num(x) for x in S["phi"][lev]["phi_pooled_ci"])
        ind = _num(S["phi"][lev]["phi_indep_mix"])
        ilo, ihi = (_num(x) for x in S["phi"][lev]["phi_indep_mix_boot_ci"])
        mh = _num(S["phi"][lev]["phi_mh"])
        mlo, mhi = (_num(x) for x in S["phi"][lev]["phi_mh_ci"])
        ax.plot([max(ilo, 0), ihi], [y + 0.19] * 2, color="0.6", lw=0.9, zorder=2)
        ax.plot(ind, y + 0.19, "o", ms=4.0, mfc="white", mec="0.45", mew=0.9, zorder=4)
        ax.plot([max(rlo, 0), rhi], [y, y], color=C[lev], lw=1.3, zorder=2)
        ax.plot(raw, y, "o", ms=5.0, color=C[lev], zorder=5)
        ax.plot([max(mlo, 0), mhi], [y - 0.19] * 2, color="0.2", lw=0.9, zorder=2)
        ax.plot(mh, y - 0.19, "s", ms=3.8, color="0.2", zorder=4)
    short = {"same_model": "same\nmodel", "same_family": "same family,\ndiff. size",
             "cross_family": "cross-\nfamily"}
    ax.set_yticks(ys)
    ax.set_yticklabels([short[l] for l in levels], fontsize=6.8)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(-0.12, 1.02)
    ax.set_ylim(-0.55, len(levels) - 0.45)
    ax.set_xlabel("$\\varphi$")
    ax.set_title("(b) what the difficulty mix explains", fontsize=8.5, loc="left")
    ax.plot([], [], "o", color="0.35", ms=4.5, label="raw pooled")
    ax.plot([], [], "o", mfc="white", mec="0.45", ms=4.2, label="difficulty only")
    ax.plot([], [], "s", color="0.2", ms=3.6, label="Mantel--Haenszel")
    ax.legend(fontsize=6.2, loc="lower right", handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.25)

    fig.savefig(FIGS / "fig1_phi.pdf")
    fig.savefig(FIGS / "fig1_phi.png", dpi=220)
    plt.close(fig)
    print("wrote figures/fig1_phi.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 2: realised vs nominal redundancy
# ---------------------------------------------------------------------------
def fig_redundancy():
    configs = [c for c in ["same_model", "same_family", "cross_family"]
               if c in S["panels"] and np.isfinite(_num(S["panels"][c]["per_k"]["5"]["error"]))]
    if not configs:
        print("fig_redundancy: no data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3),
                             gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.3))
    ax = axes[0]
    ks = [1, 3, 5]
    for c in configs:
        real = [_num(S["panels"][c]["per_k"][str(k)]["error"]) for k in ks]
        ind = [_num(S["panels"][c]["per_k"][str(k)]["indep_stratified"]) for k in ks]
        lo = [_num(S["panels"][c]["per_k"][str(k)]["error_ci"][0]) for k in ks]
        hi = [_num(S["panels"][c]["per_k"][str(k)]["error_ci"][1]) for k in ks]
        ax.fill_between(ks, lo, hi, color=C[c], alpha=0.10, lw=0)
        ax.plot(ks, real, "-o", color=C[c], ms=4.0, lw=1.5, zorder=5,
                label=LBL[c].replace("\n", " "))
        ax.plot(ks, ind, color=C[c], lw=1.1, alpha=0.9, zorder=4,
                dashes=(2.2, 1.6))
    ax.set_xticks(ks)
    ax.set_xlim(0.85, 5.15)
    ax.set_ylim(0, 0.46)
    ax.set_xlabel("panel size $k$ (members)")
    ax.set_ylabel("panel error rate")
    ax.set_title("(a) realised (solid) vs independence (dashed)", fontsize=8.5, loc="left")
    ax.legend(fontsize=6.4, loc="upper left", handlelength=1.9, borderaxespad=0.2,
              labelspacing=0.25)

    ax = axes[1]
    xs = np.arange(len(configs))
    keffs = [_num(S["panels"][c]["k_eff"]) for c in configs]
    ax.bar(xs, keffs, 0.55, color=[C[c] for c in configs], zorder=3)
    ax.axhline(5, color="0.3", lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.annotate("nominal $k=5$", (len(configs) - 0.52, 5.15), fontsize=6.4,
                color="0.3", ha="right")
    for x, c, ke in zip(xs, configs, keffs):
        if np.isfinite(ke):
            ax.annotate(f"{ke:.1f}", (x, ke + 0.12), ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL[c].replace("\n", "\n") for c in configs], fontsize=7)
    ax.set_ylabel("effective redundancy $k_{\\mathrm{eff}}$")
    ax.set_ylim(0, 6.2)
    ax.set_title("(b) redundancy actually bought", fontsize=8.5, loc="left")

    fig.savefig(FIGS / "fig2_redundancy.pdf")
    fig.savefig(FIGS / "fig2_redundancy.png", dpi=220)
    plt.close(fig)
    print("wrote figures/fig2_redundancy.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 3: checker catch rate by author-checker relatedness
# ---------------------------------------------------------------------------
def fig_checkers():
    order = ["same-same", "same-cross", "cross-same", "cross-cross"]
    order = [c for c in order if c in S.get("checkers", {})]
    if not order:
        print("fig_checkers: no data")
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    xs = np.arange(len(order))
    vals = [_num(S["checkers"][c]["catch"]) for c in order]
    lo = [_num(S["checkers"][c]["catch_ci"][0]) for c in order]
    hi = [_num(S["checkers"][c]["catch_ci"][1]) for c in order]
    # colour by the checker model: the four cells are two pairs sharing a checker
    checker = {"same-same": "M1", "same-cross": "M3",
               "cross-same": "M1", "cross-cross": "M3"}
    col = {"M1": "#9E2A2B", "M3": "#4F7CA6"}
    cols = [col[checker[c]] for c in order]
    ax.bar(xs, vals, 0.58, color=cols, zorder=3)
    ax.errorbar(xs, vals,
                yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                fmt="none", ecolor="0.2", lw=0.9, capsize=2.5, zorder=4)
    for x, v, h in zip(xs, vals, hi):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}", (x, h + 0.035), ha="center", fontsize=7.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("-", "\n") for c in order], fontsize=7.2)
    ax.set_ylabel("catch rate")
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("author–checker pair", labelpad=2)
    ax.plot([], [], "s", color=col["M1"], ms=5, label="checker M1")
    ax.plot([], [], "s", color=col["M3"], ms=5, label="checker M3")
    ax.legend(fontsize=6.4, loc="upper left", handletextpad=0.4, borderaxespad=0.2)
    fig.savefig(FIGS / "fig3_checkers.pdf")
    fig.savefig(FIGS / "fig3_checkers.png", dpi=220)
    plt.close(fig)
    print("wrote figures/fig3_checkers.{pdf,png}")


if __name__ == "__main__":
    fig_phi()
    fig_redundancy()
    fig_checkers()
