"""Figure 1 — framework and data foundation."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from variantbenchpy.realdata.fignat_common import (
    W2, MM, PRED6, COL, DARK, SRC, FROZEN_COV, letters)


def _box(ax, x, y, w, h, fc="#F2F2F2", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=DARK, lw=0.6, ls=ls))


def _arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, lw=0.6, color=DARK))


def workflow(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    top = [("Data sources", "ClinVar GRCh38 (2026-08-16)\nAlphaMissense; REVEL v1.3\n"
                            "dbNSFP 5.3.1a: CADD,\nSIFT, PolyPhen-2; EVE"),
           ("Harmonization", "exact chr:pos:ref:alt keys\nMANE-informed primary gene\n"
                             "MANE Select transcript rule\nfrozen score orientation"),
           ("Design grid", "evaluation unit \u00d7\ncomposition \u00d7 evidence\n"
                           "2 \u00d7 3 \u00d7 3 = 18 designs\nper predictor layer"),
           ("Reliability audit", "winner retention\nKendall's W\nrank distribution\n"
                                 "gene-clustered bootstrap"),
           ("Selection-aware inference", "K-fold gene cross-fitting\n\u0394\u0394AUROC with Holm\n"
                                         "numerical vs supported\nchange")]
    w, g, y0, h = 0.172, 0.035, 0.40, 0.58
    xs = [i * (w + g) for i in range(5)]
    for i, (hd, bd) in enumerate(top):
        _box(ax, xs[i], y0, w, h)
        ax.text(xs[i] + w / 2, y0 + h - 0.05, hd, ha="center", va="top", fontsize=6.4,
                fontweight="bold")
        ax.text(xs[i] + w / 2, y0 + h - 0.17, bd, ha="center", va="top", fontsize=5.8,
                linespacing=1.35)
        if i < 4:
            _arrow(ax, (xs[i] + w + 0.003, y0 + h / 2), (xs[i + 1] - 0.003, y0 + h / 2))
    bot = [("Temporal exposure-resistant evaluation",
            "germline First-in-ClinVar dates; publication-date proxies"),
           ("Independent experimental validation",
            "ProteinGym v1.3 DMS \u2192 MANE GRCh38 SNVs; MaveDB check")]
    bx = [xs[1] + 0.02, xs[3] - 0.005]
    bw = [xs[3] - xs[1] - 0.045, xs[4] + w - xs[3] + 0.005]
    for (hd, bd), x, ww in zip(bot, bx, bw):
        _box(ax, x, 0.0, ww, 0.27, fc="white", ls=(0, (3, 1.5)))
        ax.text(x + ww / 2, 0.19, hd, ha="center", va="center", fontsize=6.2, fontweight="bold")
        ax.text(x + ww / 2, 0.08, bd, ha="center", va="center", fontsize=5.8)
    for k in (2, 3):
        _arrow(ax, (xs[k] + w / 2, y0 - 0.005), (xs[k] + w / 2, 0.275))


def fig1(B, Q):
    n, npl = len(B), int((B["label"] == 1).sum())
    Q.check("F1 cohort size 225,208", n == 225208, f"{n:,}")
    Q.check("F1 P/LP 71,926", npl == 71926, f"{npl:,}")
    cov = {p: float(B[p].notna().mean()) for p in PRED6}
    for p in PRED6:
        Q.close(f"F1 coverage {p}", cov[p], FROZEN_COV[p], 6e-4)
    comp = pd.crosstab(B["label"], B["stars"].clip(upper=3))
    Q.check("F1 star totals sum to cohort", int(comp.values.sum()) == n, f"{int(comp.values.sum()):,}")

    fig = plt.figure(figsize=(W2, 112 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], width_ratios=[1, 1.2], left=0.115,
                          right=0.985, top=0.955, bottom=0.10, hspace=0.42, wspace=0.42)
    a = fig.add_subplot(gs[0, :]); workflow(a)

    b = fig.add_subplot(gs[1, 0])
    order = sorted(PRED6, key=lambda p: cov[p])
    yy = np.arange(len(order))
    b.barh(yy, [100 * cov[p] for p in order], color=[COL[p] for p in order], height=0.62)
    for i, p in enumerate(order):
        b.text(100 * cov[p] + 1.5, i, f"{100 * cov[p]:.1f}", va="center", fontsize=6)
    b.set_yticks(yy, order); b.set_xlim(0, 112); b.set_xticks([0, 25, 50, 75, 100])
    b.set_xlabel("Coverage of 225,208 ClinVar variants (%)")
    b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)

    c = fig.add_subplot(gs[1, 1])
    shades = ["#E5E5E5", "#B5B5B5", "#7A7A7A", "#3A3A3A"]
    labs = ["0 stars", "1 star", "2 stars", "\u22653 stars"]
    rows = [(1, "P/LP"), (0, "B/LB")]
    for j, (lab, _) in enumerate(rows):
        left, tot = 0, comp.loc[lab].sum()
        for k in range(4):
            v = comp.loc[lab].get(k, 0)
            c.barh(j, v / 1000, left=left / 1000, color=shades[k], height=0.55,
                   label=labs[k] if j == 0 else None, ec="white", lw=0.4)
            if v / tot > 0.08:
                c.text((left + v / 2) / 1000, j, f"{100 * v / tot:.0f}%", ha="center",
                       va="center", fontsize=5.8, color="white" if k >= 2 else DARK)
            left += v
        c.text(left / 1000 + 2, j, f"n = {int(tot):,}", va="center", fontsize=6)
    c.set_yticks([0, 1], [r[1] for r in rows]); c.set_ylim(-0.6, 1.6)
    c.set_xlabel("Variants (thousands)"); c.set_xlim(0, 185)
    c.spines["left"].set_visible(False); c.tick_params(axis="y", length=0)
    c.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.22), handlelength=1,
             columnspacing=0.8, handletextpad=0.4)

    letters(fig, [(a, "a"), (b, "b"), (c, "c")])
    pd.DataFrame({"predictor": PRED6, "coverage": [cov[p] for p in PRED6]}).to_csv(
        os.path.join(SRC, "Fig1b.csv"), index=False)
    comp.to_csv(os.path.join(SRC, "Fig1c.csv"))
    return fig
