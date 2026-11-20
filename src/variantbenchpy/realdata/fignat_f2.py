"""Figure 2 — fixed-design certainty does not imply benchmark-design robustness.

Within a predictor layer the audit restricts to complete cases, so the two coverage
levels of the grid coincide; the 36 audit configurations are 18 distinct designs,
each counted twice. All counts here use the 18 distinct designs.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variantbenchpy.realdata.fignat_common import (
    W2, MM, COL, GREY, LGREY, DARK, SRC, letters)

ORDER = ["REVEL", "AlphaMissense", "CADD", "SIFT", "PolyPhen-2"]
AUC = ["AlphaMissense_AUROC", "REVEL_AUROC", "CADD_AUROC", "SIFT_AUROC", "PolyPhen2_AUROC"]
KEY = ["Evaluation_unit", "Composition", "Evidence"]


def fig2(A, Graw, Q):
    k, nn, ret, w, wlo, whi = A["l5"]
    k6, nn6 = A["l6"][0], A["l6"][1]
    av = Graw[Graw["Coverage"] == "available"].set_index(KEY)[AUC].sort_index()
    sh = Graw[Graw["Coverage"] == "shared"].set_index(KEY)[AUC].sort_index()
    dup = float((av - sh).abs().max().max())
    Q.check("F2 coverage levels identical within layer", dup == 0.0, f"max diff {dup}")
    G = Graw[Graw["Coverage"] == "shared"].copy().reset_index(drop=True)
    nrev = int((G["Winner"] == "REVEL").sum())
    Q.check("F2 distinct designs = 18", len(G) == 18, str(len(G)))
    Q.check("F2 audit 24/36 = 12/18 distinct", (k, nn) == (24, 36) and 2 * nrev == k,
            f"{nrev}/18 vs audit {k}/{nn}")
    Q.check("F2 6-pred audit duplicated (even counts)", k6 % 2 == 0 and nn6 == 36, f"{k6}/{nn6}")
    Q.close("F2 Kendall W (5-pred)", w, 0.91, 1e-9)
    Q.report_new("F2 winner by evaluation unit (18 distinct designs)",
                 pd.crosstab(G["Evaluation_unit"], G["Winner"]).to_dict())

    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.9], left=0.105, right=0.985, top=0.95,
                          bottom=0.095, hspace=0.62, wspace=1.35)

    a = fig.add_subplot(gs[0, 0:2])
    M = np.array([A["dist"][p] for p in ORDER])
    im = a.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(5):
        for j in range(5):
            if M[i, j] > 0:
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.8,
                       color="white" if M[i, j] > 0.55 else DARK)
    a.set_xticks(range(5), [str(r) for r in range(1, 6)]); a.set_yticks(range(5), ORDER)
    a.set_xlabel("Rank across 18 designs")
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(length=0)
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.04)
    cb.set_label("Proportion", fontsize=6); cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_linewidth(0.4)

    b = fig.add_subplot(gs[0, 2:4])
    y = np.arange(5)[::-1]
    for i, p in enumerate(ORDER):
        cfg, bt = A["boot"][p]
        b.plot([cfg, bt], [y[i], y[i]], color=LGREY, lw=1.6, zorder=1)
        b.plot(bt, y[i], "o", color=COL[p], ms=4.2, zorder=3)
        b.plot(cfg, y[i], "o", mfc="white", mec=COL[p], mew=1.0, ms=4.2, zorder=3)
    b.set_yticks(y, ORDER); b.set_xlim(-0.05, 1.05)
    b.set_xlabel("Probability of rank 1")
    b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    b.plot([], [], "o", color=DARK, ms=3.5, label="Bootstrap, fixed design")
    b.plot([], [], "o", mfc="white", mec=DARK, ms=3.5, label="Across 18 designs")
    b.legend(loc="lower right", handletextpad=0.3)

    c = fig.add_subplot(gs[0, 4:6])
    G["gap"] = G["REVEL_AUROC"] - G["AlphaMissense_AUROC"]
    units = [("variant", "Pooled\nvariant"), ("equal-gene", "Equal\ngene")]
    rng = np.random.default_rng(3)
    lim = max(abs(G["gap"].min()), abs(G["gap"].max())) * 1.5
    for xi, (u, _) in enumerate(units):
        g = G.loc[G["Evaluation_unit"] == u, "gap"].to_numpy()
        c.scatter(xi + rng.uniform(-0.17, 0.17, len(g)), g, s=10, lw=0.3, ec="white", zorder=3,
                  c=[COL["REVEL"] if v > 0 else COL["AlphaMissense"] for v in g])
        c.text(xi, lim * 0.97, f"REVEL first\n{int((g > 0).sum())}/{len(g)}", ha="center",
               va="top", fontsize=5.6, color=GREY)
    c.axhline(0, color=GREY, lw=0.6, ls=(0, (2, 2)))
    c.set_xticks([0, 1], [u[1] for u in units]); c.set_xlim(-0.55, 1.55); c.set_ylim(-lim, lim)
    c.set_ylabel("AUROC difference\n(REVEL \u2212 AlphaMissense)")

    d = fig.add_subplot(gs[1, 0:3])
    lay = [("5 predictors", A["l5"]), ("6 predictors", A["l6"])]
    for i, (name, (kk, nm, rr, ww, lo, hi)) in enumerate(lay):
        d.bar(i - 0.17, rr, 0.32, color=LGREY, ec=DARK, lw=0.5,
              label="Winner retention" if i == 0 else None)
        d.bar(i + 0.17, ww, 0.32, color="#5A5A5A", label="Kendall's W" if i == 0 else None)
        d.errorbar(i + 0.17, ww, yerr=[[ww - lo], [hi - ww]], color=DARK, capsize=1.6, lw=0.6)
        d.text(i - 0.17, rr + 0.03, f"{kk // 2}/{nm // 2}", ha="center", fontsize=5.8)
    d.set_xticks([0, 1], [l[0] for l in lay]); d.set_ylim(0, 1.15); d.set_yticks([0, .5, 1])
    d.set_ylabel("Value"); d.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.13))

    e = fig.add_subplot(gs[1, 3:6])
    dims = ["evaluation_unit", "evidence", "composition"]
    nice = ["Evaluation unit", "Evidence", "Composition"]
    yy = np.arange(3)[::-1]
    e.barh(yy, [A["shares"][x] for x in dims], color="#5A5A5A", height=0.55)
    for i, x in enumerate(dims):
        e.text(A["shares"][x] + 1.2, yy[i], f"{A['shares'][x]}%", va="center", fontsize=6)
    e.set_yticks(yy, nice); e.set_xlim(0, 90)
    e.set_xlabel("Share of rank sensitivity (%, descriptive)")
    e.spines["left"].set_visible(False); e.tick_params(axis="y", length=0)

    letters(fig, [(a, "a"), (b, "b"), (c, "c"), (d, "d"), (e, "e")])
    G.to_csv(os.path.join(SRC, "Fig2c.csv"), index=False)
    pd.DataFrame(A["dist"], index=[f"rank{r}" for r in range(1, 6)]).T.to_csv(
        os.path.join(SRC, "Fig2a.csv"))
    return fig
