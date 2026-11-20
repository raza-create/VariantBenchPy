"""Figure 3 — complete-case coverage restructures the benchmark."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variantbenchpy.realdata.fignat_common import (
    W2, MM, PRED5, PRED6, COL, GREY, LGREY, DARK, SRC, GENE, letters, forest, ptxt)

PAIRS = ["REVEL vs AlphaMissense", "AlphaMissense vs CADD", "CADD vs SIFT", "SIFT vs PolyPhen-2"]


def fig3(B, S, Q):
    sh5 = B[PRED5].notna().all(axis=1)
    sh6 = B[PRED6].notna().all(axis=1)
    n5, n6 = int(sh5.sum()), int(sh6.sum())
    g5, g6 = B.loc[sh5, GENE].nunique(), B.loc[sh6, GENE].nunique()
    vc = B.loc[sh6, GENE].value_counts()
    neff = (vc.sum() ** 2) / (vc ** 2).sum()
    p5, p6 = float(B.loc[sh5, "label"].mean()), float(B.loc[sh6, "label"].mean())
    Q.check("F3 5-pred shared 187,155 / 13,599", (n5, g5) == (187155, 13599), f"{n5:,}/{g5:,}")
    Q.check("F3 6-pred shared 53,552 / 1,981", (n6, g6) == (53552, 1981), f"{n6:,}/{g6:,}")
    Q.check("F3 N_eff = 282", round(neff) == 282, f"{neff:.1f}")
    Q.close("F3 6-pred P/LP", p6, 0.569, 6e-4)
    Q.report_new("F3 5-pred shared P/LP prevalence", round(p5, 4))
    isP = B["label"] == 1
    cov = {p: (float(B.loc[isP, p].notna().mean()), float(B.loc[~isP, p].notna().mean()))
           for p in PRED6}
    Q.close("F3 EVE P/LP coverage", cov["EVE"][0], 0.464, 6e-4)
    Q.close("F3 EVE B/LB coverage", cov["EVE"][1], 0.172, 6e-4)
    amc = S["pairs"].get("AlphaMissense vs CADD")
    Q.check("F3 AM-CADD +0.026 [+0.018,+0.035]",
            amc is not None and amc[:3] == (0.026, 0.018, 0.035), str(amc))

    fig = plt.figure(figsize=(W2, 120 * MM))
    gs = fig.add_gridspec(2, 6, left=0.105, right=0.95, top=0.95, bottom=0.10,
                          hspace=0.62, wspace=1.45)

    a = fig.add_subplot(gs[0, 0:2])
    a.bar([0, 1], [100 * p5, 100 * p6], 0.55, color=[LGREY, "#5A5A5A"], ec=DARK, lw=0.5)
    for i, v in enumerate([p5, p6]):
        a.text(i, 100 * v + 1.5, f"{100 * v:.1f}%", ha="center", fontsize=6)
    a.set_xticks([0, 1], [f"5 predictors\n{n5:,} var.\n{g5:,} genes",
                          f"6 predictors\n{n6:,} var.\n{g6:,} genes\nN$_{{eff}}$ = {neff:.0f}"],
                 fontsize=5.8)
    a.set_ylabel("Pathogenic/likely pathogenic (%)"); a.set_ylim(0, 70)

    b = fig.add_subplot(gs[0, 2:4])
    order = sorted(PRED6, key=lambda p: cov[p][0] - cov[p][1])
    for i, p in enumerate(order):
        pl, bl = 100 * cov[p][0], 100 * cov[p][1]
        b.plot([bl, pl], [i, i], color=LGREY, lw=1.6, zorder=1)
        b.plot(pl, i, "o", color=COL[p], ms=4, zorder=3)
        b.plot(bl, i, "o", mfc="white", mec=COL[p], mew=1, ms=4, zorder=3)
    b.set_yticks(range(len(order)), order); b.set_xlim(0, 105); b.set_xlabel("Coverage (%)")
    b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    b.plot([], [], "o", color=DARK, ms=3.5, label="P/LP")
    b.plot([], [], "o", mfc="white", mec=DARK, ms=3.5, label="B/LB")
    b.legend(loc="lower left", handletextpad=0.3)

    c = fig.add_subplot(gs[0, 4:6])
    forest(c, PRED5, S["delta"], COL)
    c.set_xlabel("\u0394AUROC (6- minus 5-predictor)")

    d = fig.add_subplot(gs[1, 0:3])
    keys = [k for k in PAIRS if k in S["pairs"]]
    vals = {k: S["pairs"][k][:3] for k in keys}
    cols = {k: (DARK if vals[k][1] > 0 or vals[k][2] < 0 else GREY) for k in keys}
    yy = forest(d, keys, vals, cols, labels=[k.replace(" vs ", " \u2212 ") for k in keys])
    for k, yv in zip(keys, yy):
        d.text(1.02, yv, ptxt(S["pairs"][k][3]), transform=d.get_yaxis_transform(),
               va="center", fontsize=6)
    d.set_xlabel("\u0394\u0394AUROC (change in pairwise gap)")

    e = fig.add_subplot(gs[1, 3:6])
    yy = np.arange(5)[::-1]
    for i, p in enumerate(PRED5):
        s1, s2, pv = S["het"][p]
        e.plot([s1, s2], [yy[i], yy[i]], color=LGREY, lw=1.6, zorder=1)
        e.plot(s1, yy[i], "o", mfc="white", mec=COL[p], mew=1, ms=4, zorder=3)
        e.plot(s2, yy[i], "o", color=COL[p], ms=4, zorder=3)
        e.text(1.02, yy[i], ptxt(pv), transform=e.get_yaxis_transform(), va="center", fontsize=6)
    e.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2)))
    e.set_yticks(yy, PRED5); e.spines["left"].set_visible(False); e.tick_params(axis="y", length=0)
    e.set_xlabel("\u0394AUROC under EVE-constrained restriction")
    e.plot([], [], "o", mfc="white", mec=DARK, ms=3.5, label="1 star")
    e.plot([], [], "o", color=DARK, ms=3.5, label="\u22652 stars")
    e.legend(loc="lower left", handletextpad=0.3)

    letters(fig, [(a, "a"), (b, "b"), (c, "c"), (d, "d"), (e, "e")])
    pd.DataFrame({p: {"PLP": cov[p][0], "BLB": cov[p][1]} for p in PRED6}).T.to_csv(
        os.path.join(SRC, "Fig3b.csv"))
    pd.DataFrame(S["delta"], index=["delta", "lo", "hi"]).T.to_csv(os.path.join(SRC, "Fig3c.csv"))
    pd.DataFrame(S["het"], index=["one_star", "ge2_star", "holm_p"]).T.to_csv(
        os.path.join(SRC, "Fig3e.csv"))
    return fig
