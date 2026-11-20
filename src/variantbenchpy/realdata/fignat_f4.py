"""Figure 4 — independent validation: temporal exposure and DMS benchmark."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variantbenchpy.realdata.fignat_common import (
    W2, MM, PRED5, COL, GREY, LGREY, DARK, SRC, AM_CUT, EVE_CUT, TEMP23, TEMP21,
    DMS_MED, CAT_GAP, letters, forest)

DORDER = ["AlphaMissense", "REVEL", "SIFT", "CADD", "PolyPhen-2"]
NICE_CAT = {"OrganismalFitness": "Organismal fitness"}


def fig4(B, Q):
    sh5 = B[PRED5].notna().all(axis=1)
    post23 = int((sh5 & (B["first_in_clinvar"] > AM_CUT)).sum())
    post21 = int((sh5 & (B["first_in_clinvar"] > EVE_CUT)).sum())
    Q.check("F4 post-2023 n = 70,661", post23 == 70661, f"{post23:,}")
    Q.check("F4 post-2021 n = 110,050", post21 == 110050, f"{post21:,}")

    R = pd.read_parquet("data/dms_spearman_shared5_14e.parquet")
    Wd = R.pivot_table(index=["assay", "protein"], columns="predictor",
                       values="spearman_rho").dropna(subset=PRED5)
    nprot = Wd.index.get_level_values("protein").nunique()
    Q.check("F4 DMS 36 assays / 32 proteins", (len(Wd), nprot) == (36, 32), f"{len(Wd)}/{nprot}")
    med = {p: float(Wd[p].median()) for p in PRED5}
    for p in PRED5:
        Q.close(f"F4 DMS median rho {p}", med[p], DMS_MED[p], 6e-4)
    Bt = pd.read_parquet("data/dms_bootstrap_14f.parquet")
    ci = {p: (med[p], float(Bt[p].quantile(.025)), float(Bt[p].quantile(.975))) for p in PRED5}
    dd = Bt["AlphaMissense"] - Bt["REVEL"]
    dlo, dhi = float(dd.quantile(.025)), float(dd.quantile(.975))
    Q.close("F4 AM-REVEL CI low", dlo, 0.006, 6e-4)
    Q.close("F4 AM-REVEL CI high", dhi, 0.142, 6e-4)
    pbest = float((Bt[PRED5].idxmax(axis=1) == "AlphaMissense").mean())
    Q.close("F4 P(best) AlphaMissense", pbest, 0.982, 6e-4)
    PW = pd.read_parquet("data/dms_protein_weighted_14g.parquet")
    ref = float(PW["AlphaMissense"].median() - PW["REVEL"].median())
    Q.close("F4 protein-weighted AM-REVEL", ref, 0.050, 6e-4)
    L = pd.read_parquet("data/dms_lopo_14h.parquet")
    Q.check("F4 LOPO 32 configs, AM first in all",
            len(L) == 32 and bool((L["winner"] == "AlphaMissense").all()),
            f"{len(L)} / {int((L['winner'] == 'AlphaMissense').sum())}")
    Q.close("F4 LOPO min gap", float(L["am_minus_revel"].min()), 0.034, 6e-4)
    T = pd.read_parquet("data/dms_assaytype_14i.parquet")
    na = T.groupby("category")["assay"].nunique()
    npr = T.groupby("category")["protein"].nunique()
    elig = [c for c in na.index if na[c] >= 3 and npr[c] >= 3]
    gaps = {}
    for c in elig:
        pp = T[T["category"] == c].groupby("protein")[PRED5].median()
        gaps[c] = float(pp["AlphaMissense"].median() - pp["REVEL"].median())
        Q.close(f"F4 category gap {c}", gaps[c], CAT_GAP[c], 6e-4)

    fig = plt.figure(figsize=(W2, 122 * MM))
    gs = fig.add_gridspec(2, 6, left=0.10, right=0.985, top=0.95, bottom=0.11,
                          hspace=0.75, wspace=1.45)

    a = fig.add_subplot(gs[0, 0:2])
    yrs = B.loc[sh5, "first_in_clinvar"].dt.year.dropna().astype(int).value_counts().sort_index()
    a.bar(yrs.index, yrs.values / 1000, color=LGREY, ec=DARK, lw=0.4, width=0.8)
    top = a.get_ylim()[1]
    for x, lab, col in [(2021 + 300 / 365, "EVE", COL["EVE"]),
                        (2023 + 265 / 365, "AlphaMissense", COL["AlphaMissense"])]:
        a.axvline(x, color=col, lw=0.9, ls=(0, (3, 1.5)))
        a.text(x - 0.3, top * 0.98, lab, rotation=90, ha="right", va="top", fontsize=5.6, color=col)
    a.set_xlabel("First in ClinVar (germline)"); a.set_ylabel("Variants (thousands)")
    a.set_xticks([2014, 2018, 2022, 2026])

    b = fig.add_subplot(gs[0, 2:4])
    yy = np.arange(5)[::-1]
    for i, p in enumerate(PRED5):
        for vals, off, filled in [(TEMP23, 0.16, True), (TEMP21, -0.16, False)]:
            v, lo, hi = vals[p]
            b.plot([lo, hi], [yy[i] + off] * 2, color=COL[p], lw=1.0)
            b.plot(v, yy[i] + off, "o", ms=3.6, mec=COL[p], mew=0.9,
                   mfc=COL[p] if filled else "white", zorder=3)
    b.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2)))
    b.set_yticks(yy, PRED5); b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    b.set_xlabel("\u0394AUROC (subset \u2212 full)"); b.set_xlim(-0.004, 0.036)
    b.plot([], [], "o", color=DARK, ms=3.3, label="After 2023-09-22")
    b.plot([], [], "o", mfc="white", mec=DARK, ms=3.3, label="After 2021-10-27")
    b.legend(loc="lower right", handletextpad=0.3)

    c = fig.add_subplot(gs[0, 4:6])
    bp = c.boxplot([Wd[p].to_numpy() for p in DORDER], positions=range(5), widths=0.5,
                   showfliers=False, patch_artist=True, medianprops=dict(color=DARK, lw=0.9),
                   whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5), boxprops=dict(lw=0.5))
    for patch, p in zip(bp["boxes"], DORDER):
        patch.set_facecolor(COL[p]); patch.set_alpha(0.35)
    rng = np.random.default_rng(1)
    for i, p in enumerate(DORDER):
        c.scatter(i + rng.uniform(-0.15, 0.15, len(Wd)), Wd[p], s=4, color=COL[p], lw=0, zorder=3)
    c.set_xticks(range(5), DORDER, rotation=35, ha="right"); c.set_ylim(0, 0.9)
    c.set_ylabel("Per-assay Spearman \u03c1")

    d = fig.add_subplot(gs[1, 0:2])
    forest(d, DORDER, ci, COL)
    d.set_xlim(0.25, 0.66); d.set_xlabel("Median assay \u03c1 (95% CI)")
    d.text(0.98, 0.02, f"AM \u2212 REVEL {med['AlphaMissense'] - med['REVEL']:+.3f}\n"
                       f"({dlo:+.3f} to {dhi:+.3f})\nP(best) = {pbest:.3f}",
           transform=d.transAxes, ha="right", va="bottom", fontsize=5.6)

    e = fig.add_subplot(gs[1, 2:4])
    gsort = np.sort(L["am_minus_revel"].to_numpy())
    e.plot(np.arange(1, len(gsort) + 1), gsort, "o", ms=2.8, color=COL["AlphaMissense"])
    e.axhline(ref, color=DARK, lw=0.6, label="All proteins")
    e.axhline(0, color=GREY, lw=0.6, ls=(0, (2, 2)))
    e.set_xlabel("Protein removed (ordered)"); e.set_ylim(-0.01, 0.08)
    e.set_ylabel("AlphaMissense \u2212 REVEL\n(protein-weighted median \u03c1)")
    e.set_xticks([1, 16, 32]); e.legend(loc="upper left")

    f = fig.add_subplot(gs[1, 4:6])
    co = sorted(gaps, key=gaps.get)
    yy = np.arange(len(co))
    f.barh(yy, [gaps[c0] for c0 in co], color=COL["AlphaMissense"], height=0.6)
    for i, c0 in enumerate(co):
        f.text(gaps[c0] + 0.004, i, f"{int(na[c0])}/{int(npr[c0])}", va="center", fontsize=5.6)
    f.set_yticks(yy, [NICE_CAT.get(c0, c0) for c0 in co]); f.set_xlim(0, 0.2)
    f.set_xlabel("AlphaMissense \u2212 REVEL\n(protein-weighted median \u03c1)")
    f.spines["left"].set_visible(False); f.tick_params(axis="y", length=0)

    letters(fig, [(a, "a"), (b, "b"), (c, "c"), (d, "d"), (e, "e"), (f, "f")])
    yrs.rename("variants").to_csv(os.path.join(SRC, "Fig4a.csv"))
    pd.DataFrame({p: {"post2023": TEMP23[p], "post2021": TEMP21[p]} for p in PRED5}).T.to_csv(
        os.path.join(SRC, "Fig4b.csv"))
    Wd.reset_index().to_csv(os.path.join(SRC, "Fig4c.csv"), index=False)
    pd.DataFrame(ci, index=["median", "lo", "hi"]).T.to_csv(os.path.join(SRC, "Fig4d.csv"))
    L.to_csv(os.path.join(SRC, "Fig4e.csv"), index=False)
    pd.DataFrame({"gap": gaps, "assays": na[elig], "proteins": npr[elig]}).to_csv(
        os.path.join(SRC, "Fig4f.csv"))
    return fig
