"""STEP 15C — Main Figure 5. Frozen Step 14 artifacts; no new analysis.

Panel B reads real per-assay Spearman values from
data/dms_spearman_shared5_14e.parquet. Panels A/C/D use frozen Step 14 results.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPEARMAN = "data/dms_spearman_shared5_14e.parquet"
BOOT = "data/dms_bootstrap_14f.parquet"
NAMES = ["AlphaMissense", "REVEL", "SIFT", "CADD", "PolyPhen-2"]
CI = {"AlphaMissense": (0.519, 0.450, 0.594), "REVEL": (0.440, 0.375, 0.527),
      "SIFT": (0.409, 0.360, 0.444), "CADD": (0.391, 0.332, 0.421),
      "PolyPhen-2": (0.349, 0.292, 0.395)}
AMR = (0.079, 0.006, 0.142)
PBEST = {"AlphaMissense": 0.982, "REVEL": 0.018}
FUNNEL = [("ProteinGym v1.3", "96 human assays / 81 proteins\n329,664 single-AA mutants"),
          ("UNIQUE_MANE_GRCH38_SNV", "38,517 variants\n38 assays / 34 proteins"),
          ("5-predictor shared", "37,197 variants\n36 assays / 32 proteins")]


def load_rho():
    if not os.path.exists(SPEARMAN):
        return None
    R = pd.read_parquet(SPEARMAN)
    W = R.pivot_table(index=["assay", "protein"], columns="predictor",
                      values="spearman_rho").dropna(subset=NAMES)
    return {n: W[n].to_numpy() for n in NAMES}


def refresh_ci():
    if not os.path.exists(BOOT):
        return CI
    B = pd.read_parquet(BOOT)
    out = {}
    for n in NAMES:
        if n in B.columns:
            out[n] = (CI[n][0], float(B[n].quantile(0.025)), float(B[n].quantile(0.975)))
        else:
            out[n] = CI[n]
    return out


def build(rho, ci, out="figures/Figure5_DMS_validation"):
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.2))
    fig.suptitle("Figure 5 — Independent experimental (DMS) validation",
                 fontsize=12, fontweight="bold", y=0.98)

    a = ax[0, 0]; a.axis("off")
    for i, (t, sub) in enumerate(FUNNEL):
        yy = 0.84 - i * 0.28
        a.add_patch(plt.Rectangle((0.10, yy - 0.105), 0.80, 0.17,
                                  fc=["#4C72B0", "#55A868", "#C44E52"][i], alpha=0.88,
                                  transform=a.transAxes))
        a.text(0.5, yy + 0.032, t, ha="center", fontsize=9.5, fontweight="bold",
               color="white", transform=a.transAxes)
        a.text(0.5, yy - 0.048, sub, ha="center", fontsize=8, color="white",
               transform=a.transAxes)
        if i < 2:
            a.annotate("", xy=(0.5, yy - 0.175), xytext=(0.5, yy - 0.108),
                       xycoords=a.transAxes, textcoords=a.transAxes,
                       arrowprops=dict(arrowstyle="-|>", color="#333", lw=2.2))
    a.text(0.5, 0.05, "MaveDB validation: BRCA2 reconstructable SNVs = 51\n"
           "exact GRCh38 chr:pos:REF:ALT agreement = 51/51", ha="center", fontsize=8.5,
           transform=a.transAxes,
           bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="grey"))
    a.set_title("A. External benchmark construction", fontsize=10, loc="left")

    b = ax[0, 1]
    if rho is not None:
        parts = b.violinplot([rho[n] for n in NAMES], positions=range(len(NAMES)),
                             showmedians=False, showextrema=False, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0"); pc.set_alpha(0.35)
        rs = np.random.default_rng(0)
        for i, n in enumerate(NAMES):
            b.scatter(rs.normal(i, 0.05, len(rho[n])), rho[n], s=9, color="#333",
                      alpha=0.55, zorder=3)
    for i, n in enumerate(NAMES):
        b.plot([i - 0.35, i + 0.35], [ci[n][0]] * 2, color="#C44E52", lw=2.2, zorder=4)
        b.text(i, 1.02, f"{ci[n][0]:.3f}", ha="center", fontsize=8,
               fontweight="bold", color="#C44E52")
    b.set_xticks(range(len(NAMES)), NAMES, rotation=18, ha="right", fontsize=8.5)
    b.set_ylabel("per-assay Spearman ρ"); b.set_ylim(0, 1.12)
    b.set_title("B. Per-assay experimental performance (36 assays; red = median)",
                fontsize=10, loc="left")

    c = ax[1, 0]; y = np.arange(len(NAMES))[::-1]
    for i, n in enumerate(NAMES):
        m, lo, hi = ci[n]
        c.plot([lo, hi], [y[i]] * 2, c="#333", lw=1.8)
        for e in (lo, hi):
            c.plot([e, e], [y[i] - 0.11, y[i] + 0.11], c="#333", lw=1.8)
        c.scatter([m], [y[i]], s=60, color="#8172B2", zorder=3)
        c.text(0.705, y[i], f"{m:.3f} [{lo:.3f}, {hi:.3f}]", va="center",
               ha="right", fontsize=8)
    c.set_yticks(y, NAMES, fontsize=9); c.set_xlim(0.26, 0.72); c.set_ylim(-0.9, 4.5)
    c.set_xlabel("median assay Spearman ρ (5,000 protein-clustered replicates)")
    c.set_title("C. Protein-clustered inference", fontsize=10, loc="left")
    c.text(0.5, -0.032, f"AlphaMissense − REVEL: {AMR[0]:+.3f} [{AMR[1]:+.3f}, {AMR[2]:+.3f}]"
           f"   |   P(best): AlphaMissense {PBEST['AlphaMissense']:.3f}, "
           f"REVEL {PBEST['REVEL']:.3f}", transform=c.transAxes, ha="center", fontsize=8.2,
           bbox=dict(boxstyle="round,pad=0.35", fc="#f0f0f0", ec="grey"))

    d = ax[1, 1]; d.axis("off")
    rows = [("Evaluation unit",
             "assay-weighted:   AM > REVEL > SIFT > CADD > PolyPhen-2\n"
             "protein-weighted: AM > REVEL > SIFT > CADD > PolyPhen-2\n"
             "Kendall concordance = 1.000"),
            ("Leave-one-protein-out",
             "AM #1 in 32/32 configurations\n"
             "minimum AM−REVEL gap = +0.034\nreversals = 0"),
            ("ProteinGym assay categories",
             "AM #1 in 5/5 eligible categories\n"
             "AM−REVEL gap range = +0.014 to +0.148")]
    for i, (h, t) in enumerate(rows):
        yy = 0.90 - i * 0.31
        d.text(0.02, yy, h, fontsize=9.5, fontweight="bold", transform=d.transAxes)
        d.text(0.05, yy - 0.05, t, fontsize=8.2, va="top", family="monospace",
               transform=d.transAxes)
    d.set_title("D. Robustness summary", fontsize=10, loc="left")

    fig.text(0.5, 0.030, "The leading experimental result is robust to protein clustering, "
             "evaluation-unit choice, leave-one-protein-out analysis, and ProteinGym "
             "assay category.", ha="center", fontsize=9, style="italic")
    fig.text(0.5, 0.009, "EVE is excluded from primary inference (2.1% coverage of the DMS "
             "SNV benchmark; 791 shared variants, 7 assays with n>=20).",
             ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=[0, 0.048, 1, 0.955])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    rho = load_rho()
    ci = refresh_ci()
    out = build(rho, ci)
    print("FIGURE 5 — GENERATION QC")
    print("=" * 24)
    print("A. construction funnel:")
    for t, sub in FUNNEL:
        print(f"   {t:<26}{sub.replace(chr(10), ' | ')}")
    print("   MaveDB validation: BRCA2 51 reconstructable, 51/51 exact agreement")
    print(f"B. per-assay rho source: "
          f"{'data/dms_spearman_shared5_14e.parquet (real values)' if rho else 'UNAVAILABLE'}")
    if rho:
        for n in NAMES:
            v = rho[n]
            print(f"   {n:<15}n_assays={len(v)}  median={np.median(v):.3f}  "
                  f"min={v.min():.3f}  max={v.max():.3f}")
    print(f"C. protein-clustered CI source: "
          f"{'data/dms_bootstrap_14f.parquet' if os.path.exists(BOOT) else 'frozen constants'}")
    for n in NAMES:
        m, lo, hi = ci[n]
        print(f"   {n:<15}{m:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"   AM - REVEL {AMR[0]:+.3f} [{AMR[1]:+.3f}, {AMR[2]:+.3f}]; "
          f"P(best) AM {PBEST['AlphaMissense']:.3f} / REVEL {PBEST['REVEL']:.3f}")
    print("D. robustness: unit Kendall 1.000; LOPO 32/32 (min gap +0.034, 0 reversals); "
          "assay categories 5/5 (gap +0.014 to +0.148)")
    print("   EVE excluded from primary inference (2.1% coverage) — noted in caption")
    print(f"\nsaved: {out}.pdf, {out}.png")
