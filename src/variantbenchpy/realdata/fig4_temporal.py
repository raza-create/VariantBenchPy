"""STEP 15C — Main Figure 4. Frozen Step 13 values; no new analysis.

Panel A reads the frozen provenance artifact (clinvar_temporal_provenance.parquet).
Panels B-D use frozen Step 13E/13G results.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROV = "data/clinvar_temporal_provenance.parquet"
NAMES = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
P23 = {"AlphaMissense": (0.010, 0.008, 0.012), "REVEL": (0.011, 0.009, 0.013),
       "CADD": (0.021, 0.019, 0.024), "SIFT": (0.023, 0.020, 0.026),
       "PolyPhen-2": (0.030, 0.027, 0.032)}
STD = {"AlphaMissense": (0.010, 0.008), "REVEL": (0.011, 0.011),
       "CADD": (0.021, 0.011), "SIFT": (0.023, 0.015), "PolyPhen-2": (0.030, 0.021)}
P21 = {"AlphaMissense": (0.0058, 0.0044, 0.0072), "REVEL": (0.0050, 0.0035, 0.0064),
       "CADD": (0.0118, 0.0098, 0.0138), "SIFT": (0.0147, 0.0123, 0.0172),
       "PolyPhen-2": (0.0199, 0.0175, 0.0223)}
FALLBACK_YEARS = {2013: 5157, 2014: 5669, 2015: 5946, 2016: 11194, 2017: 10209,
                  2018: 9558, 2019: 20922, 2020: 9459, 2021: 13293, 2022: 17447,
                  2023: 35413, 2024: 32620, 2025: 35010, 2026: 13158}


def year_counts():
    if not os.path.exists(PROV):
        return FALLBACK_YEARS, False
    d = pd.read_parquet(PROV)
    y = pd.to_datetime(d["germline_date_created"], errors="coerce").dt.year.dropna()
    return y.astype(int).value_counts().sort_index().to_dict(), True


def build(years, out="figures/Figure4_temporal_exposure"):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.8))
    fig.suptitle("Figure 4 — Temporal exposure-resistant evaluation",
                 fontsize=12, fontweight="bold", y=0.98)
    y_idx = np.arange(len(NAMES))[::-1]

    a = ax[0, 0]
    yr = sorted(years); vals = [years[k] for k in yr]
    a.bar(yr, vals, color="#4C72B0", width=0.75)
    top = max(vals)
    a.axvline(2021.82, color="#55A868", ls="--", lw=1.8)
    a.axvline(2023.73, color="#C44E52", ls="--", lw=1.8)
    a.annotate("2021-10-27\nEVE proxy", xy=(2021.82, top * 1.02), xytext=(2017.6, top * 1.16),
               fontsize=7.5, color="#55A868", fontweight="bold", ha="center",
               arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.2))
    a.annotate("2023-09-22\nAlphaMissense proxy", xy=(2023.73, top * 0.92),
               xytext=(2025.2, top * 1.18), fontsize=7.5, color="#C44E52",
               fontweight="bold", ha="center",
               arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
    a.set_ylim(0, top * 1.34); a.set_ylabel("variants")
    a.set_xlabel("germline First-in-ClinVar year")
    a.set_title("A. Temporal distribution of benchmark variants", fontsize=10, loc="left")
    a.text(0.02, 0.97, "PUBLICATION-DATE PROXIES,\nNOT VERIFIED TRAINING CUTOFFS",
           transform=a.transAxes, fontsize=7.5, fontweight="bold", color="#B22222",
           va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B22222", lw=1))

    b = ax[0, 1]
    for i, n in enumerate(NAMES):
        d, lo, hi = P23[n]
        b.plot([lo, hi], [y_idx[i], y_idx[i]], c="#333", lw=1.6)
        for e in (lo, hi):
            b.plot([e, e], [y_idx[i] - 0.12, y_idx[i] + 0.12], c="#333", lw=1.6)
        b.scatter([d], [y_idx[i]], s=55, color="#55A868", zorder=3)
        b.text(0.0455, y_idx[i], f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
               va="center", ha="right", fontsize=7.6)
    b.axvline(0, c="grey", ls="--", lw=1.2)
    b.set_yticks(y_idx, NAMES, fontsize=9); b.set_xlim(-0.004, 0.047)
    b.set_xlabel("ΔAUROC (post-2023 − full)")
    b.set_title("B. Full → post-2023 ΔAUROC (paired gene-clustered CI)",
                fontsize=10, loc="left")

    c = ax[1, 0]
    x = np.arange(len(NAMES))
    c.bar(x - 0.2, [STD[n][0] for n in NAMES], 0.4, label="observed", color="#4C72B0")
    c.bar(x + 0.2, [STD[n][1] for n in NAMES], 0.4, label="standardized", color="#DD8452")
    for i, n in enumerate(NAMES):
        c.text(i - 0.2, STD[n][0] + 0.0008, f"{STD[n][0]:+.3f}", ha="center", fontsize=7.2)
        c.text(i + 0.2, STD[n][1] + 0.0008, f"{STD[n][1]:+.3f}", ha="center", fontsize=7.2)
    c.set_xticks(x, NAMES, rotation=18, ha="right", fontsize=8.5); c.set_ylim(0, 0.043)
    c.set_ylabel("ΔAUROC"); c.legend(fontsize=8, frameon=False, loc="upper left")
    c.set_title("C. Observed vs composition-standardized", fontsize=10, loc="left")
    c.text(0.98, 0.96, "SENSITIVITY, NOT CAUSAL DECOMPOSITION", transform=c.transAxes,
           ha="right", va="top", fontsize=7.5, fontweight="bold", color="#B22222",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B22222", lw=1))

    d = ax[1, 1]
    for i, n in enumerate(NAMES):
        v, lo, hi = P21[n]
        d.plot([lo, hi], [y_idx[i], y_idx[i]], c="#333", lw=1.4)
        d.scatter([v], [y_idx[i]], s=45, color="#8172B2", zorder=3)
        d.text(0.0288, y_idx[i], f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]",
               va="center", ha="right", fontsize=7.2)
    d.axvline(0, c="grey", ls="--", lw=1.2)
    d.set_yticks(y_idx, NAMES, fontsize=9); d.set_xlim(-0.002, 0.030)
    d.set_xlabel("ΔAUROC (post-2021 − full)")
    d.set_title("D. Post-2021 sensitivity (secondary)", fontsize=10, loc="left")

    fig.text(0.5, 0.012, "Performance does not deteriorate under temporal exposure-resistant "
             "evaluation; leading rankings are largely preserved although some "
             "relative-performance gaps change.", ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    years, from_artifact = year_counts()
    out = build(years)
    print("FIGURE 4 — GENERATION QC")
    print("=" * 24)
    print(f"A. year distribution source: "
          f"{'data/clinvar_temporal_provenance.parquet' if from_artifact else 'frozen constants'}")
    print(f"   years {min(years)}-{max(years)}, total {sum(years.values()):,} variants; "
          "landmarks 2021-10-27 (EVE), 2023-09-22 (AlphaMissense)")
    print("   label: PUBLICATION-DATE PROXIES, NOT VERIFIED TRAINING CUTOFFS")
    print("B. full -> post-2023 ΔAUROC [paired gene-clustered 95% CI]:")
    for n, (v, lo, hi) in P23.items():
        print(f"   {n:<15}{v:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("C. observed vs composition-standardized ΔAUROC:")
    for n, (o, s) in STD.items():
        print(f"   {n:<15}{o:+.3f}  ->  {s:+.3f}")
    print("   label: SENSITIVITY, NOT CAUSAL DECOMPOSITION")
    print("D. post-2021 sensitivity ΔAUROC (all positive):")
    for n, (v, lo, hi) in P21.items():
        print(f"   {n:<15}{v:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"\nsaved: {out}.pdf, {out}.png")
