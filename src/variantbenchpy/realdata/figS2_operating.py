"""Supplementary Figure S2 — Step 9 simulation operating characteristics.

Verified frozen rerun (200 replicates per cell, Wilson 95% intervals);
source grid: data/step9_verification_grid.csv.
Simulation only; not a biological benchmark result.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID = "data/step9_verification_grid.csv"
G = [40, 80, 160]
NULL = {"naive": {40: (0.260, 0.204, 0.325), 80: (0.195, 0.146, 0.255),
                  160: (0.100, 0.066, 0.149)},
        "crossfit": {40: (0.070, 0.042, 0.114), 80: (0.050, 0.027, 0.090),
                     160: (0.045, 0.024, 0.083)}}
POW = {40: (0.185, 0.137, 0.245), 80: (0.360, 0.297, 0.429), 160: (0.755, 0.691, 0.809)}
N_REP = 200


def refresh():
    if not os.path.exists(GRID):
        return NULL, POW, False
    d = pd.read_csv(GRID)
    n, p = {"naive": {}, "crossfit": {}}, {}
    for _, r in d.iterrows():
        cell = (float(r["rate"]), float(r["ci_lo"]), float(r["ci_hi"]))
        if float(r["effect"]) == 0.0 and r["method"] in n:
            n[r["method"]][int(r["genes"])] = cell
        elif float(r["effect"]) > 0 and r["method"] == "crossfit":
            p[int(r["genes"])] = cell
    return (n if all(n[m] for m in n) else NULL), (p or POW), True


def build(null, pw, out="figures/FigureS2_simulation_operating_characteristics"):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.1))
    fig.suptitle("Figure S2 — Step 9 simulation operating characteristics",
                 fontsize=12, fontweight="bold", y=0.98)

    a = ax[0]; x = np.arange(len(G))
    for m, col, off in [("naive", "#C44E52", -0.17), ("crossfit", "#4C72B0", 0.17)]:
        pts = [null[m][g][0] for g in G]
        err = [[null[m][g][0] - null[m][g][1] for g in G],
               [null[m][g][2] - null[m][g][0] for g in G]]
        a.bar(x + off, pts, 0.32, yerr=err, capsize=4, color=col, label=m)
        for i, g in enumerate(G):
            a.text(x[i] + off, null[m][g][2] + 0.014, f"{null[m][g][0]:.3f}",
                   ha="center", fontsize=7.8)
    a.axhline(0.05, ls="--", c="#333", lw=1.3)
    a.text(-0.45, 0.058, "nominal 0.05", fontsize=8, color="#333", ha="left")
    a.set_xticks(x, [f"{g} genes" for g in G])
    a.set_ylabel("false-positive rate (effect = 0)")
    a.set_ylim(0, 0.40); a.legend(fontsize=9, frameon=False)
    a.set_title("A. Null calibration", fontsize=10, loc="left")

    b = ax[1]
    pts = [pw[g][0] for g in G]
    err = [[pw[g][0] - pw[g][1] for g in G], [pw[g][2] - pw[g][0] for g in G]]
    b.errorbar(G, pts, yerr=err, fmt="o-", color="#55A868", capsize=5, lw=2, ms=9)
    for g in G:
        b.text(g, pw[g][2] + 0.035, f"{pw[g][0]:.3f}\n[{pw[g][1]:.3f}, {pw[g][2]:.3f}]",
               ha="center", fontsize=8)
    b.set_xticks(G, [str(g) for g in G]); b.set_xlabel("independent simulated genes")
    b.set_ylabel("power"); b.set_ylim(0, 1.05); b.set_xlim(20, 180)
    b.set_title("B. Power for ΔΔAUROC ≈ 0.05 (cross-fit)", fontsize=10, loc="left")

    fig.text(0.5, 0.055, "SIMULATION OPERATING CHARACTERISTICS — NOT A BIOLOGICAL "
             "BENCHMARK RESULT", ha="center", fontsize=9, fontweight="bold",
             color="#B22222",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#B22222", lw=1.2))
    fig.text(0.5, 0.010, f"{N_REP} replicates per cell; Wilson 95% intervals. Power "
             "estimates apply only to the simulated gene counts and should not be "
             "extrapolated to larger benchmarks.", ha="center", fontsize=7.6, color="#444")
    fig.tight_layout(rect=[0, 0.115, 1, 0.945])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    null, pw, parsed = refresh()
    out = build(null, pw)
    print("FIGURE S2 — GENERATION QC")
    print("=" * 25)
    print(f"source: {'data/step9_verification_grid.csv (parsed)' if parsed else 'frozen constants'}")
    print("null calibration values (effect = 0, false-positive rate [Wilson 95% CI]):")
    for g in G:
        n_, c_ = null["naive"][g], null["crossfit"][g]
        print(f"   {g:>3} genes   naive {n_[0]:.3f} [{n_[1]:.3f}, {n_[2]:.3f}]   |   "
              f"crossfit {c_[0]:.3f} [{c_[1]:.3f}, {c_[2]:.3f}]")
    print("power values (ΔΔAUROC ≈ 0.05, cross-fit):")
    for g in G:
        v = pw[g]
        print(f"   {g:>3} genes   {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
    print("label: SIMULATION OPERATING CHARACTERISTICS — NOT A BIOLOGICAL BENCHMARK RESULT")
    print(f"\nsaved: {out}.pdf, {out}.png")
