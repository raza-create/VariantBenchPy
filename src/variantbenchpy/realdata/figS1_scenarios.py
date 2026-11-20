"""Supplementary Figure S1 — Step 9 simulation scenario validation.

Verified frozen rerun values (50 replicates per scenario, 120 simulated genes).
Intervals are 2.5th-97.5th percentiles of winner retention across replicates.
Simulation validation only; not a biological benchmark result.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = [("Stable", 0.99, 0.92, 1.00),
             ("Dominant", 1.00, 1.00, 1.00),
             ("Gene-dependent", 0.48, 0.09, 0.84),
             ("Informative-missingness", 0.56, 0.23, 0.85),
             ("Close-performance", 0.45, 0.13, 0.80)]
N_REP, N_GENES = 50, 120


def build(out="figures/FigureS1_simulation_scenarios"):
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    y = np.arange(len(SCENARIOS))[::-1]
    cols = ["#55A868", "#4C72B0", "#DD8452", "#C44E52", "#8172B2"]
    for i, (lab, m, lo, hi) in enumerate(SCENARIOS):
        ax.plot([lo, hi], [y[i]] * 2, c="#333", lw=2)
        for e in (lo, hi):
            ax.plot([e, e], [y[i] - 0.12, y[i] + 0.12], c="#333", lw=2)
        ax.scatter([m], [y[i]], s=95, color=cols[i], zorder=3,
                   edgecolor="#222", linewidth=0.6)
        ax.text(1.055, y[i], f"{m:.2f} [{lo:.2f}, {hi:.2f}]", va="center",
                fontsize=9, family="monospace")
    ax.set_yticks(y, [s[0] for s in SCENARIOS], fontsize=10)
    ax.set_xlim(0, 1.32); ax.set_ylim(-1.15, 4.6)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.axvline(1.0, ls=":", c="grey", lw=1)
    ax.set_xlabel("winner retention across prespecified design configurations")
    ax.set_title("Figure S1 — Step 9 simulation scenario validation",
                 fontsize=12, fontweight="bold", loc="left", pad=14)
    ax.text(0.5, 0.045, "SIMULATION VALIDATION — NOT A BIOLOGICAL BENCHMARK RESULT",
            transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold",
            color="#B22222",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#B22222", lw=1.2))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.015, f"{N_REP} replicates per scenario, {N_GENES} simulated genes; "
             "intervals are 2.5th–97.5th percentiles of retention across replicates.",
             ha="center", fontsize=8, color="#444")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = build()
    print("FIGURE S1 — GENERATION QC")
    print("=" * 25)
    print(f"scenario values ({N_REP} reps, {N_GENES} genes; verified frozen rerun):")
    for lab, m, lo, hi in SCENARIOS:
        print(f"   {lab:<26}{m:.2f}")
    print("intervals (2.5th–97.5th percentile of winner retention):")
    for lab, m, lo, hi in SCENARIOS:
        print(f"   {lab:<26}[{lo:.2f}, {hi:.2f}]   width={hi-lo:.2f}")
    print("label: SIMULATION VALIDATION — NOT A BIOLOGICAL BENCHMARK RESULT")
    print(f"\nsaved: {out}.pdf, {out}.png")
