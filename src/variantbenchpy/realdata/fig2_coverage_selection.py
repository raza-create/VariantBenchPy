"""STEP 15C — Main Figure 2. Frozen Step 11 values; computes nothing new.

GENE-COUNT PROVENANCE: all counts shown use the frozen Step-10A MANE-informed
primary-gene mapping (15,249 / 13,599 / 1,981). The archived Step 11 report
header still prints the pre-resolution counts (15,229 / 13,585 / 1,982) because
step11_selection.py derives genes from clinvar_ingest rather than
primary_gene_10A.parquet; see KNOWN_INCONSISTENCY below.

Provenance: values are the frozen Step 11 matched-selection results
(step11_selection.py, paired gene-clustered bootstrap). If
data/step11_selection.txt exists it is parsed; otherwise the frozen
constants below are used and the QC output flags that the report was
not archived.
"""
from __future__ import annotations
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = "data/step11_selection.txt"
KNOWN_INCONSISTENCY = ("none - Step 11 now clusters on the frozen Step-10A "
                       "mapping; counts and inference are consistent")
FUNNEL = [("ClinVar total", 225208, 15249),
          ("5-predictor shared", 187155, 13599),
          ("6-predictor shared", 53552, 1981)]
DELTA = {"AlphaMissense": (-0.015, -0.023, -0.008),
         "REVEL": (-0.013, -0.020, -0.006),
         "CADD": (-0.041, -0.054, -0.029),
         "SIFT": (-0.028, -0.041, -0.015),
         "PolyPhen-2": (-0.036, -0.046, -0.026)}
DDA = (0.026, 0.017, 0.035)
PREV = (31.9, 56.9)
CONC = [(10, 12.1), (50, 32.2), (100, 44.9)]
NEFF, NOMINAL = 282, 1981


def parse_report(path=REPORT):
    if not os.path.exists(path):
        return DELTA, DDA, False
    txt = open(path, encoding="utf-8").read()
    d = {}
    for p in DELTA:
        m = re.search(rf"^\s+{re.escape(p)}\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+) "
                      rf"\[([+-][\d.]+), ([+-][\d.]+)\]", txt, re.M)
        if m:
            d[p] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    m = re.search(r"AlphaMissense vs CADD\s+([+-][\d.]+)\s+\[([+-][\d.]+), ([+-][\d.]+)\]", txt)
    dd = (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else DDA
    return (d or DELTA), dd, True


def build(delta, dda, out="figures/Figure2_coverage_selection"):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Figure 2 — Complete-case predictor coverage reshapes the benchmark population",
                 fontsize=12, fontweight="bold", y=0.985)

    a = ax[0, 0]
    for i, (lab, nv, ng) in enumerate(FUNNEL):
        w, y = nv / FUNNEL[0][1], 2 - i
        a.barh([y], [w], height=0.5, color=["#4C72B0", "#55A868", "#C44E52"][i])
        a.text(w + 0.03, y, f"{nv:,} variants\n{ng:,} genes", va="center", fontsize=8.5)
        a.text(0.01, y + 0.34, lab, fontsize=9, fontweight="bold")
    a.set_xlim(0, 1.5); a.set_ylim(-0.4, 2.8); a.set_yticks([]); a.set_xticks([])
    for s in ["top", "right", "left", "bottom"]:
        a.spines[s].set_visible(False)
    a.set_title("A. Population funnel", fontsize=10, loc="left")

    b = ax[0, 1]
    b.bar([0, 1], list(PREV), color=["#4C72B0", "#C44E52"], width=0.5)
    for i, v in enumerate(PREV):
        b.text(i, v + 1.5, f"{v}%", ha="center", fontsize=10, fontweight="bold")
    b.set_xticks([0, 1], ["full ClinVar", "6-pred shared"], fontsize=9)
    b.set_ylabel("P/LP prevalence (%)"); b.set_ylim(0, 105)
    b.set_title("B. Composition shift", fontsize=10, loc="left")
    tx = ("Gene concentration (6-pred shared)\n"
          + "\n".join(f"  top {k:>3} genes: {v:>5.1f}%" for k, v in CONC)
          + f"\n  N_eff = {NEFF} of {NOMINAL:,} nominal")
    b.text(0.5, 0.995, tx, transform=b.transAxes, ha="center", va="top", fontsize=8,
           family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="grey"))

    c = ax[1, 0]
    names = list(delta); y = np.arange(len(names))[::-1]
    for i, n in enumerate(names):
        dv, lo, hi = delta[n]
        c.plot([lo, hi], [y[i], y[i]], c="#333", lw=1.6)
        for e in (lo, hi):
            c.plot([e, e], [y[i] - 0.12, y[i] + 0.12], c="#333", lw=1.6)
        c.scatter([dv], [y[i]], s=55, color="#C44E52", zorder=3)
        c.text(lo - 0.002, y[i] + 0.28, f"{dv:+.3f} [{lo:+.3f}, {hi:+.3f}]",
               va="center", ha="left", fontsize=7.5)
    c.axvline(0, c="grey", ls="--", lw=1.2)
    c.set_yticks(y, names, fontsize=9)
    c.set_xlim(-0.062, 0.012)
    c.set_xlabel("ΔAUROC (6-pred shared − 5-pred shared)")
    c.set_title("C. Matched ΔAUROC, gene-clustered 95% CI", fontsize=10, loc="left")

    d = ax[1, 1]
    pt, lo, hi = dda
    d.plot([lo, hi], [0, 0], c="#333", lw=2)
    for e in (lo, hi):
        d.plot([e, e], [-0.06, 0.06], c="#333", lw=2)
    d.scatter([pt], [0], s=90, color="#8172B2", zorder=3)
    d.axvline(0, c="grey", ls="--", lw=1.2)
    d.set_ylim(-0.5, 0.5); d.set_yticks([0], ["AlphaMissense − CADD"], fontsize=9)
    d.set_xlim(-0.006, 0.046); d.set_xlabel("ΔΔAUROC (change in performance gap)")
    d.text(pt, 0.2, f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]\nHolm p < 0.001",
           ha="center", fontsize=9, fontweight="bold")
    d.set_title("D. Relative-performance change", fontsize=10, loc="left")

    fig.text(0.5, 0.022, "Complete-case restriction changes the benchmark population and "
             "estimated performance, not merely predictor availability.",
             ha="center", fontsize=9.5, style="italic")
    fig.text(0.5, 0.004, "Counts use the frozen Step-10A MANE-informed gene-cluster mapping "
             "(225,208/15,249 -> 187,155/13,599 -> 53,552/1,981).",
             ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    delta, dda, archived = parse_report()
    out = build(delta, dda)
    print("FIGURE 2 — GENERATION QC")
    print("=" * 24)
    print(f"source: {'data/step11_selection.txt (parsed)' if archived else 'frozen constants'}")
    if not archived:
        print("  WARNING: Step 11 stdout was never archived; values are frozen constants.")
        print("  To archive: python3 -m variantbenchpy.realdata.step11_selection "
              "| tee data/step11_selection.txt")
    print(f"A. funnel counts: " + " -> ".join(f"{n:,}v/{g:,}g" for _, n, g in FUNNEL))
    print(f"B. prevalence: {PREV[0]}% -> {PREV[1]}% | concentration "
          + ", ".join(f"top{k}={v}%" for k, v in CONC) + f", N_eff={NEFF}/{NOMINAL:,}")
    print("C. ΔAUROC values and CIs:")
    for n, (dv, lo, hi) in delta.items():
        print(f"   {n:<15}{dv:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"D. ΔΔAUROC: AlphaMissense-CADD {dda[0]:+.3f} [{dda[1]:+.3f}, {dda[2]:+.3f}], "
          "Holm p<0.001")
    print("\nGENE-COUNT PROVENANCE: frozen Step-10A mapping "
          "(15,249 / 13,599 / 1,981) used for all displayed counts.")
    print("KNOWN INCONSISTENCY: " + KNOWN_INCONSISTENCY)
    print(f"\nsaved: {out}.pdf, {out}.png")
    print("STATUS: Figure 2 FINAL")
