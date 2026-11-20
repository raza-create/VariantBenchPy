"""STEP 15C — Main Figure 3. Frozen Step 11 values (frozen-mapping rerun); no new analysis."""
from __future__ import annotations
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = "data/step11_selection.txt"
COV = {"AlphaMissense": (94.6, 92.3), "REVEL": (98.5, 97.1), "CADD": (99.1, 98.4),
       "SIFT": (92.7, 90.6), "PolyPhen-2": (93.6, 87.5), "EVE": (46.4, 17.2)}
OR = [("label + review-star\nmodel", 4.21, 4.12, 4.29),
      ("+ within-gene\n(cluster-robust)", 2.16, 1.94, 2.40)]
HET = {"AlphaMissense": (-0.023, -0.004, 0.006, 0.032, "0.004"),
       "REVEL": (-0.020, -0.006, 0.002, 0.023, "0.010"),
       "CADD": (-0.059, -0.012, 0.029, 0.068, "<0.001"),
       "SIFT": (-0.043, -0.002, 0.022, 0.059, "<0.001"),
       "PolyPhen-2": (-0.055, -0.004, 0.035, 0.069, "<0.001")}


def parse_report(path=REPORT):
    """Refresh HET and OR from the archived frozen-mapping Step 11 report."""
    if not os.path.exists(path):
        return HET, OR, False
    txt = open(path, encoding="utf-8").read()
    het = {}
    blk = txt[txt.index("EVIDENCE-STRATUM HETEROGENEITY"):]
    for p in HET:
        m = re.search(rf"^\s+{re.escape(p)}\s+([+-][\d.]+)\s+([+-][\d.]+)\s+"
                      rf"\[([+-][\d.]+), ([+-][\d.]+)\]\s+(\S+)", blk, re.M)
        if m:
            het[p] = (float(m.group(1)), float(m.group(2)), float(m.group(3)),
                      float(m.group(4)), m.group(5))
    o = list(OR)
    m = re.search(r"label\+stars: OR=([\d.]+) \[([\d.]+),([\d.]+)\]", txt)
    if m:
        o[0] = ("label + review-star\nmodel", float(m.group(1)), float(m.group(2)), float(m.group(3)))
    m = re.search(r"within-gene control \(cluster-robust[^)]*\): OR=([\d.]+) \[([\d.]+),([\d.]+)\]", txt)
    if m:
        o[1] = ("+ within-gene\n(cluster-robust)", float(m.group(1)),
                float(m.group(2)), float(m.group(3)))
    return (het or HET), o, True


def build(het, orv, out="figures/Figure3_missingness_mechanism"):
    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.45, wspace=0.28)
    fig.suptitle("Figure 3 — Predictor availability is associated with clinical class",
                 fontsize=12, fontweight="bold", y=0.975)

    a = fig.add_subplot(gs[0, 0])
    names = list(COV); x = np.arange(len(names))
    a.bar(x - 0.2, [COV[n][0] for n in names], 0.4, label="P/LP", color="#C44E52")
    a.bar(x + 0.2, [COV[n][1] for n in names], 0.4, label="B/LB", color="#4C72B0")
    for i, n in enumerate(names):
        d = COV[n][0] - COV[n][1]
        a.text(i, max(COV[n]) + 2.5, f"{d:+.1f}pp", ha="center", fontsize=8,
               fontweight="bold" if n == "EVE" else "normal",
               color="#B22222" if n == "EVE" else "#333")
    a.set_xticks(x, names, rotation=20, ha="right", fontsize=8.5)
    a.set_ylabel("coverage (%)"); a.set_ylim(0, 118)
    a.legend(fontsize=8, frameon=False, loc="lower left")
    a.set_title("A. Coverage by clinical class", fontsize=10, loc="left")
    a.annotate("largest difference", xy=(5, 31), xytext=(2.9, 55), fontsize=8,
               color="#B22222", arrowprops=dict(arrowstyle="->", color="#B22222"))

    b = fig.add_subplot(gs[0, 1])
    for i, (lab, o, lo, hi) in enumerate(orv):
        y = 1 - i
        b.plot([lo, hi], [y, y], c="#333", lw=2)
        for e in (lo, hi):
            b.plot([e, e], [y - 0.07, y + 0.07], c="#333", lw=2)
        b.scatter([o], [y], s=90, color="#8172B2", zorder=3)
        b.text(o, y + 0.18, f"OR {o:.2f} [{lo:.2f}, {hi:.2f}]", ha="center",
               fontsize=9, fontweight="bold")
    b.axvline(1, c="grey", ls="--", lw=1.2)
    b.text(1.03, -0.45, "OR = 1", fontsize=8, color="grey")
    b.set_yticks([0, 1], [orv[1][0], orv[0][0]], fontsize=9)
    b.set_xlim(0.8, 4.9); b.set_ylim(-0.6, 1.55)
    b.set_xlabel("odds ratio for EVE availability (P/LP vs B/LB)")
    b.set_title("B. EVE availability association", fontsize=10, loc="left")
    b.text(0.98, 0.06, "ASSOCIATIONAL, NOT CAUSAL", transform=b.transAxes, ha="right",
           fontsize=9, fontweight="bold", color="#B22222",
           bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="#B22222", lw=1.2))

    c = fig.add_subplot(gs[1, :])
    n2 = list(het); x = np.arange(len(n2))
    c.bar(x - 0.2, [het[n][0] for n in n2], 0.4, label="exactly 1-star", color="#DD8452")
    c.bar(x + 0.2, [het[n][1] for n in n2], 0.4, label="≥2-star", color="#55A868")
    c.axhline(0, c="#333", lw=1)
    for i, n in enumerate(n2):
        d1, d2, lo, hi, p = het[n]
        ptxt = f"Holm p {p}" if p.startswith("<") else f"Holm p = {p}"
        c.text(i, 0.007, f"diff [{lo:+.3f}, {hi:+.3f}]\n{ptxt}", ha="center", fontsize=7.6)
        c.text(i - 0.2, d1 - 0.004, f"{d1:+.3f}", ha="center", va="top", fontsize=7.5)
        c.text(i + 0.2, d2 - 0.004, f"{d2:+.3f}", ha="center", va="top", fontsize=7.5)
    c.set_xticks(x, n2, fontsize=9); c.set_ylim(-0.080, 0.032)
    c.set_ylabel("matched ΔAUROC (6-pred − 5-pred)")
    c.legend(fontsize=8.5, frameon=False, loc="lower right")
    c.set_title("C. Evidence-stratum heterogeneity: selection effect attenuates at "
                "higher evidence", fontsize=10, loc="left")

    fig.text(0.5, 0.012, "Predictor availability is associated with clinical class, and the "
             "selection effect is smaller among higher-evidence ClinVar variants.",
             ha="center", fontsize=9.5, style="italic")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    het, orv, parsed = parse_report()
    out = build(het, orv)
    print("FIGURE 3 — GENERATION QC")
    print("=" * 24)
    print(f"source: {'data/step11_selection.txt (parsed, frozen-mapping rerun)' if parsed else 'frozen constants'}")
    print("A. coverage by class (P/LP | B/LB | diff):")
    for n, (p_, b_) in COV.items():
        print(f"   {n:<15}{p_:>5.1f}% | {b_:>5.1f}% | {p_-b_:+5.1f}pp"
              + ("   <- largest" if n == "EVE" else ""))
    print("B. EVE availability OR:")
    for lab, o, lo, hi in orv:
        print(f"   {lab.replace(chr(10),' '):<28}OR {o:.2f} [{lo:.2f}, {hi:.2f}]")
    print("   label: ASSOCIATIONAL, NOT CAUSAL")
    print("C. evidence-stratum heterogeneity (1-star | >=2-star | diff CI | Holm p):")
    for n, (d1, d2, lo, hi, p) in het.items():
        print(f"   {n:<15}{d1:+.3f} | {d2:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p}")
    print(f"\nsaved: {out}.pdf, {out}.png")
