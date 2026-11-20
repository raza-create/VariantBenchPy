"""Supplementary Figure S5 — EVE mapping cascade and selective coverage.

Frozen Step 10/11 EVE results over the 222,152 dbNSFP-matched benchmark variants.
Panel D shows EVE coverage by evidence level (the downstream selection-effect
version of this panel appears as Figure 3C and is not duplicated here).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASCADE = [("EVE_exact_scored", 59695, "#55A868"),
           ("EVE_exact_score_missing", 10196, "#DD8452"),
           ("EVE_AA_or_alt_mismatch", 9833, "#8172B2"),
           ("EVE_position_absent", 243, "#937860"),
           ("EVE_protein_not_covered", 100452, "#C44E52"),
           ("no_protein_mapping", 41733, "#4C72B0")]
CLASS_COV = {"P/LP": 46.4, "B/LB": 17.2}
OR = [("label + review-star model", 4.21, 4.12, 4.29),
      ("+ within-gene\n(cluster-robust)", 2.16, 1.94, 2.40)]
COV_EVID = [("all", 26.5), ("≥1-star", 26.3), ("≥2-star", 33.4)]


def build(out="figures/FigureS5_eve_mapping"):
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.2))
    fig.suptitle("Figure S5 — EVE mapping cascade and selective coverage",
                 fontsize=12, fontweight="bold", y=0.98)

    a = ax[0, 0]
    tot = sum(c[1] for c in CASCADE)
    y = np.arange(len(CASCADE))[::-1]
    a.barh(y, [c[1] for c in CASCADE], color=[c[2] for c in CASCADE], height=0.62)
    for i, (n, v, _) in enumerate(CASCADE):
        a.text(v + 2500, y[i], f"{v:,} ({v/tot:.1%})", va="center", fontsize=8)
    a.set_yticks(y, [c[0] for c in CASCADE], fontsize=8)
    a.set_xlim(0, 148000); a.set_xlabel("variants")
    a.set_xticks([0, 50000, 100000], ["0", "50k", "100k"])
    a.set_title(f"A. EVE mapping cascade ({tot:,} dbNSFP-matched variants)",
                fontsize=10, loc="left")
    a.text(0.97, 0.93, f"usable EVE score\n{CASCADE[0][1]:,} ({CASCADE[0][1]/tot:.1%})",
           transform=a.transAxes, ha="right", va="top", fontsize=8.5, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.35", fc="#eaf4ea", ec="#55A868"))

    b = ax[0, 1]
    vals = [CLASS_COV["P/LP"], CLASS_COV["B/LB"]]
    b.bar([0, 1], vals, color=["#C44E52", "#4C72B0"], width=0.5)
    for i, v in enumerate(vals):
        b.text(i, v + 1.4, f"{v}%", ha="center", fontsize=11, fontweight="bold")
    b.annotate("", xy=(0.62, vals[1]), xytext=(0.62, vals[0]),
               arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
    b.text(0.66, (vals[0] + vals[1]) / 2, f"+{vals[0]-vals[1]:.1f} pp",
           fontsize=10, fontweight="bold", color="#B22222", va="center")
    b.set_xticks([0, 1], ["P/LP", "B/LB"], fontsize=10)
    b.set_ylabel("EVE coverage (%)"); b.set_ylim(0, 60); b.set_xlim(-0.55, 1.55)
    b.set_title("B. EVE coverage by clinical class", fontsize=10, loc="left")

    c = ax[1, 0]
    for i, (lab, o, lo, hi) in enumerate(OR):
        yy = 1 - i
        c.plot([lo, hi], [yy, yy], c="#333", lw=2)
        for e in (lo, hi):
            c.plot([e, e], [yy - 0.07, yy + 0.07], c="#333", lw=2)
        c.scatter([o], [yy], s=90, color="#8172B2", zorder=3)
        c.text(o, yy + 0.18, f"OR {o:.2f} [{lo:.2f}, {hi:.2f}]", ha="center",
               fontsize=9, fontweight="bold")
    c.axvline(1, c="grey", ls="--", lw=1.2)
    c.set_yticks([0, 1], [OR[1][0], OR[0][0]], fontsize=9)
    c.set_xlim(0.8, 4.9); c.set_ylim(-0.6, 1.55)
    c.set_xlabel("odds ratio for EVE availability (P/LP vs B/LB)")
    c.set_title("C. EVE availability association", fontsize=10, loc="left")
    c.text(0.98, 0.06, "ASSOCIATIONAL, NOT CAUSAL", transform=c.transAxes, ha="right",
           fontsize=9, fontweight="bold", color="#B22222",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B22222", lw=1.2))

    d = ax[1, 1]
    d.bar(range(len(COV_EVID)), [v for _, v in COV_EVID], color="#55A868", width=0.5)
    for i, (l, v) in enumerate(COV_EVID):
        d.text(i, v + 0.9, f"{v}%", ha="center", fontsize=10, fontweight="bold")
    d.set_xticks(range(len(COV_EVID)), [l for l, _ in COV_EVID], fontsize=9.5)
    d.set_ylabel("EVE coverage (%)"); d.set_ylim(0, 44)
    d.set_xlabel("ClinVar evidence level")
    d.set_title("D. EVE coverage by evidence level", fontsize=10, loc="left")

    fig.text(0.5, 0.012, "EVE availability is strongly associated with clinical class and "
             "remains so after conditioning on gene and evidence.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.955])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = build()
    tot = sum(c[1] for c in CASCADE)
    print("FIGURE S5 — GENERATION QC")
    print("=" * 25)
    print(f"A. mapping cascade over {tot:,} dbNSFP-matched variants:")
    for n, v, _ in CASCADE:
        print(f"   {n:<28}{v:>9,}  ({v/tot:6.2%})")
    print(f"   sum = {tot:,} (matches dbNSFP-matched universe: {tot == 222152})")
    print(f"   usable EVE score: {CASCADE[0][1]:,} ({CASCADE[0][1]/tot:.1%})")
    print("B. coverage by clinical class:")
    print(f"   P/LP {CLASS_COV['P/LP']}%   B/LB {CLASS_COV['B/LB']}%   "
          f"diff +{CLASS_COV['P/LP']-CLASS_COV['B/LB']:.1f} pp")
    print("C. availability association:")
    for lab, o, lo, hi in OR:
        print(f"   {lab.replace(chr(10), ' '):<30}OR {o:.2f} [{lo:.2f}, {hi:.2f}]")
    print("   label: ASSOCIATIONAL, NOT CAUSAL")
    print("D. coverage by evidence level:")
    for l, v in COV_EVID:
        print(f"   {l:<10}{v}%")
    print("   (Step-11 evidence-stratum effect is shown in Figure 3C; not duplicated here)")
    print(f"\nsaved: {out}.pdf, {out}.png")
