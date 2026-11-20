"""Supplementary Figure S3 — ClinVar cohort construction and analysis populations.

Frozen Step-10A counts with MANE-informed gene clusters. Descriptive only.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXCL = [("VUS / not provided", 2139383), ("Conflicting P/B", 110991),
        ("No germline class.", 57382), ("Other or compound", 2310)]
CLASS = {"P/LP": (71926, 31.9), "B/LB": (153282, 68.1)}
STARS = {0: 17459, 1: 142306, 2: 60554, 3: 4881, 4: 8}
POPS = [("Full benchmark", 225208, 15249), ("5-predictor shared", 187155, 13599),
        ("6-predictor shared", 53552, 1981)]


def build(out="figures/FigureS3_clinvar_cohort"):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.8))
    fig.suptitle("Figure S3 — ClinVar cohort construction and analysis populations",
                 fontsize=12, fontweight="bold", y=0.98)

    a = ax[0, 0]; a.axis("off")
    a.add_patch(plt.Rectangle((0.04, 0.80), 0.52, 0.15, fc="#4C72B0", alpha=0.9,
                              transform=a.transAxes))
    a.text(0.30, 0.875, "ClinVar GRCh38\nmissense records", ha="center", va="center",
           fontsize=9, color="white", fontweight="bold", transform=a.transAxes)
    a.annotate("", xy=(0.30, 0.60), xytext=(0.30, 0.79), xycoords=a.transAxes,
               textcoords=a.transAxes, arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
    txt = "EXCLUDED\n" + "\n".join(f"  {n:<24}{v:>10,}" for n, v in EXCL)
    a.text(0.62, 0.74, txt, fontsize=7.6, family="monospace", va="top",
           transform=a.transAxes,
           bbox=dict(boxstyle="round,pad=0.4", fc="#fbe9e7", ec="#C44E52"))
    a.add_patch(plt.Rectangle((0.04, 0.42), 0.52, 0.17, fc="#55A868", alpha=0.9,
                              transform=a.transAxes))
    a.text(0.30, 0.505, "FINAL BENCHMARK\n225,208 variants\n15,249 gene clusters",
           ha="center", va="center", fontsize=9, color="white", fontweight="bold",
           transform=a.transAxes)
    a.text(0.04, 0.30, f"Total excluded: {sum(v for _, v in EXCL):,} records",
           fontsize=8.5, transform=a.transAxes, color="#333")
    a.set_title("A. Cohort construction", fontsize=10, loc="left")

    b = ax[0, 1]
    b.barh([1], [CLASS["B/LB"][0]], color="#4C72B0", height=0.5)
    b.barh([0], [CLASS["P/LP"][0]], color="#C44E52", height=0.5)
    for y, k in [(1, "B/LB"), (0, "P/LP")]:
        n, pct = CLASS[k]
        b.text(n / 2, y, f"{n:,}\n{pct}%", ha="center", va="center", color="white",
               fontsize=10, fontweight="bold")
    b.set_yticks([0, 1], ["P/LP", "B/LB"], fontsize=10)
    b.set_xlabel("variants"); b.set_xlim(0, 175000)
    b.set_xticks([0, 50000, 100000, 150000], ["0", "50k", "100k", "150k"])
    b.set_title("B. Clinical class composition", fontsize=10, loc="left")

    c = ax[1, 0]
    ks = sorted(STARS); vals = [STARS[k] for k in ks]
    c.bar([str(k) for k in ks], vals, color="#8172B2")
    for i, v in enumerate(vals):
        c.text(i, max(v, 0) + 3800, f"{v:,}", ha="center", fontsize=8.5)
    c.set_ylim(0, 172000); c.set_xlabel("ClinVar review stars"); c.set_ylabel("variants")
    c.set_yticks([0, 50000, 100000, 150000], ["0", "50k", "100k", "150k"])
    c.set_title("C. Review-star distribution", fontsize=10, loc="left")

    d = ax[1, 1]
    x = np.arange(len(POPS))
    d.bar(x, [p[1] for p in POPS], 0.55, color=["#4C72B0", "#55A868", "#C44E52"])
    for i, (lab, nv, ng) in enumerate(POPS):
        d.text(i, nv + 7000, f"{nv:,}\n{ng:,} genes", ha="center", fontsize=8.5)
    d.set_xticks(x, [p[0] for p in POPS], rotation=12, ha="right", fontsize=9)
    d.set_ylabel("variants"); d.set_ylim(0, 280000)
    d.set_yticks([0, 100000, 200000], ["0", "100k", "200k"])
    d.set_title("D. Analysis populations", fontsize=10, loc="left")

    fig.text(0.5, 0.012, "All counts from the frozen Step-10A cohort with MANE-informed "
             "gene clusters. Descriptive; no statistics.", ha="center", fontsize=8,
             style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = build()
    tot = sum(v for _, v in EXCL)
    print("FIGURE S3 — GENERATION QC")
    print("=" * 25)
    print("A. exclusions:")
    for n, v in EXCL:
        print(f"   {n:<26}{v:>10,}")
    print(f"   {'TOTAL EXCLUDED':<26}{tot:>10,}")
    print("   final benchmark: 225,208 variants / 15,249 frozen gene clusters")
    print("B. clinical class:")
    for k, (n, pct) in CLASS.items():
        print(f"   {k:<6}{n:>8,}  ({pct}%)")
    print(f"   sum = {sum(v[0] for v in CLASS.values()):,} "
          f"(matches benchmark: {sum(v[0] for v in CLASS.values()) == 225208})")
    print("C. review stars:")
    for k in sorted(STARS):
        print(f"   {k}-star{STARS[k]:>10,}")
    print(f"   sum = {sum(STARS.values()):,} "
          f"(matches benchmark: {sum(STARS.values()) == 225208})")
    print("D. analysis populations:")
    for lab, nv, ng in POPS:
        print(f"   {lab:<22}{nv:>9,} variants / {ng:>7,} gene clusters")
    print(f"\nsaved: {out}.pdf, {out}.png")
