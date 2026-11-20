"""STEP 15C — Main Figure 1. Parses frozen Step 12 artifacts; computes nothing new."""
from __future__ import annotations
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AUDIT = "data/step12_audit.txt"
PREDS = ["REVEL", "AlphaMissense", "CADD", "SIFT", "PolyPhen-2"]


def parse_audit(path=AUDIT):
    txt = open(path, encoding="utf-8").read()
    # primary (5-predictor) layer only: cut at the SECONDARY banner
    cut = txt.find("SECONDARY")
    prim = txt[:cut] if cut > 0 else txt

    m = re.search(r"retained #1 in (\d+)/(\d+) configs \(retention ([\d.]+)\)", prim)
    ret_k, ret_n, ret = int(m.group(1)), int(m.group(2)), float(m.group(3))
    winner = re.search(r"Reference #1 = (\S+?);", prim).group(1)

    m = re.search(r"Kendall's W = ([\d.]+)\s+95% CI \[([\d.]+), ([\d.]+)\]", prim)
    w, w_lo, w_hi = float(m.group(1)), float(m.group(2)), float(m.group(3))

    block = prim[prim.index("RANK DISTRIBUTION"):]
    dist = {}
    for p in PREDS:
        r = re.search(rf"^\s+{re.escape(p)}\s+((?:[\d.]+\s+){{4}}[\d.]+)\s*$",
                      block, re.M)
        dist[p] = [float(x) for x in r.group(1).split()]

    b = prim[prim.index("STATISTICAL UNCERTAINTY IN #1"):]
    boot = {}
    for p in PREDS:
        r = re.search(rf"^\s+{re.escape(p)}\s+([\d.]+)\s+([\d.]+)\s*$", b, re.M)
        boot[p] = (float(r.group(1)), float(r.group(2)))

    d = prim[prim.index("DESIGN SENSITIVITY"):]
    shares = {}
    for dim in ["evaluation_unit", "evidence", "composition", "coverage"]:
        r = re.search(rf"^\s+{dim}\s+[\d.]+\s+\(share (\d+)%\)", d, re.M)
        shares[dim] = int(r.group(1))

    return dict(winner=winner, ret=ret, ret_k=ret_k, ret_n=ret_n, w=w, w_lo=w_lo,
                w_hi=w_hi, dist=dist, boot=boot, shares=shares)


def build(a, out="figures/Figure1_design_uncertainty"):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Figure 1 — Benchmark-design uncertainty is distinct from sampling uncertainty",
                 fontsize=12, fontweight="bold", y=0.98)

    # A. rank-distribution heatmap
    M = np.array([a["dist"][p] for p in PREDS])
    im = ax[0, 0].imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax[0, 0].set_xticks(range(5), [f"#{i+1}" for i in range(5)])
    ax[0, 0].set_yticks(range(len(PREDS)), PREDS)
    for i in range(len(PREDS)):
        for j in range(5):
            v = M[i, j]
            if v > 0:
                ax[0, 0].text(j, i, f"{v:.2f}", ha="center", va="center",
                              fontsize=9, color="white" if v > 0.55 else "black")
    ax[0, 0].set_title("A. Rank distribution across 36 specifications", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, label="proportion of specifications")

    # B. whole-ranking concordance vs winner stability
    b = ax[0, 1]
    b.barh([1], [a["ret"]], color="#4C72B0", height=0.45)
    b.barh([0], [a["w"]], xerr=[[a["w"] - a["w_lo"]], [a["w_hi"] - a["w"]]],
           color="#55A868", height=0.45, capsize=5)
    b.set_yticks([0, 1], ["Kendall's W\n(whole ranking)", f"Winner retention\n({a['winner']})"])
    b.set_xlim(0, 1.15)
    b.axvline(1.0, ls=":", c="grey", lw=1)
    b.text(a["w"] - 0.02, 0, f"{a['w']:.2f} [{a['w_lo']:.2f}, {a['w_hi']:.2f}]",
           va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    b.text(a["ret"] - 0.02, 1, f"{a['ret']:.2f} ({a['ret_k']}/{a['ret_n']})",
           va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    b.set_title("B. High whole-ranking concordance, unstable top rank", fontsize=10, loc="left")

    # C. sampling vs design uncertainty
    c = ax[1, 0]
    x = np.arange(len(PREDS))
    c.bar(x - 0.2, [a["boot"][p][1] for p in PREDS], 0.4,
          label="bootstrap P(#1), fixed design", color="#C44E52")
    c.bar(x + 0.2, [a["boot"][p][0] for p in PREDS], 0.4,
          label="proportion of specifications #1", color="#4C72B0")
    c.set_xticks(x, PREDS, rotation=20, ha="right", fontsize=8)
    c.set_ylim(0, 1.1)
    c.legend(fontsize=8, frameon=False)
    c.set_ylabel("probability / proportion")
    c.set_title("C. Sampling uncertainty vs benchmark-design uncertainty",
                fontsize=10, loc="left")

    # D. descriptive design-sensitivity shares
    d = ax[1, 1]
    dims = list(a["shares"])
    vals = [a["shares"][k] for k in dims]
    d.bar(dims, vals, color="#8172B2")
    for i, v in enumerate(vals):
        d.text(i, v + 1, f"{v}%", ha="center", fontsize=9)
    d.set_ylim(0, max(vals) * 1.25 + 5)
    d.set_ylabel("share of rank sensitivity (%)")
    d.set_xticks(range(len(dims)), dims, rotation=20, ha="right", fontsize=8)
    d.set_title("D. Design-sensitivity shares", fontsize=10, loc="left")
    d.text(0.97, 0.93, "DESCRIPTIVE", transform=d.transAxes, ha="right",
           fontsize=10, fontweight="bold", color="#B22222",
           bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="#B22222", lw=1.2))

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    a = parse_audit()
    out = build(a)
    print("FIGURE 1 — GENERATION QC")
    print("=" * 24)
    print(f"source: {AUDIT} (5-predictor primary layer only)")
    print(f"reference winner: {a['winner']}")
    print(f"A. rank distribution rows (must each sum to 1.00):")
    for p in PREDS:
        print(f"   {p:<15}{a['dist'][p]}  sum={sum(a['dist'][p]):.2f}")
    print(f"B. winner retention {a['ret']:.2f} ({a['ret_k']}/{a['ret_n']}); "
          f"W {a['w']:.2f} [{a['w_lo']:.2f}, {a['w_hi']:.2f}]")
    print("C. prop.#1(cfg) | P(#1) boot:")
    for p in PREDS:
        print(f"   {p:<15}{a['boot'][p][0]:.2f} | {a['boot'][p][1]:.2f}")
    print(f"D. shares: {a['shares']}  (sum={sum(a['shares'].values())}%)  [DESCRIPTIVE]")
    print(f"\nsaved: {out}.pdf, {out}.png")
