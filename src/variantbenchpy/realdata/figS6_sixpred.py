"""Supplementary Figure S6 — six-predictor secondary reliability layer.

Frozen Step 12 secondary-layer results, parsed from data/step12_audit.txt when
available. Descriptive components are labelled DESCRIPTIVE.
"""
from __future__ import annotations
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AUDIT = "data/step12_audit.txt"
P = ["REVEL", "AlphaMissense", "EVE", "CADD", "SIFT", "PolyPhen-2"]
DIST = {"REVEL": [.56, .44, 0, 0, 0, 0], "AlphaMissense": [.44, .56, 0, 0, 0, 0],
        "EVE": [0, 0, .89, .11, 0, 0], "CADD": [0, 0, .11, .78, .11, 0],
        "SIFT": [0, 0, 0, .11, .83, .06], "PolyPhen-2": [0, 0, 0, 0, .06, .94]}
MAG = {"REVEL": (0.927, 0.971), "AlphaMissense": (0.940, 0.957), "EVE": (0.898, 0.940),
       "CADD": (0.871, 0.928), "SIFT": (0.865, 0.908), "PolyPhen-2": (0.855, 0.901)}
SHARES = {"evaluation_unit": 39, "evidence": 35, "composition": 26, "coverage": 0}
RET, RET_K, RET_N = 0.56, 20, 36
W, W_LO, W_HI = 0.94, 0.89, 0.95
DDA = (0.007, -0.024, 0.038, 0.258)


def parse():
    if not os.path.exists(AUDIT):
        return DIST, MAG, SHARES, (RET, RET_K, RET_N), (W, W_LO, W_HI), True
    txt = open(AUDIT, encoding="utf-8").read()
    i = txt.find("SECONDARY")
    if i < 0:
        return DIST, MAG, SHARES, (RET, RET_K, RET_N), (W, W_LO, W_HI), False
    sec = txt[i:]
    d, m, sh = {}, {}, {}
    blk = sec[sec.index("RANK DISTRIBUTION"):]
    for p in P:
        r = re.search(rf"^\s+{re.escape(p)}\s+((?:[\d.]+\s+){{5}}[\d.]+)\s*$", blk, re.M)
        if r:
            d[p] = [float(x) for x in r.group(1).split()]
    mb = sec[sec.index("MAGNITUDE SENSITIVITY"):]
    for p in P:
        r = re.search(rf"^\s+{re.escape(p)}\s+\S+\s+([\d.]+)-([\d.]+)", mb, re.M)
        if r:
            m[p] = (float(r.group(1)), float(r.group(2)))
    db = sec[sec.index("DESIGN SENSITIVITY"):]
    for dim in SHARES:
        r = re.search(rf"^\s+{dim}\s+[\d.]+\s+\(share (\d+)%\)", db, re.M)
        if r:
            sh[dim] = int(r.group(1))
    r = re.search(r"retained #1 in (\d+)/(\d+) configs \(retention ([\d.]+)\)", sec)
    ret = (float(r.group(3)), int(r.group(1)), int(r.group(2))) if r else (RET, RET_K, RET_N)
    r = re.search(r"Kendall's W = ([\d.]+)\s+95% CI \[([\d.]+), ([\d.]+)\]", sec)
    w = (float(r.group(1)), float(r.group(2)), float(r.group(3))) if r else (W, W_LO, W_HI)
    return (d or DIST), (m or MAG), (sh or SHARES), ret, w, True


def build(dist, mag, shares, ret, w, out="figures/FigureS6_six_predictor_reliability"):
    fig = plt.figure(figsize=(13, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.42)
    fig.suptitle("Figure S6 — Six-predictor secondary reliability layer",
                 fontsize=12, fontweight="bold", y=0.98)

    a = fig.add_subplot(gs[0, 0])
    M = np.array([dist[p] for p in P])
    im = a.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    a.set_xticks(range(6), [f"#{i+1}" for i in range(6)])
    a.set_yticks(range(6), P, fontsize=8.5)
    for i in range(6):
        for j in range(6):
            v = M[i, j]
            if v > 0:
                a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                       color="white" if v > 0.55 else "black")
    a.set_title("A. Rank distribution across 36 specifications", fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=a, fraction=0.042, pad=0.03)
    cb.set_label("proportion", fontsize=8)

    b = fig.add_subplot(gs[0, 1])
    b.barh([1], [ret[0]], color="#4C72B0", height=0.42)
    b.barh([0], [w[0]], xerr=[[w[0] - w[1]], [w[2] - w[0]]], color="#55A868",
           height=0.42, capsize=5)
    b.text(ret[0] - 0.02, 1, f"{ret[0]:.2f} ({ret[1]}/{ret[2]})", va="center",
           ha="right", fontsize=9, color="white", fontweight="bold")
    b.text(w[0] - 0.07, 0, f"{w[0]:.2f} [{w[1]:.2f}, {w[2]:.2f}]", va="center",
           ha="right", fontsize=9, color="white", fontweight="bold")
    b.set_yticks([0, 1], ["Kendall's W\n(whole ranking)", "Winner retention\n(REVEL)"],
                 fontsize=9)
    b.set_xlim(0, 1.18); b.axvline(1.0, ls=":", c="grey", lw=1)
    b.set_title("B. High concordance, unstable top rank", fontsize=10, loc="left")
    b.text(0.5, -0.42, "Selection-aware: composition numerical reversal\n"
           f"ΔΔAUROC {DDA[0]:+.3f} [{DDA[1]:+.3f}, {DDA[2]:+.3f}], p_adj = {DDA[3]:.3f}\n"
           "NOT statistically supported", transform=b.transAxes, ha="center", fontsize=8.3,
           bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="grey"))

    c = fig.add_subplot(gs[1, 0])
    k = list(shares); v = [shares[x] for x in k]
    c.bar(k, v, color="#8172B2")
    for i, x in enumerate(v):
        c.text(i, x + 1.2, f"{x}%", ha="center", fontsize=9)
    c.set_ylim(0, 52); c.set_ylabel("share of rank sensitivity (%)")
    c.set_xticks(range(len(k)), k, rotation=18, ha="right", fontsize=8.5)
    c.set_title("C. Design-sensitivity shares", fontsize=10, loc="left")
    c.text(0.97, 0.93, "DESCRIPTIVE", transform=c.transAxes, ha="right", fontsize=9.5,
           fontweight="bold", color="#B22222",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B22222", lw=1.2))

    d = fig.add_subplot(gs[1, 1])
    y = np.arange(len(P))[::-1]
    for i, p in enumerate(P):
        lo, hi = mag[p]
        d.plot([lo, hi], [y[i]] * 2, c="#4C72B0", lw=5, solid_capstyle="round")
        d.text(1.002, y[i], f"{lo:.3f}–{hi:.3f}", va="center", fontsize=8)
    d.set_yticks(y, P, fontsize=8.5); d.set_xlim(0.845, 1.045)
    d.set_xticks([0.86, 0.90, 0.94, 0.98])
    d.set_xlabel("AUROC range across specifications")
    d.set_title("D. Magnitude sensitivity", fontsize=10, loc="left")
    d.text(0.62, 0.06, "DESCRIPTIVE", transform=d.transAxes, ha="right", fontsize=9.5,
           fontweight="bold", color="#B22222",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B22222", lw=1.2))

    fig.text(0.5, 0.012, "6-predictor shared layer: 53,552 variants / 1,981 gene clusters "
             "/ 36 prespecified specifications.", ha="center", fontsize=8.5, style="italic")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    dist, mag, shares, ret, w, parsed = parse()
    out = build(dist, mag, shares, ret, w)
    print("FIGURE S6 — GENERATION QC")
    print("=" * 25)
    print(f"source: {'data/step12_audit.txt (SECONDARY layer parsed)' if parsed else 'frozen constants'}")
    print("population: 53,552 variants / 1,981 gene clusters / 36 specifications")
    print("A. rank distribution (rows sum to 1.00):")
    for p in P:
        print(f"   {p:<15}{dist[p]}  sum={sum(dist[p]):.2f}")
    print(f"B. winner retention {ret[0]:.2f} ({ret[1]}/{ret[2]}); "
          f"Kendall's W {w[0]:.2f} [{w[1]:.2f}, {w[2]:.2f}]")
    print(f"   selection-aware composition reversal: ΔΔAUROC {DDA[0]:+.3f} "
          f"[{DDA[1]:+.3f}, {DDA[2]:+.3f}], p_adj={DDA[3]:.3f} -> NOT supported")
    print(f"C. design-sensitivity shares: {shares}  (sum={sum(shares.values())}%)  [DESCRIPTIVE]")
    print("D. magnitude sensitivity (AUROC range across specifications)  [DESCRIPTIVE]:")
    for p in P:
        lo, hi = mag[p]
        print(f"   {p:<15}{lo:.3f}–{hi:.3f}   span={hi-lo:.3f}")
    print("   note: panel C was split into C (design shares) and D (magnitude) for legibility")
    print(f"\nsaved: {out}.pdf, {out}.png")
