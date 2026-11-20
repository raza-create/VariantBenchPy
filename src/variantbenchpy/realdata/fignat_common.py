"""Shared style, frozen QC targets and loaders for the Nature-format figures."""
from __future__ import annotations
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

STRICT = os.environ.get("VBP_STRICT", "1") == "1"
OUT = "figures/nature"
SRC = os.path.join(OUT, "source_data")
MM = 1 / 25.4
W2 = 180 * MM
PRED5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
PRED6 = PRED5 + ["EVE"]
COL = {"AlphaMissense": "#0072B2", "REVEL": "#D55E00", "CADD": "#009E73",
       "SIFT": "#CC79A7", "PolyPhen-2": "#E69F00", "EVE": "#56B4E9"}
GREY, LGREY, DARK = "#8C8C8C", "#D9D9D9", "#222222"
GENE = "primary_gene_10A"
AM_CUT, EVE_CUT = pd.Timestamp("2023-09-22"), pd.Timestamp("2021-10-27")

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans",
                         "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "lines.linewidth": 0.8, "patch.linewidth": 0.5})

FROZEN_COV = {"AlphaMissense": .931, "REVEL": .975, "CADD": .986, "SIFT": .913,
              "PolyPhen-2": .895, "EVE": .265}
TEMP23 = {"AlphaMissense": (.010, .008, .012), "REVEL": (.011, .009, .013),
          "CADD": (.021, .019, .024), "SIFT": (.023, .020, .026),
          "PolyPhen-2": (.030, .027, .032)}          # Step 13E console output
TEMP21 = {"AlphaMissense": (.0058, .0044, .0072), "REVEL": (.0050, .0035, .0064),
          "CADD": (.0118, .0098, .0138), "SIFT": (.0147, .0123, .0172),
          "PolyPhen-2": (.0199, .0175, .0223)}       # Step 13G console output
GENE_LEVEL = {
    "mean": {"AlphaMissense": .933, "REVEL": .925, "CADD": .899, "SIFT": .879, "PolyPhen-2": .881},
    "median": {"AlphaMissense": .973, "REVEL": .969, "CADD": .934, "SIFT": .917, "PolyPhen-2": .915},
    "unique": {"AlphaMissense": 680, "REVEL": 457, "CADD": 216, "SIFT": 110, "PolyPhen-2": 87},
    "frac": {"AlphaMissense": 944.0, "REVEL": 727.9, "CADD": 389.5, "SIFT": 243.6, "PolyPhen-2": 212.1},
    "fail": {"AlphaMissense": 55, "REVEL": 77, "CADD": 86, "SIFT": 115, "PolyPhen-2": 108},
    "ties": 967, "n": 2517}
DMS_MED = {"AlphaMissense": .519, "REVEL": .440, "SIFT": .409, "CADD": .391, "PolyPhen-2": .349}
CAT_GAP = {"Activity": .148, "Binding": .054, "Expression": .037,
           "OrganismalFitness": .112, "Stability": .014}


class QC:
    def __init__(self):
        self.rows, self.new = [], []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))

    def close(self, name, got, want, tol):
        self.check(name, abs(got - want) <= tol, f"{got:.4f} vs frozen {want:.4f}")

    def report_new(self, name, value):
        self.new.append((name, value))


def letters(fig, pairs):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, s in pairs:
        bb = ax.get_tightbbox(r).transformed(inv)
        fig.text(bb.x0, ax.get_position().y1 + 0.018, s, fontsize=8,
                 fontweight="bold", ha="left", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)


def forest(ax, names, vals, colors, labels=None):
    y = np.arange(len(names))[::-1]
    for i, n in enumerate(names):
        v, lo, hi = vals[n]
        ax.plot([lo, hi], [y[i], y[i]], color=colors[n], lw=1.1, solid_capstyle="butt")
        ax.plot(v, y[i], "o", color=colors[n], ms=3.6, mec="white", mew=0.4, zorder=3)
    ax.set_yticks(y, labels or names)
    ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return y


def ptxt(p):
    return f"P {p}" if str(p).startswith("<") else f"P = {p}"


def load_bench():
    B = pd.read_parquet("data/benchmark_with_temporal.parquet")
    B["first_in_clinvar"] = pd.to_datetime(B["germline_date_created"], errors="coerce")
    return B


def parse_audit(path="data/step12_audit.txt"):
    txt = open(path, encoding="utf-8").read()
    i = txt.find("SECONDARY")
    prim, sec = txt[:i], txt[i:]

    def ret_w(t):
        m = re.search(r"retained #1 in (\d+)/(\d+) configs \(retention ([\d.]+)\)", t)
        w = re.search(r"Kendall's W = ([\d.]+)\s+95% CI \[([\d.]+), ([\d.]+)\]", t)
        return (int(m.group(1)), int(m.group(2)), float(m.group(3)),
                float(w.group(1)), float(w.group(2)), float(w.group(3)))

    blk = prim[prim.index("RANK DISTRIBUTION"):prim.index("MAGNITUDE SENSITIVITY")]
    dist = {}
    for p in PRED5:
        m = re.search(rf"^\s+{re.escape(p)}\s+((?:[\d.]+\s+){{4}}[\d.]+)\s*$", blk, re.M)
        dist[p] = [float(x) for x in m.group(1).split()]
    ub = prim[prim.index("STATISTICAL UNCERTAINTY IN #1"):prim.index("DESIGN SENSITIVITY")]
    boot = {}
    for p in PRED5:
        m = re.search(rf"^\s+{re.escape(p)}\s+([\d.]+)\s+([\d.]+)\s*$", ub, re.M)
        boot[p] = (float(m.group(1)), float(m.group(2)))
    sb = prim[prim.index("DESIGN SENSITIVITY"):]
    shares = {}
    for d in ["evaluation_unit", "evidence", "composition", "coverage"]:
        m = re.search(rf"^\s+{d}\s+[\d.]+\s+\(share (\d+)%\)", sb, re.M)
        shares[d] = int(m.group(1))
    return dict(dist=dist, boot=boot, shares=shares, l5=ret_w(prim), l6=ret_w(sec))


def parse_step11(path="data/step11_selection.txt"):
    t = open(path, encoding="utf-8").read()
    mb = t[t.index("MATCHED SELECTION EXPERIMENT"):t.index("RELATIVE-PERFORMANCE CHANGE")]
    delta = {}
    for p in PRED5:
        m = re.search(rf"^\s+{re.escape(p)}\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+) "
                      rf"\[([+-][\d.]+), ([+-][\d.]+)\]", mb, re.M)
        delta[p] = tuple(float(m.group(k)) for k in (1, 2, 3))
    rb = t[t.index("RELATIVE-PERFORMANCE CHANGE"):t.index("SENSITIVITY (matched")]
    pairs = {}
    for m in re.finditer(r"^\s+(\S+ vs \S+)\s+([+-][\d.]+)\s+\[([+-][\d.]+), ([+-][\d.]+)\]"
                         r"\s+(\S+)\s*$", rb, re.M):
        pairs[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5))
    hb = t[t.index("EVIDENCE-STRATUM HETEROGENEITY"):]
    het = {}
    for p in PRED5:
        m = re.search(rf"^\s+{re.escape(p)}\s+([+-][\d.]+)\s+([+-][\d.]+)\s+"
                      rf"\[([+-][\d.]+), ([+-][\d.]+)\]\s+(\S+)", hb, re.M)
        het[p] = (float(m.group(1)), float(m.group(2)), m.group(5))
    o1 = re.search(r"label\+stars: OR=([\d.]+) \[([\d.]+),([\d.]+)\]", t)
    o2 = re.search(r"cluster-robust[^)]*\): OR=([\d.]+) \[([\d.]+),([\d.]+)\]", t)
    ors = [tuple(float(o.group(k)) for k in (1, 2, 3)) for o in (o1, o2)]
    return dict(delta=delta, pairs=pairs, het=het, ors=ors)


def per_gene_auroc(df, score):
    """Per-gene AUROC (rank-sum identity) for genes with >=10 variants and both classes."""
    d = df[[GENE, "label", score]].dropna().copy()
    d["r"] = d.groupby(GENE)[score].rank(method="average")
    d["rpos"] = np.where(d["label"] == 1, d["r"], 0.0)
    a = d.groupby(GENE).agg(npos=("label", "sum"), n=("label", "size"), srp=("rpos", "sum"))
    a["nneg"] = a["n"] - a["npos"]
    a = a[(a["npos"] > 0) & (a["nneg"] > 0) & (a["n"] >= 10)]
    auc = (a["srp"] - a["npos"] * (a["npos"] + 1) / 2.0) / (a["npos"] * a["nneg"])
    return auc, a["n"]
