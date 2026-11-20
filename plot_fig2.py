import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SRC  = pathlib.Path("figures/nature/source_data")
GRID = SRC / "Fig2a_design_grid_recomputed.csv"
BOOT = pathlib.Path("data/reliability_bootstrap_5pred.parquet")   # adjust if you have it
OUT  = pathlib.Path("Fig/fig2.png")

AM, RV = "#0072B2", "#D55E00"
PREDS  = ["AlphaMissense","REVEL","CADD","SIFT","PolyPhen-2"]
COLOUR = {"AlphaMissense":AM,"REVEL":RV,"CADD":"#4D4D4D","SIFT":"#8A8A8A","PolyPhen-2":"#B8B8B8"}

plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42,"font.size":7,
    "axes.labelsize":7.5,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.fontsize":6.5,"axes.linewidth":0.6})

g = pd.read_csv(GRID)
g["unit"] = pd.Categorical(g["unit"], ["pooled","equal-gene"], ordered=True)
g["comp"] = pd.Categorical(g["comp"], ["all","multi","single"], ordered=True)
g["evid"] = pd.Categorical(g["evid"], ["all",">=1",">=2"], ordered=True)
g = g.sort_values(["unit","comp","evid"]).reset_index(drop=True)
g["x"] = np.arange(1, len(g)+1)

pooled, eqg = g[g.unit=="pooled"], g[g.unit=="equal-gene"]
assert len(g)==18
assert (pooled.rev_minus_am>0).sum()==9,  "pooled block should be 9/9 REVEL"
assert (eqg.rev_minus_am>0).sum()==3,     "equal-gene block should be 3/9 REVEL"
assert (g.leader=="REVEL").sum()==12,     "REVEL should lead 12 of 18"
assert pooled.REVEL.min() > eqg.REVEL.max(), "blocks look swapped"
print("guards passed: 12/18, 9/9 pooled, 3/9 equal-gene")

BOUND = 9.5
EVID = {"all":"All", ">=1":"\u22651 star", ">=2":"\u22652 stars"}
COMP = {"all":"All genes","multi":"Multi-gene","single":"Single-gene"}

fig = plt.figure(figsize=(6.85, 5.0))
outer = GridSpec(2, 5, height_ratios=[1.62, 1.0], figure=fig)
inner = outer[0,:].subgridspec(2, 1, height_ratios=[2.0,1.0], hspace=0.10)
a1, a2 = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
bx = fig.add_subplot(outer[1,0:3]); cx = fig.add_subplot(outer[1,3:5])
fig.subplots_adjust(left=0.085, right=0.90, top=0.95, bottom=0.095,
                    hspace=0.30, wspace=1.05)

for p in PREDS:
    for i0 in range(0, 18, 3):
        blk = g.iloc[i0:i0+3]
        a1.plot(blk.x, blk[p], "-o", color=COLOUR[p], lw=1.0, ms=2.6, mec="none")
ends = sorted(((g[p].iloc[-1], p) for p in PREDS), reverse=True)
SEP, placed = 0.0075, []
for yv, p in ends:
    if placed and yv > placed[-1][0] - SEP:
        yv = placed[-1][0] - SEP
    placed.append((yv, p))
for yv, p in placed:
    a1.annotate(p, xy=(18.4, yv), fontsize=6.5, va="center",
                color=COLOUR[p], annotation_clip=False)
a1.axvline(BOUND, color="#333333", lw=1.0)
a1.set_ylabel("AUROC"); a1.set_xlim(0.4,18.6); a1.set_ylim(0.855,0.992)
a1.set_xticks(g.x); a1.set_xticklabels([]); a1.tick_params(axis="x", length=0)
a1.text(5.0, 0.986, "Pooled variant", ha="center", fontsize=7.5, weight="bold")
a1.text(14.0, 0.986, "Equal gene",   ha="center", fontsize=7.5, weight="bold")
for s in ("top","right"): a1.spines[s].set_visible(False)

a2.bar(g.x, g.rev_minus_am, width=0.62,
       color=[RV if v>0 else AM for v in g.rev_minus_am], zorder=3)
a2.axhline(0, color="#333333", lw=0.8, zorder=4)
a2.axvline(BOUND, color="#333333", lw=1.0)
a2.set_ylabel("$\\Delta$AUROC\nREVEL $-$ AM", fontsize=6.8, labelpad=2)
a2.set_xlim(0.4,18.6); a2.set_ylim(-0.018,0.032)
a2.set_xticks(g.x); a2.set_xticklabels([])
for s in ("top","right"): a2.spines[s].set_visible(False)
SHORT = {"all":"All", ">=1":"\u22651", ">=2":"\u22652"}
for _, r in g.iterrows():
    a2.annotate(SHORT[str(r.evid)], xy=(r.x,-0.0196), ha="center", va="top",
                fontsize=5.6, annotation_clip=False, color="#666666")
for i in range(0, 18, 3):
    blk = g.iloc[i:i+3]
    a2.annotate(COMP[str(blk.comp.iloc[0])], xy=(blk.x.mean(),-0.0272),
                ha="center", va="top", fontsize=6.2,
                annotation_clip=False, color="#333333")
    a2.plot([blk.x.min()-0.35, blk.x.max()+0.35], [-0.0257]*2,
            color="#AAAAAA", lw=0.5, clip_on=False)

lo, hi = -0.018, 0.036
BOOT_CSV = SRC / "Fig2b_bootstrap_draws.csv"
if BOOT.exists() or BOOT_CSV.exists():
    if BOOT.exists():
        d = pd.read_parquet(BOOT)
        d = (d["REVEL"] - d["AlphaMissense"]).to_numpy()
    else:
        d = pd.read_csv(BOOT_CSV)["revel_minus_alphamissense"].to_numpy()
    from scipy.stats import gaussian_kde
    xs = np.linspace(lo, hi, 400); dn = gaussian_kde(d)(xs)
    bx.fill_between(xs, 0.08, 0.08 + dn/dn.max()*0.42,
                    color="#D8D8D8", edgecolor="#555555", lw=0.7)
    q1, q3 = np.quantile(d, [0.025, 0.975])
    bx.plot([q1,q3],[0.56,0.56], color="#555555", lw=0.8)
    bx.set_ylim(-0.65, 0.78)
    bx.annotate(f"Sampling, 95% interval\nwidth = {q3-q1:.4f}",
                xy=(q3+0.0012,0.56), ha="left", va="center", fontsize=6.5, color="#444444")
else:
    bx.annotate(f"bootstrap replicates not found at {BOOT}", xy=(0.5,0.72),
                xycoords="axes fraction", ha="center", fontsize=6.5, color="#999999")

for _, r in g.iterrows():
    bx.plot([r.rev_minus_am]*2, [-0.40,-0.14],
            color=RV if r.rev_minus_am>0 else AM, lw=1.1)
span = g.rev_minus_am.max() - g.rev_minus_am.min()
bx.plot([g.rev_minus_am.min(), g.rev_minus_am.max()], [-0.47,-0.47], color="#555555", lw=0.8)
bx.annotate(f"Across 18 designs: width = {span:.3f}", xy=(0.0012,-0.54),
            ha="left", va="top", fontsize=6.5, color="#444444")
bx.axvline(0, color="#333333", lw=0.8)
bx.set_xlim(lo,hi); bx.set_ylim(-0.65,0.65); bx.set_yticks([])
bx.set_xlabel("$\\Delta$AUROC (REVEL $-$ AlphaMissense)")
for s in ("top","right","left"): bx.spines[s].set_visible(False)

dims = ["Evaluation unit","Evidence","Composition"]; five=[74,21,5]; six=[39,35,26]
y = np.arange(3)
cx.barh(y+0.19, five, height=0.34, color="#555555",
        label="5 predictors \u2014 12/18 retained, $W$ = 0.91")
cx.barh(y-0.19, six,  height=0.34, color="#B0B0B0",
        label="6 predictors \u2014 10/18 retained, $W$ = 0.94")
for i,(a,b) in enumerate(zip(five,six)):
    cx.annotate(f"{a}%", xy=(a+1.5,i+0.19), va="center", fontsize=6.5)
    cx.annotate(f"{b}%", xy=(b+1.5,i-0.19), va="center", fontsize=6.5)
cx.set_yticks(y); cx.set_yticklabels(dims); cx.invert_yaxis()
cx.set_xlim(0,82); cx.set_xlabel("Share of rank sensitivity (%, descriptive)")
cx.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01),
          frameon=False, borderaxespad=0.0, handlelength=1.4,
          handletextpad=0.5, labelspacing=0.3)
for s in ("top","right"): cx.spines[s].set_visible(False)

for ax, ltr in [(a1,"a"),(bx,"b"),(cx,"c")]:
    ax.annotate(ltr, xy=(-0.085,1.03), xycoords="axes fraction", fontsize=9, weight="bold")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
print("written ->", OUT, "and", OUT.with_suffix(".pdf"))
