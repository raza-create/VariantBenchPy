import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SRC = pathlib.Path("figures/nature/source_data")
OUT = pathlib.Path("Fig/fig5.png")
P   = ["AlphaMissense","REVEL","CADD","SIFT","PolyPhen-2"]
AM, RV = "#0072B2", "#D55E00"
GREY = {"CADD":"#4D4D4D","SIFT":"#8A8A8A","PolyPhen-2":"#B8B8B8"}
COL  = {"AlphaMissense":AM,"REVEL":RV,**GREY}
INK, MID, LIGHT = "#1A1A1A", "#666666", "#AAAAAA"

plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42,"font.size":7,
    "axes.labelsize":7.5,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.fontsize":6.5,"axes.linewidth":0.6})

W   = pd.read_csv(SRC/"Fig5a_per_gene_auroc.csv", index_col=0)
won = pd.read_csv(SRC/"Fig5b_genes_won.csv", index_col=0).reindex(P)
low = pd.read_csv(SRC/"Fig5c_low_auroc.csv", index_col=0).reindex(P)
dec = pd.read_csv(SRC/"Fig5de_decomposition.csv", index_col=0).reindex(P)
PI  = float(dec["pi"].iloc[0]); SHARE = (1 - PI) * 100

assert len(W) == 2517, f"expected 2,517 genes, got {len(W)}"
assert abs(W["AlphaMissense"].mean() - 0.933) < 0.001
assert abs(W["REVEL"].mean()         - 0.925) < 0.001
assert int(won.loc["AlphaMissense","sole_best"]) == 680
assert abs(won["fractional"].sum() - 2517) < 1.0
assert abs(SHARE - 99.9672) < 0.01, f"between-gene share is {SHARE:.4f}, expected 99.9672"
print(f"guards passed; between-gene share {SHARE:.4f}%")

fig = plt.figure(figsize=(6.85,4.6))
gs  = GridSpec(2,6, height_ratios=[1.0,1.0], figure=fig)
ax_a=fig.add_subplot(gs[0,0:4]); ax_b=fig.add_subplot(gs[0,4:6])
ax_c=fig.add_subplot(gs[1,0:2]); ax_d=fig.add_subplot(gs[1,2:4])
ax_e=fig.add_subplot(gs[1,4:6])
fig.subplots_adjust(left=0.085,right=0.98,top=0.93,bottom=0.155,
                    hspace=0.78,wspace=1.05)

# (a) per-gene AUROC distributions
bp = ax_a.boxplot([W[p].dropna() for p in P], patch_artist=True, widths=0.55,
                  showfliers=False, medianprops=dict(color=INK, lw=1.0),
                  whiskerprops=dict(color=MID, lw=0.7),
                  capprops=dict(color=MID, lw=0.7))
for patch,p in zip(bp["boxes"],P):
    patch.set(facecolor=COL[p], alpha=0.55, edgecolor=MID, lw=0.7)
ax_a.scatter(range(1,6),[W[p].mean() for p in P], marker="D", s=14,
             color=INK, zorder=4)
ax_a.set_xticks(range(1,6)); ax_a.set_xticklabels(P, rotation=90, ha="center", fontsize=6.2)
ax_a.set_ylim(0.58,1.015); ax_a.set_ylabel("Per-gene AUROC")
ax_a.annotate(f"{len(W):,} genes with $\\geq$10 variants and both classes; axis truncated at 0.58",
              xy=(0.02,0.045), xycoords="axes fraction", fontsize=6.2, color="#555555")
for s in ("top","right"): ax_a.spines[s].set_visible(False)

# (b) genes won: sole best plus share of ties
x = np.arange(5)
ax_b.bar(x, won["sole_best"], width=0.6, color=[COL[p] for p in P], zorder=3)
ax_b.bar(x, won["fractional"]-won["sole_best"], width=0.6, bottom=won["sole_best"],
         color=[COL[p] for p in P], alpha=0.38, zorder=3)
for i,p in enumerate(P):
    ax_b.annotate(f"{won.loc[p,'fractional']:.0f}", xy=(i,won.loc[p,"fractional"]+22),
                  ha="center", fontsize=6.2)
ax_b.set_xticks(x); ax_b.set_xticklabels(P, rotation=90, ha="center", fontsize=6.2)
ax_b.set_ylim(0,1080); ax_b.set_ylabel("Genes won")
ax_b.annotate("solid, sole best\nshaded, share of 967 ties", xy=(0.40,0.78),
              xycoords="axes fraction", fontsize=6.0, color="#555555")
for s in ("top","right"): ax_b.spines[s].set_visible(False)

# (c) genes below 0.60
ax_c.bar(x, low.iloc[:,0], width=0.6, color=[COL[p] for p in P], zorder=3)
for i,p in enumerate(P):
    ax_c.annotate(f"{int(low.iloc[i,0])}", xy=(i,low.iloc[i,0]+3), ha="center", fontsize=6.2)
ax_c.set_xticks(x); ax_c.set_xticklabels(P, rotation=90, ha="center", fontsize=6.2)
ax_c.set_ylim(0,132); ax_c.set_ylabel("Genes with AUROC < 0.60")
for s in ("top","right"): ax_c.spines[s].set_visible(False)

# (d) within- vs between-gene pairs
y = np.arange(5)
for i,p in enumerate(P):
    ax_d.plot([dec.loc[p,"within"],dec.loc[p,"between"]],[i,i],color=LIGHT,lw=1.3,zorder=2)
    ax_d.scatter(dec.loc[p,"within"], i, s=26, facecolor="white", edgecolor=COL[p],
                 lw=1.2, zorder=3)
    ax_d.scatter(dec.loc[p,"between"], i, s=26, color=COL[p], zorder=3)
ax_d.set_yticks(y); ax_d.set_yticklabels([]); ax_d.invert_yaxis()
ax_d.set_ylim(4.65,-0.80)
for i,p in enumerate(P):
    ax_d.annotate(p, xy=(0.015,i-0.45), xycoords=("axes fraction","data"),
                  ha="left", va="center", fontsize=6.0, color="#333333")
ax_d.set_xlim(0.868,0.978); ax_d.set_xlabel("AUROC over variant pairs")
ax_d.scatter([],[],s=26,facecolor="white",edgecolor=INK,lw=1.2,label="within-gene")
ax_d.scatter([],[],s=26,color=INK,label="between-gene")
ax_d.legend(loc="lower center", bbox_to_anchor=(0.5,1.01), ncol=2,
            frameon=False, borderaxespad=0.0, handletextpad=0.3)
for s in ("top","right"): ax_d.spines[s].set_visible(False)

# (e) the decomposition
am, rv = dec.loc["AlphaMissense"], dec.loc["REVEL"]
vals = [am["within"]-rv["within"], am["between"]-rv["between"], am["pooled"]-rv["pooled"]]
ax_e.bar(range(3), vals, width=0.58, color="#4D4D4D", zorder=3)
for i,v in enumerate(vals):
    ax_e.annotate(f"{v:+.4f}", xy=(i, v + (0.0012 if v>0 else -0.0012)),
                  ha="center", va="bottom" if v>0 else "top", fontsize=6.2)
ax_e.axhline(0, color="#333333", lw=0.8, zorder=4)
ax_e.set_xticks(range(3))
ax_e.set_xticklabels(["Within\ngene","Between\ngenes","Pooled"], fontsize=6.5)
ax_e.set_ylim(-0.016,0.018)
ax_e.set_ylabel("AlphaMissense $-$ REVEL", fontsize=7.0, labelpad=1)
ax_e.annotate(f"{SHARE:.2f}% of pairs\nare between genes", xy=(0.97,0.97),
              xycoords="axes fraction", ha="right", va="top", fontsize=6.0, color="#555555")
for s in ("top","right"): ax_e.spines[s].set_visible(False)

for ax,l in [(ax_a,"a"),(ax_b,"b"),(ax_c,"c"),(ax_d,"d"),(ax_e,"e")]:
    ax.annotate(l, xy=(-0.12,1.05), xycoords="axes fraction", fontsize=9, weight="bold")

OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=600,bbox_inches="tight",pad_inches=0.02)
fig.savefig(OUT.with_suffix(".pdf"),bbox_inches="tight",pad_inches=0.02)
print("written ->",OUT,"and",OUT.with_suffix(".pdf"))
