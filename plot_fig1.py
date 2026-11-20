import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SRC = pathlib.Path("figures/nature/source_data")
OUT = pathlib.Path("Fig/fig1.png")
COL = {"CADD":"#4D4D4D","REVEL":"#D55E00","AlphaMissense":"#0072B2",
       "SIFT":"#8A8A8A","PolyPhen-2":"#B8B8B8","EVE":"#56B4E9"}
STARS = ["#E8E8E8","#BDBDBD","#7F7F7F","#3F3F3F"]

plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":7,
    "axes.labelsize":7.5,"xtick.labelsize":7,"ytick.labelsize":7,
    "axes.linewidth":0.6})

cov = pd.read_csv(SRC/"Fig1b_coverage.csv", index_col=0).sort_values("overall", ascending=False)
ct  = pd.read_csv(SRC/"Fig1c_review_status.csv", index_col=0)
assert ct["total"].sum() == 225_208
assert abs(cov.loc["EVE","overall"] - 26.5) < 0.1
print("guards passed")

fig = plt.figure(figsize=(6.85,4.3))
gs  = GridSpec(2,2, height_ratios=[1.02,1.0], width_ratios=[1.0,1.22], figure=fig)
ax_a = fig.add_subplot(gs[0,:]); ax_b = fig.add_subplot(gs[1,0]); ax_c = fig.add_subplot(gs[1,1])
fig.subplots_adjust(left=0.085,right=0.985,top=0.97,bottom=0.115,hspace=0.38,wspace=0.30)

# ---- (a) workflow: box widths computed so the fifth lands on 0.98 exactly
ax_a.set_xlim(0,1); ax_a.set_ylim(0,1); ax_a.axis("off")
g = 0.018; w = (0.98 - 0.02 - 4*g)/5
stages = [
 ("Data sources","ClinVar GRCh38\n2026-08-16\nAlphaMissense - REVEL v1.3\ndbNSFP v5.3.1a - EVE"),
 ("Harmonize","exact chr:pos:ref:alt\nMANE-informed gene\nMANE Select transcript\nfrozen score orientation"),
 ("Design grid","2 evaluation units\n3 compositions\n3 evidence levels\n18 designs per layer"),
 ("Reliability audit","winner retention\nKendall's $W$\nrank distribution\ngene-clustered bootstrap"),
 ("Selection-aware","K-fold gene cross-fit\n$\\Delta\\Delta$AUROC with Holm\nnumeric vs supported"),
]
for i,(title,body) in enumerate(stages):
    x = 0.02 + i*(w+g)
    ax_a.add_patch(FancyBboxPatch((x,0.52), w, 0.45, boxstyle="round,pad=0.010,rounding_size=0.012",
                   facecolor="white", edgecolor="#666666", lw=0.5, zorder=2))
    ax_a.text(x+w/2, 0.925, title, ha="center", va="top", fontsize=7.0, weight="bold", zorder=3)
    ax_a.text(x+w/2, 0.855, body, ha="center", va="top", fontsize=5.6, color="#333333",
              linespacing=1.45, zorder=3)
    if i < 4:
        ax_a.add_patch(FancyArrowPatch((x+w,0.745),(x+w+g,0.745), arrowstyle="-|>",
                       mutation_scale=7, color="#555555", lw=0.7, zorder=3))
bus_l = 0.02 + 2*(w+g) + w/2; bus_r = 0.02 + 4*(w+g) + w/2
ax_a.plot([bus_l,bus_r],[0.40,0.40], color="#888888", lw=0.7, zorder=1)
ax_a.plot([bus_l,bus_l],[0.40,0.52], color="#888888", lw=0.7, zorder=1)
for j,(title,body,xc) in enumerate([
      ("Time-robust evaluation","First-in-ClinVar dates; 2021 and 2023 landmarks", 0.30),
      ("Independent validation","ProteinGym v1.3 DMS to MANE GRCh38 SNVs; MaveDB check", 0.72)]):
    bw = 0.34
    ax_a.add_patch(FancyBboxPatch((xc-bw/2,0.04), bw, 0.26, boxstyle="round,pad=0.010,rounding_size=0.012",
                   facecolor="#FAFAFA", edgecolor="#666666", lw=0.5, ls=(0,(3,2)), zorder=2))
    ax_a.text(xc, 0.255, title, ha="center", va="top", fontsize=6.6, weight="bold", zorder=3)
    ax_a.text(xc, 0.175, body, ha="center", va="top", fontsize=5.6, color="#333333", zorder=3)
    ax_a.plot([xc,xc],[0.30,0.40], color="#888888", lw=0.7, zorder=1)

# ---- (b) coverage
y = np.arange(len(cov))
ax_b.barh(y, cov["overall"], height=0.62, color=[COL[p] for p in cov.index], zorder=3)
for i,p in enumerate(cov.index):
    ax_b.annotate(f"{cov.loc[p,'overall']:.1f}", xy=(cov.loc[p,"overall"]+1.5, i),
                  va="center", fontsize=6.4)
ax_b.axvline(100, color="#BBBBBB", lw=0.6, ls=(0,(1,2)), zorder=1)
ax_b.set_yticks(y); ax_b.set_yticklabels(cov.index, fontsize=6.6); ax_b.invert_yaxis()
ax_b.set_xlim(0,112); ax_b.set_xlabel("Coverage of 225,208 variants (%)")
ax_b.annotate("defines the six-predictor layer\n53,552 variants, 1,981 genes",
              xy=(36,len(cov)-1), va="center", ha="left",
              fontsize=5.6, color="#555555")
for s in ("top","right"): ax_b.spines[s].set_visible(False)

# ---- (c) review status, every segment labelled
rows = ["P/LP","B/LB"]
for i,cl in enumerate(rows):
    left = 0
    tot = ct.loc[cl,"total"]
    for k,col in enumerate(["0","1","2",">=3"]):
        v = ct.loc[cl,col]; pct = 100*v/tot
        ax_c.barh(i, v/1000, height=0.5, left=left/1000, color=STARS[k],
                  edgecolor="white", lw=0.4, zorder=3)
        if pct >= 20:
            ax_c.annotate(f"{pct:.1f}%", xy=((left+v/2)/1000, i), ha="center", va="center",
                          fontsize=6.2, color="white" if k >= 2 else "#333333", zorder=4)
        else:
            ax_c.annotate(f"{pct:.1f}%", xy=((left+v/2)/1000, i-0.38), ha="center", va="bottom",
                          fontsize=5.6, color="#555555", zorder=4)
            ax_c.plot([(left+v/2)/1000]*2, [i-0.37, i-0.26], color="#999999", lw=0.5, zorder=4)
        left += v
    ax_c.annotate(f"$n$ = {tot:,}", xy=(tot/1000+3, i), va="center", fontsize=6.2)
ax_c.set_yticks(range(2)); ax_c.set_yticklabels(rows); ax_c.invert_yaxis()
ax_c.set_ylim(1.75,-0.75); ax_c.set_xlim(0,178)
ax_c.set_xlabel("Variants (thousands)")
from matplotlib.patches import Patch
handles = [Patch(facecolor=STARS[k], edgecolor="white", lw=0.4, label=lab)
           for k,lab in enumerate(["0 stars","1 star","2 stars","$\\geq$3 stars"])]
ax_c.legend(handles=handles,
            loc="lower center", bbox_to_anchor=(0.5,1.01), ncol=4, frameon=False,
            borderaxespad=0.0, handlelength=1.2, handletextpad=0.4, columnspacing=1.2,
            fontsize=6.2)
for s in ("top","right"): ax_c.spines[s].set_visible(False)

for ax,l in [(ax_a,"a"),(ax_b,"b"),(ax_c,"c")]:
    ax.annotate(l, xy=(-0.055 if ax is ax_a else -0.16, 1.04), xycoords="axes fraction",
                fontsize=9, weight="bold")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
print("written ->", OUT, "and", OUT.with_suffix(".pdf"))
