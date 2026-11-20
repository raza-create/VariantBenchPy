import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SRC = pathlib.Path("figures/nature/source_data")
OUT = pathlib.Path("Fig/fig3.png")
PREDS = ["AlphaMissense","REVEL","CADD","SIFT","PolyPhen-2"]
ALL6  = PREDS + ["EVE"]
INK, MID, LIGHT = "#1A1A1A", "#666666", "#AAAAAA"

plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42,"font.size":7,
    "axes.labelsize":7.5,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.fontsize":6.5,"axes.linewidth":0.6})

def need(n):
    p = SRC/n
    if not p.exists(): raise SystemExit(f"missing {p} -- run the bootstrap first")
    return pd.read_csv(p)

cov   = need("Fig3b_coverage_by_class.csv")
cov   = cov.rename(columns={cov.columns[0]:"predictor"}).set_index("predictor")
delta = need("Fig3c_coverage_delta.csv").set_index("predictor")
strat = need("Fig3e_by_stratum.csv").set_index("predictor")

for p,w in [("AlphaMissense",-0.015),("REVEL",-0.013),("CADD",-0.041),
            ("SIFT",-0.028),("PolyPhen-2",-0.036)]:
    assert abs(delta.loc[p,"delta"]-w) < 0.0015, f"3c {p} drifted from {w}"
assert abs(cov.loc["EVE","P/LP"]-46.4) < 0.15
assert abs(cov.loc["EVE","B/LB"]-17.2) < 0.15
assert cov.drop("EVE")["gap_pp"].abs().max() <= 6.15, "a non-EVE gap exceeds 6.1 pp"
print("guards passed")

dd = need("Fig3d_pairwise_dd.csv")
dd["label"] = dd["pair"].str.replace(" - ", " \u2212 ", regex=False)

fig = plt.figure(figsize=(6.85,5.0))
gs  = GridSpec(2,6, height_ratios=[1.0,1.30], figure=fig)
ax_a=fig.add_subplot(gs[0,0:2]); ax_b=fig.add_subplot(gs[0,2:6])
ax_c=fig.add_subplot(gs[1,0:2]); ax_d=fig.add_subplot(gs[1,2:4])
ax_e=fig.add_subplot(gs[1,4:6])
fig.subplots_adjust(left=0.085,right=0.975,top=0.92,bottom=0.135,
                    hspace=0.40,wspace=0.62)

layers=["Five\npredictors","Six\npredictors"]; plp=[33.3,56.9]
for i,(lab,v) in enumerate(zip(layers,plp)):
    ax_a.bar(i,v,width=0.46,color="#555555",zorder=3)
    ax_a.bar(i,100-v,width=0.46,bottom=v,color="#D4D4D4",zorder=3)
    ax_a.annotate(f"{v:.1f}%",xy=(i,v/2),ha="center",va="center",color="white",fontsize=7)
    ax_a.annotate(f"{100-v:.1f}%",xy=(i,v+(100-v)/2),ha="center",va="center",
                  color="#333333",fontsize=7)
ax_a.fill_between([0.23,0.77],[33.3,56.9],[0,0],color="#555555",alpha=0.16,lw=0,zorder=1)
ax_a.set_xticks(range(2)); ax_a.set_xticklabels(["",""])
for i,lab in enumerate(["Five predictors","Six predictors"]):
    ax_a.annotate(lab,xy=(i,-4),ha="center",va="top",fontsize=6.5,
                  annotation_clip=False)
ax_a.set_ylim(0,100); ax_a.set_xlim(-0.55,1.62); ax_a.set_ylabel("Variants (%)")
ax_a.annotate("187,155 variants\n13,599 genes",xy=(0,-16),ha="center",va="top",
              fontsize=5.6,color="#555555",annotation_clip=False)
ax_a.annotate("53,552 variants\n1,981 genes\n$N_{\\mathrm{eff}}$ = 282",xy=(1,-16),
              ha="center",va="top",fontsize=5.6,color="#555555",annotation_clip=False)
ax_a.annotate("P/LP",xy=(1.34,28),fontsize=6.5,color="#555555")
ax_a.annotate("B/LB",xy=(1.34,78),fontsize=6.5,color="#AAAAAA")
for s in ("top","right"): ax_a.spines[s].set_visible(False)

cb = cov.loc[ALL6].sort_values("gap_pp",ascending=False); y=np.arange(len(cb))
ax_b.axhspan(-0.5,0.5,color="#000000",alpha=0.045,lw=0,zorder=0)
for i,(p,r) in enumerate(cb.iterrows()):
    ax_b.plot([r["B/LB"],r["P/LP"]],[i,i],color=LIGHT,lw=1.3,zorder=2)
    ax_b.scatter(r["B/LB"],i,s=26,facecolor="white",edgecolor=INK,lw=1.0,zorder=3)
    ax_b.scatter(r["P/LP"],i,s=26,color=INK,zorder=3)
    ax_b.annotate(f"{r['gap_pp']:+.1f}",xy=(101.5,i),va="center",fontsize=6.5,color="#333333")
ax_b.annotate("Gap (pp)",xy=(101.5,-0.62),va="center",fontsize=6.2,color="#555555")
ax_b.set_yticks(y); ax_b.set_yticklabels([])
ax_b.invert_yaxis(); ax_b.set_ylim(len(cb)-0.30, -0.85)
for i, lab in enumerate(cb.index):
    ax_b.annotate(lab, xy=(0.015, i-0.33), xycoords=("axes fraction","data"),
                  ha="left", va="center", fontsize=6.4, color="#333333")
ax_b.set_xlim(10,112); ax_b.set_xlabel("Variants scored (%)")
ax_b.scatter([],[],s=26,color=INK,label="P/LP")
ax_b.scatter([],[],s=26,facecolor="white",edgecolor=INK,lw=1.0,label="B/LB")
ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
            frameon=False, borderaxespad=0.0, handletextpad=0.3)
for s in ("top","right"): ax_b.spines[s].set_visible(False)

d = delta.loc[PREDS]; y=np.arange(len(d))
ax_c.errorbar(d["delta"],y,xerr=[d["delta"]-d["lo"],d["hi"]-d["delta"]],fmt="o",ms=4,
              color=INK,ecolor=MID,elinewidth=0.9,capsize=2,capthick=0.9,zorder=3)
ax_c.axvline(0,color="#333333",lw=0.8,zorder=2)
ax_c.set_yticks(y); ax_c.set_yticklabels([])
for i, lab in enumerate(d.index):
    ax_c.annotate(lab, xy=(0.015, i-0.33), xycoords=("axes fraction","data"),
                  ha="left", va="bottom", fontsize=6.0, color="#333333"); ax_c.tick_params(axis="y", pad=1, length=0)
ax_c.invert_yaxis(); ax_c.set_ylim(len(d)-0.30, -0.85)
ax_c.set_xlim(-0.058,0.010); ax_c.set_xlabel("$\\Delta$AUROC (subset $-$ layer)")
for s in ("top","right"): ax_c.spines[s].set_visible(False)

y=np.arange(len(dd))
for i, r in dd.reset_index(drop=True).iterrows():
    sup = r["holm"] < 0.05
    ax_d.errorbar(r["dd"], i, xerr=[[r["dd"]-r["lo"]],[r["hi"]-r["dd"]]], fmt="o",
                  ms=5 if sup else 3.6, color=INK, ecolor=MID,
                  elinewidth=1.5 if sup else 0.9, capsize=2, capthick=0.9, zorder=3)
    ps = "$P$ < 0.001" if r["holm"] < 0.001 else f"$P$ = {r['holm']:.3f}"
    ax_d.annotate(ps, xy=(0.985, i-0.33), xycoords=("axes fraction","data"),
                  ha="right", va="bottom", fontsize=6.2, color="#333333")
ax_d.axvline(0, color="#333333", lw=0.8, zorder=2)
ax_d.set_yticks(y); ax_d.set_yticklabels([])
for i, lab in enumerate(dd["label"]):
    ax_d.annotate(lab, xy=(0.015, i-0.33), xycoords=("axes fraction","data"),
                  ha="left", va="bottom", fontsize=6.0, color="#333333", zorder=5,
                  bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
ax_d.tick_params(axis="y", pad=1, length=0)
ax_d.invert_yaxis(); ax_d.set_ylim(len(dd)-0.30, -0.85); ax_d.set_xlim(-0.032, 0.058)
ax_d.set_xlabel("$\\Delta\\Delta$AUROC (change in gap)")
for s in ("top","right"): ax_d.spines[s].set_visible(False)

s = strat.loc[PREDS]; y=np.arange(len(s))
for i,(p,r) in enumerate(s.iterrows()):
    ax_e.plot([r["one_star"],r["two_star"]],[i,i],color=LIGHT,lw=1.3,zorder=2)
    ax_e.errorbar(r["one_star"],i,xerr=[[r["one_star"]-r["one_lo"]],[r["one_hi"]-r["one_star"]]],
                  fmt="o",ms=4,mfc="white",mec=INK,mew=1.0,ecolor=MID,elinewidth=0.9,
                  capsize=1.8,capthick=0.8,zorder=3)
    ax_e.errorbar(r["two_star"],i,xerr=[[r["two_star"]-r["two_lo"]],[r["two_hi"]-r["two_star"]]],
                  fmt="o",ms=4,color=INK,ecolor=MID,elinewidth=0.9,
                  capsize=1.8,capthick=0.8,zorder=3)
ax_e.axvline(0,color="#333333",lw=0.8,zorder=2)
ax_e.set_yticks(y); ax_e.set_yticklabels(s.index); ax_e.tick_params(axis="y", pad=1, length=0)
ax_e.invert_yaxis(); ax_e.set_ylim(len(s)-0.30, -0.85)
ax_e.set_xlim(-0.082,0.016); ax_e.set_xlabel("$\\Delta$AUROC")

r0 = s.iloc[0]
ax_e.scatter([], [], s=20, facecolor="white", edgecolor=INK, lw=1.0, label="1 star")
ax_e.scatter([], [], s=20, color=INK, label="\u22652 stars")
ax_e.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
            borderaxespad=0.0, handletextpad=0.3, columnspacing=1.2)
for sp in ("top","right"): ax_e.spines[sp].set_visible(False)

for ax,l in [(ax_a,"a"),(ax_b,"b"),(ax_c,"c"),(ax_d,"d"),(ax_e,"e")]:
    ax.annotate(l,xy=(-0.10 if ax is ax_b else -0.16,1.06),xycoords="axes fraction",
                fontsize=9,weight="bold")

OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=600,bbox_inches="tight",pad_inches=0.02)
fig.savefig(OUT.with_suffix(".pdf"),bbox_inches="tight",pad_inches=0.02)
print("written ->",OUT,"and",OUT.with_suffix(".pdf"))
n = sum(1 for p in PREDS if strat.loc[p,"two_lo"] <= 0 <= strat.loc[p,"two_hi"])
print(f"note: {n} of 5 predictors have a >=2-star interval covering zero")
