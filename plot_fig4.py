import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SRC = pathlib.Path("figures/nature/source_data")
OUT = pathlib.Path("Fig/fig4.png")
P   = ["AlphaMissense","REVEL","SIFT","CADD","PolyPhen-2"]   # DMS order
AM, RV = "#0072B2", "#D55E00"
COL = {"AlphaMissense":AM,"REVEL":RV,"CADD":"#4D4D4D","SIFT":"#8A8A8A","PolyPhen-2":"#B8B8B8"}
INK, MID, LIGHT = "#1A1A1A", "#666666", "#AAAAAA"

plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":7,
    "axes.labelsize":7.5,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.fontsize":6.5,"axes.linewidth":0.6})

tmp  = pd.read_csv(SRC/"Fig4b_temporal.csv").set_index("predictor")
assay= pd.read_csv(SRC/"Fig4c_per_assay.csv")
med  = pd.read_csv(SRC/"Fig4d_median_ci.csv").set_index("predictor")
lopo = pd.read_csv(SRC/"Fig4e_lopo.csv")
cat  = pd.read_csv(SRC/"Fig4f_by_category.csv").set_index("category")

for p,w in [("AlphaMissense",0.010),("REVEL",0.011),("CADD",0.021),
            ("SIFT",0.023),("PolyPhen-2",0.030)]:
    assert abs(tmp.loc[p,"d2023"]-w)<0.0015, f"4b {p} drifted"
assert abs(med.loc["AlphaMissense","median"]-0.519)<0.001
assert abs(cat.loc["Activity","diff"]-0.148)<0.0005, "4f must be protein-weighted"
assert abs(cat.loc["Stability","diff"]-0.014)<0.0005
ALLP = 0.0496
assert abs(lopo.am_minus_revel.min()-0.0342)<0.0005
print("guards passed")

dates = pd.to_datetime(
    pd.read_parquet("data/benchmark_with_temporal.parquet")["first_in_clinvar"])

fig = plt.figure(figsize=(6.85,4.9))
gs  = GridSpec(2,6, height_ratios=[1.0,1.05], figure=fig)
ax = [fig.add_subplot(gs[r,c:c+2]) for r in (0,1) for c in (0,2,4)]
a,b,c,d,e,f = ax
fig.subplots_adjust(left=0.085,right=0.975,top=0.93,bottom=0.115,
                    hspace=0.72,wspace=1.30)

# (a) First-in-ClinVar
yrs = dates.dt.year.dropna().astype(int)
a.hist(yrs, bins=range(yrs.min(), yrs.max()+2), color="#C4C4C4",
       edgecolor=MID, lw=0.4, align="left", zorder=2)
for yr,lab in [(2021.82,"EVE"),(2023.73,"AM")]:
    a.axvline(yr, color=INK, lw=0.9, ls=(0,(3,2)), zorder=3)
    a.annotate(lab, xy=(yr,1.02), xycoords=("data","axes fraction"),
               ha="center", va="bottom", fontsize=6.0, color="#333333")
a.set_xlabel("Year first in ClinVar"); a.set_ylabel("Variants (thousands)")
a.set_yticklabels([f"{t/1000:.0f}" for t in a.get_yticks()])
for s in ("top","right"): a.spines[s].set_visible(False)

# (b) temporal delta, two landmarks
y = np.arange(5); PB = ["AlphaMissense","REVEL","CADD","SIFT","PolyPhen-2"]
for i,p in enumerate(PB):
    r = tmp.loc[p]
    b.errorbar(r.d2021, i, xerr=[[r.d2021-r.lo2021],[r.hi2021-r.d2021]], fmt="o",
               ms=4, mfc="white", mec=INK, mew=1.0, ecolor=MID, elinewidth=0.9,
               capsize=1.8, capthick=0.8, zorder=3)
    b.errorbar(r.d2023, i, xerr=[[r.d2023-r.lo2023],[r.hi2023-r.d2023]], fmt="o",
               ms=4, color=INK, ecolor=MID, elinewidth=0.9,
               capsize=1.8, capthick=0.8, zorder=3)
b.set_yticks(y); b.set_yticklabels([]); b.invert_yaxis(); b.set_ylim(4.70,-1.05)
for i,p in enumerate(PB):
    b.annotate(p, xy=(0.03,i-0.42), xycoords=("axes fraction","data"),
               ha="left", va="bottom", fontsize=6.0, color="#333333")
b.set_xlim(0.002,0.036)
b.set_xticks([0,0.01,0.02,0.03]); b.set_xlabel("$\\Delta$AUROC (subset $-$ layer)")
b.plot([],[],"o",ms=4,mfc="white",mec=INK,mew=1.0,ls="none",label="after 2021")
b.plot([],[],"o",ms=4,color=INK,ls="none",label="after 2023")
b.legend(loc="lower center", bbox_to_anchor=(0.5,1.01), ncol=2,
         frameon=False, borderaxespad=0.0, handletextpad=0.3)
for s in ("top","right"): b.spines[s].set_visible(False)

# (c) per-assay Spearman
data = [assay.loc[assay.predictor==p,"spearman_rho"].dropna() for p in P]
bp = c.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
               medianprops=dict(color=INK,lw=1.0),
               whiskerprops=dict(color=MID,lw=0.7), capprops=dict(color=MID,lw=0.7))
for patch,p in zip(bp["boxes"],P):
    patch.set(facecolor=COL[p], alpha=0.55, edgecolor=MID, lw=0.7)
for i,p in enumerate(P):
    v = data[i]; c.scatter(np.random.default_rng(i).normal(i+1,0.07,len(v)), v,
                           s=5, color=COL[p], alpha=0.5, lw=0, zorder=4)
c.set_xticks(range(1,6)); c.set_xticklabels(P, rotation=90, fontsize=6.2)
c.set_ylim(-0.05,0.95); c.set_ylabel("Per-assay Spearman $\\rho$")
c.annotate("36 assays, 32 proteins", xy=(0.03,0.10), xycoords="axes fraction",
           fontsize=6.0, color="#555555")
for s in ("top","right"): c.spines[s].set_visible(False)

# (d) median with CI
for i,p in enumerate(P):
    r = med.loc[p]
    d.errorbar(r["median"], i, xerr=[[r["median"]-r.lo],[r.hi-r["median"]]],
               fmt="o", ms=4.5, color=COL[p], ecolor=MID, elinewidth=0.9,
               capsize=2, capthick=0.8, zorder=3)
d.set_yticks(range(5)); d.set_yticklabels([]); d.invert_yaxis(); d.set_ylim(4.65,-0.95)
for i,p in enumerate(P):
    d.annotate(f"{p}  {med.loc[p,'median']:.3f}", xy=(0.015,i-0.33),
               xycoords=("axes fraction","data"), ha="left", va="center",
               fontsize=6.0, color="#333333")
d.set_xlim(0.26,0.65); d.set_xlabel("Median per-assay $\\rho$ (95% CI)")
d.annotate("AM $-$ REVEL  +0.079 (0.006\u20130.142)\n$P$(best) = 0.982",
           xy=(0.97,0.03), xycoords="axes fraction", ha="right", va="bottom",
           fontsize=5.8, color="#555555")
d.set_ylim(5.15,-0.95)
for s in ("top","right"): d.spines[s].set_visible(False)

# (e) leave-one-protein-out
v = np.sort(lopo.am_minus_revel.values)
e.scatter(range(1,len(v)+1), v, s=11, color=AM, zorder=3)
e.axhline(ALLP, color=INK, lw=0.9, ls=(0,(3,2)), zorder=2)
e.annotate(f"all 32 proteins, {ALLP:.3f}", xy=(3,ALLP), xytext=(0,-11),
           textcoords="offset points", ha="left", fontsize=6.0, color="#333333")
e.set_xlim(0,33); e.set_ylim(0,0.075)
e.set_xlabel("Protein removed (sorted)")
e.set_ylabel("$\\Delta\\rho$ (AlphaMissense $-$ REVEL)", fontsize=7.0)
e.annotate("AlphaMissense first after\nevery single removal", xy=(0.06,0.97), va="top",
           xycoords="axes fraction", fontsize=6.0, color="#555555")
for s in ("top","right"): e.spines[s].set_visible(False)

# (f) by assay category
cf = cat.sort_values("diff")
yy = np.arange(len(cf))
f.barh(yy, cf["diff"], height=0.58, color=AM, zorder=3)
for i,(k,r) in enumerate(cf.iterrows()):
    f.annotate(f"{r['diff']:.3f}", xy=(r["diff"]+0.004, i), va="center", fontsize=6.2)
    inside = r["diff"] > 0.03
    f.annotate(f"{int(r.assays)}/{int(r.proteins)}",
               xy=(0.004 if inside else r["diff"]+0.042, i), va="center",
               fontsize=5.8, color="white" if inside else "#555555")
f.set_yticks(yy)
f.set_yticklabels([k.replace("OrganismalFitness","Organismal\nfitness") for k in cf.index],
                  fontsize=6.0)
f.set_xlim(0,0.205); f.set_xlabel("$\\Delta\\rho$, protein-weighted")
f.set_title("assays/proteins inside bars", fontsize=5.8,
            color="#555555", loc="left", pad=3)
for s in ("top","right"): f.spines[s].set_visible(False)

for axx,l in zip(ax,"abcdef"):
    axx.annotate(l, xy=(-0.14,1.06), xycoords="axes fraction", fontsize=9, weight="bold")

OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=600,bbox_inches="tight",pad_inches=0.02)
fig.savefig(OUT.with_suffix(".pdf"),bbox_inches="tight",pad_inches=0.02)
print("written ->",OUT,"and",OUT.with_suffix(".pdf"))
