"""Figure 5 — mechanism: pooled vs gene-level evaluation reward different comparisons.

Pooled AUROC over all P/LP x B/LB pairs decomposes exactly into within-gene pairs
and between-gene pairs (Mann-Whitney U is additive over pairs). Variant weighting
of per-gene AUROC is also reported (Extended Data) because it does NOT explain the
pooled REVEL lead.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from variantbenchpy.realdata.fignat_common import (
    W2, MM, PRED5, COL, GREY, LGREY, DARK, SRC, GENE, GENE_LEVEL, letters, per_gene_auroc)

POOLED_S01 = {"AlphaMissense": 0.960003, "REVEL": 0.967292, "CADD": 0.924963,
              "SIFT": 0.907868, "PolyPhen-2": 0.902037}   # frozen Step 12, 5-pred pooled


def pair_decomposition(S, score):
    d = S[[GENE, "label", score]].dropna().copy()
    y = d["label"].to_numpy()
    npos = int(y.sum()); nneg = len(y) - npos
    r_all = d[score].rank(method="average").to_numpy()
    u_tot = r_all[y == 1].sum() - npos * (npos + 1) / 2.0
    d["r"] = d.groupby(GENE)[score].rank(method="average")
    d["rpos"] = np.where(d["label"] == 1, d["r"], 0.0)
    a = d.groupby(GENE).agg(np_=("label", "sum"), n=("label", "size"), srp=("rpos", "sum"))
    a["nn"] = a["n"] - a["np_"]
    u_w = float((a["srp"] - a["np_"] * (a["np_"] + 1) / 2.0).sum())
    n_t = float(npos) * float(nneg)
    n_w = float((a["np_"] * a["nn"]).sum())
    n_b = n_t - n_w
    return {"pooled": u_tot / n_t, "within": u_w / n_w, "between": (u_tot - u_w) / n_b,
            "share_between": n_b / n_t, "pairs_within": n_w, "pairs_between": n_b}


def fig5(B, Q):
    S = B[B[PRED5].notna().all(axis=1)]
    aucs, sizes = {}, None
    for p in PRED5:
        a, n = per_gene_auroc(S, p)
        aucs[p] = a
        sizes = n if sizes is None else sizes
    G = pd.DataFrame(aucs).dropna()
    n = sizes.reindex(G.index).astype(int)
    Q.check("F5 eligible genes 2,517", len(G) == GENE_LEVEL["n"], f"{len(G):,}")
    for p in PRED5:
        Q.close(f"F5 per-gene mean {p}", float(G[p].mean()), GENE_LEVEL["mean"][p], 6e-4)
        Q.close(f"F5 per-gene median {p}", float(G[p].median()), GENE_LEVEL["median"][p], 6e-4)
    R = G.round(10)
    is_max = R.eq(R.max(axis=1), axis=0)
    ntop = is_max.sum(axis=1)
    ties = int((ntop > 1).sum())
    unique = is_max[ntop == 1].sum()
    frac = is_max.div(ntop, axis=0).sum()
    fail = (G < 0.6).sum()
    Q.check("F5 tied genes 967", ties == GENE_LEVEL["ties"], str(ties))
    for p in PRED5:
        Q.check(f"F5 unique wins {p}", int(unique[p]) == GENE_LEVEL["unique"][p], str(int(unique[p])))
        Q.close(f"F5 fractional wins {p}", float(frac[p]), GENE_LEVEL["frac"][p], 0.06)
        Q.check(f"F5 genes AUROC<0.60 {p}", int(fail[p]) == GENE_LEVEL["fail"][p], str(int(fail[p])))

    dec = {p: pair_decomposition(S, p) for p in PRED5}
    for p in PRED5:
        Q.close(f"F5 decomposition reproduces pooled {p}", dec[p]["pooled"], POOLED_S01[p], 1e-5)
    Q.report_new("F5 pair decomposition (within-gene / between-gene / pooled AUROC)",
                 {p: f"{dec[p]['within']:.4f} / {dec[p]['between']:.4f} / {dec[p]['pooled']:.4f}"
                  for p in PRED5})
    sb = dec["AlphaMissense"]["share_between"]
    Q.report_new("F5 share of P/LP x B/LB pairs that are between-gene",
                 f"{sb:.4f} ({dec['AlphaMissense']['pairs_between']:.3e} of "
                 f"{dec['AlphaMissense']['pairs_between'] + dec['AlphaMissense']['pairs_within']:.3e})")
    gaps = {k: dec["AlphaMissense"][k] - dec["REVEL"][k] for k in ("within", "between", "pooled")}
    Q.report_new("F5 AlphaMissense - REVEL by pair type", {k: f"{v:+.4f}" for k, v in gaps.items()})

    delta = G["AlphaMissense"] - G["REVEL"]
    rho, prho = spearmanr(n, delta)
    rng = np.random.default_rng(0)
    nv, dv = n.to_numpy(), delta.to_numpy()
    bs = [spearmanr(nv[ix], dv[ix])[0]
          for ix in (rng.integers(0, len(dv), len(dv)) for _ in range(1000))]
    rlo, rhi = np.percentile(bs, [2.5, 97.5])
    q = pd.qcut(n.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    qs = pd.DataFrame({"n": n, "d": delta, "q": q}).groupby("q", observed=True).agg(
        genes=("d", "size"), n_min=("n", "min"), n_max=("n", "max"),
        mean_d=("d", "mean"), median_d=("d", "median"))
    unw, wtd = float(delta.mean()), float(np.average(delta, weights=n))
    Q.report_new("F5/ED gene size vs AM-REVEL (Spearman)",
                 f"{rho:.3f} (bootstrap 95% CI {rlo:.3f} to {rhi:.3f}), P = {prho:.2e}")
    Q.report_new("F5/ED mean AM-REVEL per gene: unweighted vs variant-weighted",
                 f"{unw:+.4f} vs {wtd:+.4f}")

    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(2, 6, left=0.085, right=0.985, top=0.95, bottom=0.115,
                          hspace=0.7, wspace=1.5)
    x = np.arange(5)

    a = fig.add_subplot(gs[0, 0:3])
    bp = a.boxplot([G[p].to_numpy() for p in PRED5], positions=x, widths=0.5,
                   showfliers=False, patch_artist=True, medianprops=dict(color=DARK, lw=0.9),
                   whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5), boxprops=dict(lw=0.5))
    for patch, p in zip(bp["boxes"], PRED5):
        patch.set_facecolor(COL[p]); patch.set_alpha(0.4)
    for i, p in enumerate(PRED5):
        a.plot(i, G[p].mean(), "D", ms=3, color=COL[p], mec=DARK, mew=0.4, zorder=3)
    a.set_xticks(x, PRED5, rotation=20, ha="right")
    a.set_ylabel(f"Per-gene AUROC ({len(G):,} genes)"); a.set_ylim(0.3, 1.03)

    b = fig.add_subplot(gs[0, 3:6])
    b.bar(x, [unique[p] for p in PRED5], 0.6, color=[COL[p] for p in PRED5])
    b.bar(x, [frac[p] - unique[p] for p in PRED5], 0.6, bottom=[unique[p] for p in PRED5],
          color=[COL[p] for p in PRED5], alpha=0.35)
    for i, p in enumerate(PRED5):
        b.text(i, frac[p] + 15, f"{frac[p]:.0f}", ha="center", fontsize=6)
    b.bar([], [], color=GREY, label="Sole best")
    b.bar([], [], color=GREY, alpha=0.35, label="Share of ties")
    b.set_xticks(x, PRED5, rotation=20, ha="right"); b.set_ylabel("Genes won")
    b.set_ylim(0, max(frac) * 1.18); b.legend(loc="upper right")

    c = fig.add_subplot(gs[1, 0:2])
    c.bar(x, [fail[p] for p in PRED5], 0.6, color=[COL[p] for p in PRED5])
    for i, p in enumerate(PRED5):
        c.text(i, fail[p] + 2, str(int(fail[p])), ha="center", fontsize=6)
    c.set_xticks(x, PRED5, rotation=35, ha="right")
    c.set_ylabel("Genes with AUROC < 0.60"); c.set_ylim(0, max(fail) * 1.2)

    d = fig.add_subplot(gs[1, 2:4])
    yy = np.arange(5)[::-1]
    for i, p in enumerate(PRED5):
        w_, b_ = dec[p]["within"], dec[p]["between"]
        d.plot([w_, b_], [yy[i], yy[i]], color=LGREY, lw=1.6, zorder=1)
        d.plot(w_, yy[i], "o", mfc="white", mec=COL[p], mew=1, ms=4, zorder=3)
        d.plot(b_, yy[i], "o", color=COL[p], ms=4, zorder=3)
    d.set_yticks(yy, PRED5); d.spines["left"].set_visible(False); d.tick_params(axis="y", length=0)
    d.set_xlabel("AUROC over variant pairs")
    d.plot([], [], "o", mfc="white", mec=DARK, ms=3.5, label="Within-gene pairs")
    d.plot([], [], "o", color=DARK, ms=3.5, label="Between-gene pairs")
    d.legend(loc="lower left", handletextpad=0.3)

    e = fig.add_subplot(gs[1, 4:6])
    keys = ["within", "between", "pooled"]
    vals = [gaps[k] for k in keys]
    e.bar(range(3), vals, 0.6,
          color=[COL["AlphaMissense"] if v > 0 else COL["REVEL"] for v in vals])
    lim = max(abs(v) for v in vals) * 1.7
    for i, v in enumerate(vals):
        e.text(i, v + (lim * 0.04 if v >= 0 else -lim * 0.04), f"{v:+.4f}", ha="center",
               va="bottom" if v >= 0 else "top", fontsize=6)
    e.axhline(0, color=DARK, lw=0.6)
    e.set_xticks(range(3), ["Within\ngene", "Between\ngenes", "Pooled"])
    e.set_ylabel("AlphaMissense \u2212 REVEL (AUROC)"); e.set_ylim(-lim, lim)
    e.text(0.98, 0.03, f"{100 * sb:.1f}% of pairs\nare between genes", transform=e.transAxes,
           ha="right", va="bottom", fontsize=5.6, color=GREY)

    letters(fig, [(a, "a"), (b, "b"), (c, "c"), (d, "d"), (e, "e")])
    out = G.copy(); out["n_variants"] = n; out["am_minus_revel"] = delta
    out.to_csv(os.path.join(SRC, "Fig5_genes.csv"))
    pd.DataFrame(dec).T.to_csv(os.path.join(SRC, "Fig5de_pair_decomposition.csv"))
    qs.to_csv(os.path.join(SRC, "ExtData_gene_size_quartiles.csv"))
    return fig
