"""STEP 13E — temporal exposure-resistant performance audit (paired gene-clustered)."""
from __future__ import annotations
from math import erfc, sqrt
import numpy as np
import pandas as pd
from variantbenchpy.metrics.fast import fast_auroc
from variantbenchpy.evaluation.weighting import gene_weighted_metric

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
PRED6 = PREDS5 + ["EVE"]
CUT = "2023-09-22"
WITHIN_MIN = 10
PAIRS = [("REVEL", "AlphaMissense"), ("AlphaMissense", "CADD")]


def _holm(p):
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def _pooled(y, s):
    m = ~np.isnan(s)
    return fast_auroc(y[m], s[m]) if np.unique(y[m]).size == 2 else np.nan


def _macro(y, s, g):
    return gene_weighted_metric(y, s, g, "auroc", min_n=WITHIN_MIN)[0]


def _paired_boot(sub, preds, post_mask, gcol, unit, n_boot, seed):
    """PAIRED gene-clustered bootstrap: nested samples (post subset of full).
    One gene resample per iteration -> compute BOTH full and post on it."""
    y = sub["label"].to_numpy(); g = sub[gcol].to_numpy()
    pm = post_mask.to_numpy()
    mat = {p: sub[p].to_numpy() for p in preds}
    f = lambda yy, ss, gg: _pooled(yy, ss) if unit == "pooled" else _macro(yy, ss, gg)
    pts_full = {p: f(y, mat[p], g) for p in preds}
    pts_post = {p: f(y[pm], mat[p][pm], g[pm]) for p in preds}
    uniq = np.unique(g); idx = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    dfull = {p: [] for p in preds}; dpost = {p: [] for p in preds}
    for _ in range(n_boot):
        pick = np.concatenate([idx[u] for u in rng.choice(uniq, uniq.size, replace=True)])
        yy, gg2, pp = y[pick], g[pick], pm[pick]
        if np.unique(yy).size < 2 or np.unique(yy[pp]).size < 2:
            continue
        for p in preds:
            ss = mat[p][pick]
            dfull[p].append(f(yy, ss, gg2))
            dpost[p].append(f(yy[pp], ss[pp], gg2[pp]))
    return (pts_full, pts_post,
            {p: np.array(v, "float64") for p, v in dfull.items()},
            {p: np.array(v, "float64") for p, v in dpost.items()})


def _layer(df, preds, gcol, n_boot, seed, title):
    sh = df[preds].notna().all(axis=1)
    sub = df[sh].copy()
    post_mask = sub["first_in_clinvar"] > pd.Timestamp(CUT)
    L = ["", "=" * 50, title, "=" * 50,
         f"Full:       {len(sub):,} variants / {sub[gcol].nunique():,} genes",
         f"Post-2023:  {int(post_mask.sum()):,} variants / "
         f"{sub.loc[post_mask, gcol].nunique():,} genes  (> {CUT})",
         "ΔAUROC CI: paired gene-clustered bootstrap (nested samples)"]
    store = {}
    for unit, label in [("pooled", "POOLED AUROC"), ("macro", "MACRO_GENE_N10 AUROC")]:
        pf, pp_, df_, dp_ = _paired_boot(sub, preds, post_mask, gcol, unit, n_boot, seed)
        store[unit] = (pf, pp_, df_, dp_)
        L += ["", label, "  " + f"{'Predictor':<15}{'Full':>8}{'Post2023':>10}{'ΔAUROC [95% CI]':>26}"]
        for p in preds:
            d = pp_[p] - pf[p]
            B = min(len(df_[p]), len(dp_[p]))
            diff = dp_[p][:B] - df_[p][:B]
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            L.append("  " + f"{p:<15}{pf[p]:>8.3f}{pp_[p]:>10.3f}   {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    rk = lambda d: " > ".join(sorted(d, key=d.get, reverse=True))
    L += ["", "POOLED RANKING",
          f"  Full:       {rk(store['pooled'][0])}",
          f"  Post-2023:  {rk(store['pooled'][1])}",
          "", "MACRO-GENE RANKING",
          f"  Full:       {rk(store['macro'][0])}",
          f"  Post-2023:  {rk(store['macro'][1])}"]

    pf, pp_, df_, dp_ = store["pooled"]
    pairs = [(a, b) for a, b in PAIRS if a in preds and b in preds]
    L += ["", "RELATIVE-PERFORMANCE CHANGE (pooled ΔΔAUROC, paired gene-clustered)",
          "  " + f"{'Pair':<28}{'ΔΔAUROC':>9}{'95% CI':>22}{'Holm p':>10}"]
    rows, pv = [], []
    for a, b in pairs:
        B = min(len(df_[a]), len(df_[b]), len(dp_[a]), len(dp_[b]))
        dd = (dp_[a][:B] - dp_[b][:B]) - (df_[a][:B] - df_[b][:B])
        pt = (pp_[a] - pp_[b]) - (pf[a] - pf[b])
        lo, hi = np.nanpercentile(dd, [2.5, 97.5])
        pv.append(erfc(abs(pt / (np.nanstd(dd) + 1e-12)) / sqrt(2)))
        rows.append((f"{a} vs {b}", pt, lo, hi))
    for (nm, pt, lo, hi), pa in zip(rows, _holm(np.array(pv))):
        ps = "<0.001" if pa < 0.001 else f"{pa:.3f}"
        L.append("  " + f"{nm:<28}{pt:>+9.3f}   [{lo:+.3f}, {hi:+.3f}] {ps:>8}")

    # composition standardization: reweight post-2023 to full's (gene x star-band) mix
    L += ["", "TEMPORAL COMPOSITION SENSITIVITY (standardized to full-cohort composition)",
          "  " + f"{'Predictor':<15}{'observed Δ':>12}{'standardized Δ':>16}"]
    sub["_band"] = np.where(sub["stars"] >= 2, "2+", np.where(sub["stars"] >= 1, "1", "0"))
    strata = sub[gcol].astype(str) + "|" + sub["_band"]
    wf = strata.value_counts(normalize=True)
    wp = strata[post_mask].value_counts(normalize=True)
    common = wp.index.intersection(wf.index)
    ratio = (wf[common] / wp[common])
    w = strata[post_mask].map(ratio).fillna(0.0).to_numpy()
    ypost = sub.loc[post_mask, "label"].to_numpy()
    for p in preds:
        s_post = sub.loc[post_mask, p].to_numpy()
        m = ~np.isnan(s_post) & (w > 0)
        std = _weighted_auroc(ypost[m], s_post[m], w[m])
        L.append("  " + f"{p:<15}{pp_[p]-pf[p]:>+12.3f}{std-pf[p]:>+16.3f}")
    L.append("  (sensitivity/standardization analysis, NOT causal decomposition)")
    return "\n".join(L)


def _weighted_auroc(y, s, w):
    """Weighted AUROC via weighted Mann-Whitney."""
    o = np.argsort(s, kind="mergesort")
    y, s, w = y[o], s[o], w[o]
    pos = y == 1
    cw_neg = np.cumsum(np.where(~pos, w, 0.0))
    tot_neg = cw_neg[-1] if len(cw_neg) else 0.0
    tot_pos = w[pos].sum()
    if tot_neg <= 0 or tot_pos <= 0:
        return np.nan
    below = cw_neg[pos] - np.where(pos, w, 0.0)[pos] * 0.0
    return float((below * w[pos]).sum() / (tot_pos * tot_neg))


def run(path="data/benchmark_with_temporal.parquet", n_boot=400, seed=1):
    df = pd.read_parquet(path)
    df["first_in_clinvar"] = pd.to_datetime(df["germline_date_created"], errors="coerce")
    g = "primary_gene_10A"
    L = ["STEP 13E/13F — TEMPORAL PERFORMANCE AUDIT", "=" * 41,
         _layer(df, PREDS5, g, n_boot, seed, "PRIMARY — 5-PREDICTOR MATCHED BENCHMARK"),
         _layer(df, PRED6, g, n_boot, seed + 7, "SECONDARY — 6-PREDICTOR (EVE-CONSTRAINED)"),
         "", "INTERPRETATION",
         f"  Temporal exposure-resistant evaluation: variants first public in ClinVar after {CUT}.",
         "  Absence from ClinVar does not exclude exposure via HGMD, gnomAD, UniProt or",
         "  literature; this is NOT a leakage-free claim."]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
