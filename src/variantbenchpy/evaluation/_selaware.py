"""Selection-aware reversal inference (K-fold gene-level cross-fitting) for Step 9."""
from __future__ import annotations

import numpy as np
import pandas as pd

from variantbenchpy.evaluation.weighting import config_scores
from variantbenchpy.evaluation.reliability import _axes
from variantbenchpy.harmonize.schema import COL_GENE, COL_VARIANT_ID
from variantbenchpy.metrics.fast import fast_auroc

VALIDATED_POWER = {40: 0.185, 80: 0.360, 160: 0.755}  # verified crossfit rerun,
# 200 reps/cell, effect=0.6 (realized ddAUROC ~= 0.05); see data/step9_verification_grid.csv
GENE_WARNING_THRESHOLD = 80


def _ddelta_point(sm, base_mask, pert_mask, a, b):
    labels = sm.labels
    idx = sm.scores.index
    lab = labels.notna()
    y = labels[lab].to_numpy()
    sa = sm.scores[a][lab].to_numpy(); sb = sm.scores[b][lab].to_numpy()
    bm = base_mask.reindex(idx).fillna(False).to_numpy(bool)[lab.to_numpy()]
    pm = pert_mask.reindex(idx).fillna(False).to_numpy(bool)[lab.to_numpy()]
    def D(m):
        yy = y[m]
        if m.sum() == 0 or np.unique(yy).size < 2:
            return np.nan
        return fast_auroc(yy, sa[m]) - fast_auroc(yy, sb[m])
    return D(pm) - D(bm)


def _select_flip(sm, dim, base_masks, options, fold_mask, preds):
    base_top = config_scores(sm, base_masks["_all"] & fold_mask, "auroc", "variant").reindex(preds).idxmax()
    for lvl, m in options.items():
        if lvl == base_masks[dim][0]:
            continue
        pert = m.copy()
        for other, val in base_masks.items():
            if other == "_all" or other == dim:
                continue
            pert = pert & val[1]
        top = config_scores(sm, pert & fold_mask, "auroc", "variant").reindex(preds).idxmax()
        if top != base_top:
            return base_top, top, pert
    return None


def _holm(pvals, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def selection_aware_reversals(sm, gene_groups=None, evidence_masks=None, metric="auroc",
                              k_folds=5, alpha=0.05, seed=0):
    from scipy.stats import t as tdist
    idx = sm.scores.index
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(idx)
    uniq = pd.unique(genes.dropna())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    folds = [set(uniq[perm[j::k_folds]]) for j in range(k_folds)]
    fold_mask = [pd.Series(genes.isin(f).to_numpy(), index=idx) for f in folds]
    all_true = pd.Series(True, index=idx)

    cov, unit, comp, evid = _axes(sm, gene_groups, evidence_masks)
    preds = sm.predictor_names
    base_masks = {"coverage": ("available", cov["available"]),
                  "composition": ("all-genes", comp["all-genes"]),
                  "evidence": ("all", evid["all"])}
    base_masks["_all"] = (base_masks["coverage"][1] & base_masks["composition"][1]
                          & base_masks["evidence"][1])
    base_mask = base_masks["_all"]
    dim_options = {"coverage": cov, "composition": comp, "evidence": evid}

    recs = []
    for dim, options in dim_options.items():
        full = _select_flip(sm, dim, base_masks, options, all_true, preds)
        rec = {"dimension": dim, "numerical": "YES" if full else "NO", "new_top": "",
               "ddauroc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
               "raw_p": np.nan, "holm_p": np.nan, "supported": "—"}
        if full:
            base_top, new_top, _ = full
            rec["new_top"] = new_top
            dks = []
            for k in range(k_folds):
                sel = _select_flip(sm, dim, base_masks, options, ~fold_mask[k], preds)
                if not sel:
                    continue
                bt, nt, pert = sel
                dk = _ddelta_point(sm, base_mask & fold_mask[k], pert & fold_mask[k], nt, bt)
                if not np.isnan(dk):
                    dks.append(dk)
            if len(dks) >= 3:
                arr = np.array(dks, float); mean = arr.mean(); sd = arr.std(ddof=1); n = len(arr)
                rec["ddauroc"] = round(float(mean), 4)
                if sd > 0:
                    half = tdist.ppf(0.975, n - 1) * sd / np.sqrt(n)
                    rec["ci_lo"] = round(float(mean - half), 4)
                    rec["ci_hi"] = round(float(mean + half), 4)
                    rec["raw_p"] = float(tdist.sf(mean / (sd / np.sqrt(n)), df=n - 1))
                else:
                    rec["raw_p"] = 0.0 if mean > 0 else 1.0
        recs.append(rec)

    bt_full = config_scores(sm, base_mask, metric, "variant").reindex(preds).idxmax()
    alt = config_scores(sm, base_mask, metric, "gene").reindex(preds).idxmax()
    recs.append({"dimension": "evaluation_unit", "numerical": "YES" if alt != bt_full else "NO",
                 "new_top": alt if alt != bt_full else "", "ddauroc": np.nan, "ci_lo": np.nan,
                 "ci_hi": np.nan, "raw_p": np.nan, "holm_p": np.nan, "supported": "—"})

    df = pd.DataFrame(recs)
    tested = df["raw_p"].notna().to_numpy()
    if tested.any():
        df.loc[tested, "holm_p"] = _holm(df.loc[tested, "raw_p"].to_numpy(), alpha)
        df["supported"] = np.where(df["holm_p"].notna() & (df["holm_p"] < alpha), "YES",
                                   np.where(df["numerical"] == "YES",
                                            np.where(df["dimension"] == "evaluation_unit", "—", "NO"),
                                            "—"))
    return df


def reliability_warning(n_gene_clusters):
    if n_gene_clusters >= GENE_WARNING_THRESHOLD:
        return None
    return (f"STATISTICAL RELIABILITY WARNING\n"
            f"  Independent gene clusters: {n_gene_clusters}\n"
            f"  Selection-aware reversal inference may have limited calibration and/or\n"
            f"  power below ~{GENE_WARNING_THRESHOLD} genes. Interpret inferential reversal\n"
            f"  claims cautiously (simulation-based guidance, not a universal threshold).")


def power_guidance(n_gene_clusters, target_ddauroc=0.05):
    ks = sorted(VALIDATED_POWER)
    lines = [f"  Observed gene clusters (resampling units): {n_gene_clusters}; target ΔΔAUROC ~= {target_ddauroc:.02f}",
             "  Validated cross-fit power at ΔΔAUROC ~= 0.05 (measured points):"]
    for k in ks:
        lines.append(f"    {k:>3d} genes: {VALIDATED_POWER[k]*100:.1f}%")
    lines += ["  These validated power estimates apply only to the simulated gene counts",
              "  and should not be directly extrapolated to larger benchmarks; power",
              "  depends on effect size, gene dependence, benchmark composition, and",
              "  analysis design.",
              "  (simulation-based guidance, not a biological rule)."]
    return "\n".join(lines)
