"""Benchmark Reliability Validation Suite."""
from __future__ import annotations

import numpy as np
import pandas as pd

from variantbenchpy.harmonize.schema import COL_GENE, COL_VARIANT_ID
from variantbenchpy import VariantSet, CanonicalKey
from variantbenchpy.datasim import simulate_benchmark, scenario_specs
from variantbenchpy.evaluation.reliability import (
    build_design_grid, _grid_frames, one_factor_reversals, kendalls_w,
    _axes, config_scores, _delta_contrast_ci,
)

_COLS = {"P1": "am_pathogenicity", "P2": "EVE_scores", "P3": "pph2_prob",
         "P4": "sift_score", "P5": "llr"}
_PRED_COLS = {"am_pathogenicity": ("P1", False, 0.5), "EVE_scores": ("P2", False, 0.5),
              "pph2_prob": ("P3", False, 0.5), "sift_score": ("P4", True, 0.05),
              "llr": ("P5", True, None)}


def build_scorematrix(specs, n_genes=50, seed=0, min_variants=5, max_variants=90):
    variants, raw = simulate_benchmark(n_genes=n_genes, min_variants=min_variants,
                                       max_variants=max_variants, specs=specs, seed=seed)
    wide = variants.copy()
    rng = np.random.default_rng(seed)
    wide["review_status"] = rng.choice(["1-star", "2-star", "3-star"], len(wide), p=[0.5, 0.3, 0.2])
    for name, rd in raw.items():
        sc = rd.columns[-1]; idc, subc = rd.columns[0], rd.columns[1]
        wide = wide.merge(rd.rename(columns={idc: "uniprot_id", subc: "protein_variant",
                                             sc: _COLS[name]}),
                          on=["uniprot_id", "protein_variant"], how="left")
    vs = VariantSet.from_dataframe(
        wide, column_map={"gene": "gene", "protein_id": "uniprot_id",
                          "protein_sub": "protein_variant", "clinical_significance": "clnsig"},
        key=CanonicalKey.PROTEIN, extras={"review_status": "review_status"},
        predictor_columns=_PRED_COLS)
    genes = vs.variants.set_index("variant_id")["gene"]
    grp = {g: f"grp{gi % 3}" for gi, g in enumerate(sorted(genes.unique()))}
    gg = genes.map(grp); gg.index = vs.score_matrix.scores.index
    rs = vs.variants.set_index("variant_id")["review_status"].reindex(vs.score_matrix.scores.index)
    ev = {"review>=2star": rs.isin(["2-star", "3-star"])}
    return vs.score_matrix, gg, ev


def audit_lite(sm, gene_groups, evidence_masks, n_boot_rev=200, seed=0):
    preds = sm.predictor_names
    configs = build_design_grid(sm, "auroc", gene_groups, evidence_masks)
    ranks_df, _, _ = _grid_frames(configs, preds)
    w = kendalls_w(ranks_df[preds])
    base_top, rev = one_factor_reversals(sm, "auroc", gene_groups, evidence_masks,
                                         n_boot=n_boot_rev, seed=seed)
    retention = float((ranks_df[base_top] == 1).mean())
    return {"w": w, "winner_retention": retention,
            "n_numerical": int((rev["reversal"] == "YES").sum()),
            "n_ranking": int((rev["support"] == "ranking").sum()),
            "n_relperf": int((rev["support"] == "relperf").sum())}


def _naive_relperf_detect(sm, gene_groups, evidence_masks, n_boot=200, seed=0):
    _, rev = one_factor_reversals(sm, "auroc", gene_groups, evidence_masks, n_boot=n_boot, seed=seed)
    return bool((rev["support"] == "relperf").any())


def _split_relperf_detect(sm, gene_groups, evidence_masks, n_boot=200, seed=0):
    idx = sm.scores.index
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(idx)
    uniq = pd.unique(genes.dropna())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    sel_genes = set(uniq[perm[: len(uniq) // 2]])
    sel_mask = pd.Series(genes.isin(sel_genes).to_numpy(), index=idx)
    test_mask = ~sel_mask
    cov, unit, comp, evid = _axes(sm, gene_groups, evidence_masks)
    preds = sm.predictor_names
    base_masks = {"coverage": ("available", cov["available"]),
                  "composition": ("all-genes", comp["all-genes"]),
                  "evidence": ("all", evid["all"])}
    base_mask = base_masks["coverage"][1] & base_masks["composition"][1] & base_masks["evidence"][1]
    base_top_sel = config_scores(sm, base_mask & sel_mask, "auroc", "variant").reindex(preds).idxmax()
    levels = {"coverage": cov, "composition": comp, "evidence": evid}
    for dim, options in levels.items():
        for lvl, m in options.items():
            if lvl == base_masks[dim][0]:
                continue
            pert = m.copy()
            for other, (blvl, bmask) in base_masks.items():
                if other != dim:
                    pert = pert & bmask
            top_sel = config_scores(sm, pert & sel_mask, "auroc", "variant").reindex(preds).idxmax()
            if top_sel != base_top_sel:
                _, dlo, dhi = _delta_contrast_ci(sm, base_mask & test_mask, pert & test_mask,
                                                 top_sel, base_top_sel, n_boot=n_boot,
                                                 ci=1 - 0.05 / 3, seed=seed)
                if (dlo > 0) or (dhi < 0):
                    return True
                break
    return False


def structural_power(levels, n_reps=200, n_genes=90, seed0=0, n_boot=200, max_variants=80):
    from variantbenchpy.datasim import structural_specs
    rows = []
    for eff in levels:
        specs = structural_specs(eff)
        num = sup = 0
        for i in range(n_reps):
            sm, gg, ev = build_scorematrix(specs, n_genes=n_genes, seed=seed0 + i, max_variants=max_variants)
            _, rev = one_factor_reversals(sm, "auroc", gg, ev, n_boot=1, seed=1)
            num += (rev.loc[rev.dimension == "composition", "reversal"] == "YES").any()
            sup += _split_relperf_detect(sm, gg, ev, n_boot=n_boot, seed=1)
        rows.append({"effect": eff, "n_reps": n_reps,
                     "numerical_rate": round(num / n_reps, 3),
                     "supported_rate": round(sup / n_reps, 3)})
    return pd.DataFrame(rows)


def naive_vs_selection(effect, n_reps=200, n_genes=90, seed0=0, n_boot=200):
    from variantbenchpy.datasim import structural_specs
    specs = structural_specs(effect)
    naive = split = 0
    for i in range(n_reps):
        sm, gg, ev = build_scorematrix(specs, n_genes=n_genes, seed=seed0 + i, max_variants=80)
        naive += _naive_relperf_detect(sm, gg, ev, n_boot=n_boot, seed=1)
        split += _split_relperf_detect(sm, gg, ev, n_boot=n_boot, seed=1)
    return {"effect": effect, "n_reps": n_reps,
            "naive_fp_rate": round(naive / n_reps, 3),
            "selection_aware_fp_rate": round(split / n_reps, 3)}


def negative_controls(n_reps=40, n_genes=60, seed0=0, n_boot=150):
    rows = []
    for kind in ["stable", "dominant", "equal_performance_null", "close",
                 "gene_dependent", "informative_missingness"]:
        num = ret = sup = 0
        for i in range(n_reps):
            sm, gg, ev = build_scorematrix(scenario_specs(kind), n_genes=n_genes, seed=seed0 + i)
            r = audit_lite(sm, gg, ev, n_boot_rev=n_boot, seed=1)
            num += r["n_numerical"] > 0
            sup += _split_relperf_detect(sm, gg, ev, n_boot=n_boot, seed=1)
            ret += r["winner_retention"]
        rows.append({"scenario": kind, "n_reps": n_reps,
                     "mean_winner_retention": round(ret / n_reps, 2),
                     "numerical_rate": round(num / n_reps, 2),
                     "supported_rate": round(sup / n_reps, 2)})
    return pd.DataFrame(rows)


def realized_effect(effect, n_genes=200, reps=4, seed0=0, max_variants=80):
    from variantbenchpy.datasim import structural_specs
    from variantbenchpy.metrics.fast import fast_auroc
    tgt, non, ddd = [], [], []
    for i in range(reps):
        sm, gg, ev = build_scorematrix(structural_specs(effect), n_genes=n_genes,
                                       seed=seed0 + i, max_variants=max_variants)
        genes = sm.table.set_index("variant_id")["gene"]
        grp = {g: gi % 3 for gi, g in enumerate(sorted(genes.unique()))}
        gidx = genes.map(grp); gidx.index = sm.scores.index
        y = sm.labels
        def dauroc(mask):
            mm = mask & y.notna()
            yy = y[mm].to_numpy()
            return (fast_auroc(yy, sm.scores["P2"][mm].to_numpy())
                    - fast_auroc(yy, sm.scores["P1"][mm].to_numpy()))
        t = dauroc(gidx == 1); n = dauroc(gidx.isin([0, 2]))
        tgt.append(t); non.append(n); ddd.append(t - n)
    return {"effect": effect,
            "target_dAUROC": round(float(np.mean(tgt)), 4),
            "nontarget_dAUROC": round(float(np.mean(non)), 4),
            "realized_ddAUROC": round(float(np.mean(ddd)), 4)}


def _ddelta_boot(sm, base_mask, pert_mask, a, b, n_boot, seed):
    from variantbenchpy.metrics.fast import fast_auroc
    labels = sm.labels
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    idx = sm.scores.index
    lab = labels.notna().to_numpy()
    y = labels[labels.notna()].to_numpy()
    g = genes[labels.notna()].to_numpy()
    sa = sm.scores[a][labels.notna()].to_numpy()
    sb = sm.scores[b][labels.notna()].to_numpy()
    bm = base_mask.reindex(idx).fillna(False).to_numpy(bool)[lab]
    pm = pert_mask.reindex(idx).fillna(False).to_numpy(bool)[lab]
    uniq = np.unique(g)
    base_idx = {gr: np.where((g == gr) & bm)[0] for gr in uniq}
    pert_idx = {gr: np.where((g == gr) & pm)[0] for gr in uniq}
    def D(ix):
        if ix.size == 0 or np.unique(y[ix]).size < 2:
            return np.nan
        return fast_auroc(y[ix], sa[ix]) - fast_auroc(y[ix], sb[ix])
    rng = np.random.default_rng(seed)
    dd = np.empty(n_boot)
    for k in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        d0 = D(np.concatenate([base_idx[c] for c in chosen]))
        d1 = D(np.concatenate([pert_idx[c] for c in chosen]))
        dd[k] = d1 - d0
    dd = dd[~np.isnan(dd)]
    if dd.size < 2:
        return np.nan, np.nan
    return float(dd.mean()), float(dd.std(ddof=1))


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


def _ddelta_point(sm, base_mask, pert_mask, a, b):
    from variantbenchpy.metrics.fast import fast_auroc
    labels = sm.labels
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
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


def _crossfit_relperf_detect(sm, gene_groups, evidence_masks, n_boot=200, seed=0,
                             alpha=0.05, k_folds=5):
    """K-fold gene-level cross-fitting. For each dimension and fold, select the reversal
    on the other K-1 folds and compute the held-out DeltaD point on the left-out fold.
    Combine the K independent held-out points with a one-sided one-sample t-test
    (between-fold variance, t for small K), then Holm across dimensions."""
    from scipy.stats import t as tdist
    idx = sm.scores.index
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(idx)
    uniq = pd.unique(genes.dropna())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    folds = [set(uniq[perm[j::k_folds]]) for j in range(k_folds)]
    fold_mask = [pd.Series(genes.isin(f).to_numpy(), index=idx) for f in folds]

    cov, unit, comp, evid = _axes(sm, gene_groups, evidence_masks)
    preds = sm.predictor_names
    base_masks = {"coverage": ("available", cov["available"]),
                  "composition": ("all-genes", comp["all-genes"]),
                  "evidence": ("all", evid["all"])}
    base_masks["_all"] = (base_masks["coverage"][1] & base_masks["composition"][1]
                          & base_masks["evidence"][1])
    base_mask = base_masks["_all"]
    dim_options = {"coverage": cov, "composition": comp, "evidence": evid}

    pvals = []
    for dim, options in dim_options.items():
        dks = []
        for k in range(k_folds):
            test_m = fold_mask[k]
            sel = _select_flip(sm, dim, base_masks, options, ~test_m, preds)
            if not sel:
                continue
            base_top, new_top, pert = sel
            dk = _ddelta_point(sm, base_mask & test_m, pert & test_m, new_top, base_top)
            if not np.isnan(dk):
                dks.append(dk)
        if len(dks) < 3:
            continue
        dks = np.array(dks, float)
        sd = dks.std(ddof=1)
        if sd <= 0:
            pvals.append(0.0 if dks.mean() > 0 else 1.0)
            continue
        tstat = dks.mean() / (sd / np.sqrt(len(dks)))
        pvals.append(float(tdist.sf(tstat, df=len(dks) - 1)))
    if not pvals:
        return False
    pvals = sorted(pvals)
    m = len(pvals)
    return any(p <= alpha / (m - i) for i, p in enumerate(pvals))


def method_comparison(effect, n_reps=40, n_genes=80, seed0=0, n_boot=200):
    from variantbenchpy.datasim import structural_specs
    specs = structural_specs(effect)
    naive = split = xfit = 0
    for i in range(n_reps):
        sm, gg, ev = build_scorematrix(specs, n_genes=n_genes, seed=seed0 + i, max_variants=80)
        naive += _naive_relperf_detect(sm, gg, ev, n_boot=n_boot, seed=1)
        split += _split_relperf_detect(sm, gg, ev, n_boot=n_boot, seed=1)
        xfit += _crossfit_relperf_detect(sm, gg, ev, n_boot=n_boot, seed=1)
    return {"effect": effect, "n_reps": n_reps,
            "naive": round(naive / n_reps, 3), "split": round(split / n_reps, 3),
            "crossfit": round(xfit / n_reps, 3)}


# --- final validation grid with Wilson 95% CIs --------------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (round((c - h) / d, 3), round((c + h) / d, 3))


_METHOD_FN = {
    "naive": lambda sm, gg, ev, s: _naive_relperf_detect(sm, gg, ev, n_boot=100, seed=1),
    "split": lambda sm, gg, ev, s: _split_relperf_detect(sm, gg, ev, n_boot=100, seed=1),
    "crossfit": lambda sm, gg, ev, s: _crossfit_relperf_detect(sm, gg, ev, seed=1),
}


def run_grid(methods, effects, gene_counts, reps, seed0=0, max_variants=70,
             csv_path=None, verbose=True):
    """methods: dict {name: n_reps}. Returns tidy DataFrame with rate + Wilson CI.
    Prints each row as it finishes and (optionally) appends to csv_path for resumability."""
    import time
    from variantbenchpy.datasim import structural_specs
    t0 = time.time()
    rows = []
    for ng in gene_counts:
        for eff in effects:
            for mname in methods:
                N = methods[mname]
                fn = _METHOD_FN[mname]
                hits = 0
                for i in range(N):
                    sm, gg, ev = build_scorematrix(structural_specs(eff), n_genes=ng,
                                                   seed=seed0 + i, max_variants=max_variants)
                    hits += fn(sm, gg, ev, seed0 + i)
                lo, hi = wilson_ci(hits, N)
                row = {"method": mname, "genes": ng, "effect": eff, "n_reps": N,
                       "rate": round(hits / N, 3), "ci_lo": lo, "ci_hi": hi}
                rows.append(row)
                if verbose:
                    print(f"  {mname:8s} genes={ng:3d} eff={eff:.1f}  "
                          f"rate={row['rate']:.3f} [{lo:.3f},{hi:.3f}]  ({time.time()-t0:.0f}s)")
                if csv_path:
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return pd.DataFrame(rows)
