"""Rank Reliability Audit."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from variantbenchpy.coverage.analysis import shared_subset_mask
from variantbenchpy.evaluation.stability import kendalls_w, probability_best
from variantbenchpy.evaluation.weighting import config_scores, gene_weighted_metric
from variantbenchpy.harmonize.schema import COL_GENE, COL_VARIANT_ID
from variantbenchpy.metrics.classification import auprc, binary_metrics
from variantbenchpy.metrics.fast import fast_auroc

_DESIGN_DIMS = ["coverage", "evaluation_unit", "composition", "evidence"]
_METRICS = ["auroc", "auprc", "mcc"]
_REFERENCE = {"coverage": "available", "evaluation_unit": "variant",
              "composition": "all-genes", "evidence": "all"}
BOOT_REFERENCE_NOTE = ("Bootstrap P(#1): gene-clustered resampling under the shared-subset, "
                       "variant-weighted, AUROC reference design.")


def _axes(sm, gene_groups, evidence_masks):
    idx = sm.scores.index
    cov = {"available": pd.Series(True, index=idx), "shared": shared_subset_mask(sm)}
    unit = {"variant": "variant", "equal-gene": "gene"}
    comp = {"all-genes": pd.Series(True, index=idx)}
    if gene_groups is not None:
        gg = gene_groups.reindex(idx)
        for grp in sorted(pd.unique(gg.dropna())):
            comp[f"genes:{grp}"] = (gg == grp)
    evid = {"all": pd.Series(True, index=idx)}
    if evidence_masks:
        for k, m in evidence_masks.items():
            evid[k] = m.reindex(idx).fillna(False)
    return cov, unit, comp, evid


def build_design_grid(sm, metric="auroc", gene_groups=None, evidence_masks=None):
    cov, unit, comp, evid = _axes(sm, gene_groups, evidence_masks)
    configs = []
    for ck, cm in cov.items():
        for uk, uw in unit.items():
            for pk, pm in comp.items():
                for ek, em in evid.items():
                    mask = cm & pm & em
                    configs.append({"coverage": ck, "evaluation_unit": uk,
                                    "composition": pk, "evidence": ek, "weighting": uw,
                                    "mask": mask, "scores": config_scores(sm, mask, metric, uw)})
    return configs


def _grid_frames(configs, preds):
    ranks, values, dims = [], [], []
    for c in configs:
        s = c["scores"].reindex(preds)
        if s.notna().all():
            dims.append({k: c[k] for k in _DESIGN_DIMS})
            values.append(s.to_dict())
            ranks.append(s.rank(ascending=False, method="min").to_dict())
    dims_df = pd.DataFrame(dims)
    ranks_df = pd.concat([dims_df, pd.DataFrame(ranks)], axis=1)
    values_df = pd.concat([dims_df, pd.DataFrame(values)], axis=1)
    return ranks_df, values_df, dims_df


def _reference_row(values_df):
    m = pd.Series(True, index=values_df.index)
    for d, lvl in _REFERENCE.items():
        m &= values_df[d] == lvl
    hits = values_df[m]
    return hits.iloc[0] if len(hits) else values_df.iloc[0]


def rank_distribution(ranks_df, preds):
    M = len(preds)
    dist = pd.DataFrame(index=preds, columns=[f"#{r}" for r in range(1, M + 1)], dtype="float64")
    ent = {}
    for p in preds:
        counts = ranks_df[p].value_counts().reindex(range(1, M + 1), fill_value=0)
        prop = counts / counts.sum()
        dist.loc[p] = prop.to_numpy()
        q = prop[prop > 0].to_numpy()
        h = float(-(q * np.log(q)).sum() / np.log(M)) if M > 1 else 0.0
        ent[p] = max(0.0, h)
    dist["prop_rank1"] = dist["#1"]
    return dist, pd.Series(ent, name="entropy")


def magnitude_sensitivity(values_df, ranks_df, preds):
    ref = _reference_row(values_df)
    rows = []
    for p in preds:
        v = values_df[p].astype("float64")
        deltas = v - float(ref[p])
        rows.append({"predictor": p,
                     "rank_best": int(ranks_df[p].min()), "rank_worst": int(ranks_df[p].max()),
                     "auroc_min": round(float(v.min()), 3), "auroc_max": round(float(v.max()), 3),
                     "median_abs_delta": round(float(deltas.abs().median()), 3),
                     "worst_delta": round(float(deltas.min()), 3)})
    return pd.DataFrame(rows)


def pairwise_win_proportion(ranks_df, preds):
    W = pd.DataFrame(index=preds, columns=preds, dtype="float64")
    for a in preds:
        for b in preds:
            W.loc[a, b] = np.nan if a == b else round(float((ranks_df[a] < ranks_df[b]).mean()), 3)
    return W


def rank_sensitivity(ranks_df, preds):
    out = {}
    for D in _DESIGN_DIMS:
        others = [d for d in _DESIGN_DIMS if d != D]
        shifts = []
        for _, grp in ranks_df.groupby(others):
            if grp[D].nunique() < 2:
                continue
            R = grp[preds].to_numpy("float64")
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    shifts.append(np.nanmean(np.abs(R[i] - R[j])))
        out[D] = float(np.mean(shifts)) if shifts else 0.0
    s = pd.Series(out).sort_values(ascending=False)
    share = (s / s.sum()).round(3) if s.sum() > 0 else s
    return s, share


def objective_sensitivity(sm, gene_groups=None):
    idx = sm.scores.index
    ref = pd.Series(True, index=idx)
    preds = sm.predictor_names
    rankings = {m: config_scores(sm, ref, m, "variant").reindex(preds).rank(ascending=False, method="min")
                for m in _METRICS}
    rank_tbl = pd.DataFrame(rankings)
    rows = []
    for a, b in [("auroc", "auprc"), ("auroc", "mcc"), ("auprc", "mcc")]:
        rho = rank_tbl[[a, b]].corr(method="spearman").iloc[0, 1]
        rows.append({"comparison": f"{a.upper()} vs {b.upper()}",
                     "spearman_rank_corr": round(float(rho), 3),
                     "same_top_predictor": bool(rank_tbl[a].idxmin() == rank_tbl[b].idxmin())})
    return rank_tbl, pd.DataFrame(rows)


def _paired_delta_ci(sm, mask, a, b, n_boot=400, ci=0.95, seed=0):
    labels = sm.labels
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    sub = mask.reindex(sm.scores.index).fillna(False) & labels.notna()
    y = labels[sub].to_numpy(); g = genes[sub].to_numpy()
    sa = sm.scores[a][sub].to_numpy(); sb = sm.scores[b][sub].to_numpy()
    keep = ~(np.isnan(y) | np.isnan(sa) | np.isnan(sb))
    y, g, sa, sb = y[keep], g[keep], sa[keep], sb[keep]
    if y.size == 0 or np.unique(y).size < 2:
        return np.nan, np.nan, np.nan
    point = fast_auroc(y, sa) - fast_auroc(y, sb)
    uniq = np.unique(g)
    idx_by = {gr: np.where(g == gr)[0] for gr in uniq}
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for k in range(n_boot):
        ix = np.concatenate([idx_by[c] for c in rng.choice(uniq, size=uniq.size, replace=True)])
        d[k] = fast_auroc(y[ix], sa[ix]) - fast_auroc(y[ix], sb[ix])
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(point), float(lo), float(hi)


def _delta_contrast_ci(sm, base_mask, pert_mask, a, b, n_boot=400, ci=0.95, seed=0):
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

    D0 = D(np.concatenate([base_idx[gr] for gr in uniq]))
    D1 = D(np.concatenate([pert_idx[gr] for gr in uniq]))
    point = D1 - D0
    rng = np.random.default_rng(seed)
    dd = np.empty(n_boot)
    for k in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        d0 = D(np.concatenate([base_idx[c] for c in chosen]))
        d1 = D(np.concatenate([pert_idx[c] for c in chosen]))
        dd[k] = d1 - d0
    dd = dd[~np.isnan(dd)]
    lo, hi = (np.percentile(dd, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
              if dd.size else (np.nan, np.nan))
    return float(point), float(lo), float(hi)


def one_factor_reversals(sm, metric="auroc", gene_groups=None, evidence_masks=None,
                         n_boot=400, seed=0):
    cov, unit, comp, evid = _axes(sm, gene_groups, evidence_masks)
    preds = sm.predictor_names
    base_masks = {"coverage": ("available", cov["available"]),
                  "composition": ("all-genes", comp["all-genes"]),
                  "evidence": ("all", evid["all"])}
    base_mask = base_masks["coverage"][1] & base_masks["composition"][1] & base_masks["evidence"][1]
    base_top = config_scores(sm, base_mask, metric, "variant").reindex(preds).idxmax()

    def classify(delta_lo, dd_lo, dd_hi):
        if (dd_lo > 0) or (dd_hi < 0):
            return "relperf"
        if delta_lo > 0:
            return "ranking"
        return "numerical"

    levels = {"coverage": cov, "composition": comp, "evidence": evid}
    rows = []
    for dim, options in levels.items():
        rec = {"dimension": dim, "reversal": "NO", "new_top": "", "delta": np.nan,
               "lo": np.nan, "hi": np.nan, "dD": np.nan, "dD_lo": np.nan,
               "dD_hi": np.nan, "support": ""}
        for lvl, m in options.items():
            if lvl == base_masks[dim][0]:
                continue
            mask = m.copy()
            for other, (blvl, bmask) in base_masks.items():
                if other != dim:
                    mask = mask & bmask
            top = config_scores(sm, mask, metric, "variant").reindex(preds).idxmax()
            if top != base_top:
                dlt, lo, hi = _paired_delta_ci(sm, mask, top, base_top, n_boot=n_boot, seed=seed)
                dd, dlo, dhi = _delta_contrast_ci(sm, base_mask, mask, top, base_top,
                                                  n_boot=n_boot, seed=seed)
                rec = {"dimension": dim, "reversal": "YES", "new_top": f"{lvl}->{top}",
                       "delta": round(dlt, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                       "dD": round(dd, 4), "dD_lo": round(dlo, 4), "dD_hi": round(dhi, 4),
                       "support": classify(lo, dlo, dhi)}
                break
        rows.append(rec)

    alt_top = config_scores(sm, base_mask, metric, "gene").reindex(preds).idxmax()
    if alt_top != base_top:
        dlt, lo, hi = _paired_delta_ci(sm, base_mask, alt_top, base_top, n_boot=n_boot, seed=seed)
        rows.append({"dimension": "evaluation_unit", "reversal": "YES",
                     "new_top": f"equal-gene->{alt_top}", "delta": round(dlt, 4),
                     "lo": round(lo, 4), "hi": round(hi, 4), "dD": np.nan,
                     "dD_lo": np.nan, "dD_hi": np.nan,
                     "support": "ranking" if lo > 0 else "numerical"})
    else:
        rows.append({"dimension": "evaluation_unit", "reversal": "NO", "new_top": "",
                     "delta": np.nan, "lo": np.nan, "hi": np.nan, "dD": np.nan,
                     "dD_lo": np.nan, "dD_hi": np.nan, "support": ""})
    return base_top, pd.DataFrame(rows)


def kendalls_w_ci(sm, metric="auroc", gene_groups=None, evidence_masks=None,
                  n_boot=200, ci=0.95, seed=0):
    configs = build_design_grid(sm, metric, gene_groups, evidence_masks)
    preds = sm.predictor_names
    P = len(preds)
    ranks_df, _, _ = _grid_frames(configs, preds)
    point = kendalls_w(ranks_df[preds])
    labeled = sm.labels.notna()
    order = sm.scores.index[labeled]
    y = sm.labels[labeled].to_numpy()
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    gene_lbl = genes[labeled].to_numpy()
    score = {p: sm.scores[p][labeled].to_numpy() for p in preds}
    uniq = np.unique(gene_lbl); G = uniq.size
    gid = pd.Series(np.arange(G), index=uniq).reindex(gene_lbl).to_numpy()
    idx_by = [np.where(gid == k)[0] for k in range(G)]
    variant_cfgs, gene_cfgs = [], []
    for c in configs:
        cmask = c["mask"].reindex(order).fillna(False).to_numpy(bool)
        if c["weighting"] == "variant":
            variant_cfgs.append(cmask)
        else:
            present = sorted({gid[i] for i in np.where(cmask)[0]})
            A = np.full((len(present), P), np.nan)
            for row, k in enumerate(present):
                sel = np.where(cmask & (gid == k))[0]
                if np.unique(y[sel]).size == 2:
                    for j, p in enumerate(preds):
                        A[row, j] = fast_auroc(y[sel], score[p][sel])
            gene_cfgs.append((np.array(present), A))
    rng = np.random.default_rng(seed)
    ws = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, G, size=G), minlength=G)
        ix = (np.concatenate([np.tile(idx_by[k], counts[k]) for k in range(G) if counts[k]])
              if variant_cfgs else None)
        rows = []
        for cmask in variant_cfgs:
            sel = ix[cmask[ix]]
            if sel.size == 0 or np.unique(y[sel]).size < 2:
                continue
            vals = np.array([fast_auroc(y[sel], score[p][sel]) for p in preds])
            if np.isnan(vals).any():
                continue
            rows.append(pd.Series(vals).rank(ascending=False, method="min").to_numpy())
        for present, A in gene_cfgs:
            w = counts[present].astype("float64") * (~np.isnan(A[:, 0]))
            den = w.sum()
            if den <= 0:
                continue
            vals = (w @ np.nan_to_num(A)) / den
            rows.append(pd.Series(vals).rank(ascending=False, method="min").to_numpy())
        if len(rows) >= 2:
            ws.append(kendalls_w(pd.DataFrame(rows, columns=preds)))
    ws = np.array(ws, "float64"); ws = ws[~np.isnan(ws)]
    lo, hi = (np.percentile(ws, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
              if ws.size else (np.nan, np.nan))
    return float(point), float(lo), float(hi), int(ws.size)


@dataclass
class RankReliabilityAudit:
    n_variants: int; n_genes: int; preds: list; n_specs: int
    w_point: float; w_lo: float; w_hi: float
    winner: str; winner_retained: int; winner_retention: float
    rank_dist: pd.DataFrame; entropy: pd.Series; magnitude: pd.DataFrame
    prop_rank1: pd.Series; boot_prob1: pd.Series
    design_sensitivity: pd.Series; design_share: pd.Series
    objective_agreement: pd.DataFrame; selection_aware: pd.DataFrame; pairwise: pd.DataFrame
    n_gene_clusters: int = 0

    def format(self) -> str:
        M = len(self.preds)
        order = self.prop_rank1.sort_values(ascending=False).index.tolist()
        L = ["RANK RELIABILITY AUDIT", "=" * 22,
             f"Variants: {self.n_variants:,}   Genes: {self.n_genes}   "
             f"Predictors: {len(self.preds)}   Specifications: {self.n_specs}", "",
             "WINNER STABILITY",
             f"  Reference #1 = {self.winner};  retained #1 in "
             f"{self.winner_retained}/{self.n_specs} configs (retention {self.winner_retention:.2f})",
             "GLOBAL CONCORDANCE (whole ranking)",
             f"  Kendall's W = {self.w_point:.2f}  95% CI [{self.w_lo:.2f}, {self.w_hi:.2f}]",
             "  (point estimate with CI; interpret alongside the other reliability measures)", "",
             "RANK DISTRIBUTION (proportion of configurations)"]
        L.append("  " + f"{'':<14}" + "".join(f"{'#'+str(r):>7}" for r in range(1, M + 1)))
        for p in order:
            L.append("  " + f"{p:<14}" + "".join(f"{self.rank_dist.loc[p, f'#{r}']:>7.2f}"
                                                 for r in range(1, M + 1)))
        L += ["", "MAGNITUDE SENSITIVITY  (dAUROC = AUROC_config - AUROC_reference; "
              "reference = shared-coverage / all evidence / pooled_variant / primary composition)",
              "  " + f"{'':<14}{'rank':>8}{'AUROC range':>16}{'med|dAUROC|':>13}{'worst dAUROC':>14}"]
        mg = self.magnitude.set_index("predictor")
        for p in order:
            r = mg.loc[p]
            L.append("  " + f"{p:<14}{str(int(r['rank_best']))+'-'+str(int(r['rank_worst'])):>8}"
                     f"{f'{r.auroc_min:.3f}-{r.auroc_max:.3f}':>16}"
                     f"{r.median_abs_delta:>13.3f}{r.worst_delta:>14.3f}")
        L += ["", "RANK DISPERSION (normalised entropy: 0=fixed rank, 1=dispersed)"]
        for p in order:
            L.append(f"  {p:<14}{self.entropy[p]:>6.2f}")
        L += ["", "STATISTICAL UNCERTAINTY IN #1", f"  ({BOOT_REFERENCE_NOTE})",
              f"  {'':<14}{'prop.#1(cfg)':>14}{'P(#1) boot':>12}"]
        for p in order:
            L.append(f"  {p:<14}{self.prop_rank1[p]:>14.2f}{self.boot_prob1.get(p, float('nan')):>12.2f}")
        L += ["", "DESIGN SENSITIVITY (normalised rank-sensitivity share; DESCRIPTIVE)"]
        for d, v in self.design_sensitivity.items():
            L.append(f"  {d:<18}{v:>6.2f}   (share {self.design_share[d]:.0%})")
        L += ["", "OBJECTIVE SENSITIVITY (metrics answer different questions; not fragility)"]
        for _, r in self.objective_agreement.iterrows():
            L.append(f"  {r['comparison']:<16} rank-corr={r['spearman_rank_corr']:>5.2f}  "
                     f"same #1: {'yes' if r['same_top_predictor'] else 'NO'}")
        from variantbenchpy.evaluation._selaware import reliability_warning, power_guidance
        L += ["", "SELECTION-AWARE REVERSALS (K-fold cross-fit; Holm across dimensions)",
              "  " + f"{'Dimension':<16}{'Numerical':<11}{'Supported':<11}"
              f"{'ΔΔAUROC [95% CI]':<28}{'p_adj'}"]
        for _, r in self.selection_aware.iterrows():
            if pd.isna(r["ddauroc"]):
                cis = ""
            elif pd.isna(r["ci_lo"]):
                cis = f"{r['ddauroc']:+.3f}"
            else:
                cis = f"{r['ddauroc']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
            padj = "" if pd.isna(r.get("holm_p")) else f"{r['holm_p']:.3f}"
            L.append("  " + f"{r['dimension']:<16}{r['numerical']:<11}{r['supported']:<11}"
                     f"{cis:<28}{padj}")
        warn = reliability_warning(self.n_gene_clusters)
        L += ["", "CALIBRATION & POWER (simulation-validated operating characteristics)"]
        if warn:
            L += ["  " + ln for ln in warn.splitlines()]
        L += power_guidance(self.n_gene_clusters).splitlines()
        sup = self.selection_aware[self.selection_aware["supported"] == "YES"]
        verdict = ("no statistically supported change in relative predictor performance was "
                   "detected under any prespecified perturbation" if sup.empty else
                   "a statistically supported change in relative predictor performance was detected "
                   "under the prespecified " + ", ".join(sup["dimension"]) + " perturbation")
        L += ["", "INTERPRETATION",
              f"  Winner retention {self.winner_retention:.2f}; whole-ranking concordance "
              f"W={self.w_point:.2f} [CI {self.w_lo:.2f}, {self.w_hi:.2f}].",
              f"  After selection-aware inference, {verdict}.",
              f"  Ranking is most sensitive (descriptively) to {self.design_sensitivity.index[0]} "
              f"(share {self.design_share.iloc[0]:.0%})."]
        return "\n".join(L)


def rank_reliability_audit(sm, gene_groups=None, evidence_masks=None, metric="auroc",
                           n_boot_w=200, n_boot_prob=1000, n_boot_rev=400, seed=0):
    preds = sm.predictor_names
    configs = build_design_grid(sm, metric, gene_groups, evidence_masks)
    ranks_df, values_df, _ = _grid_frames(configs, preds)
    dist, ent = rank_distribution(ranks_df, preds)
    mag = magnitude_sensitivity(values_df, ranks_df, preds)
    sens, share = rank_sensitivity(ranks_df, preds)
    _, oagree = objective_sensitivity(sm, gene_groups)
    from variantbenchpy.evaluation._selaware import selection_aware_reversals
    sar = selection_aware_reversals(sm, gene_groups, evidence_masks, metric=metric, seed=seed)
    base_top = config_scores(sm, pd.Series(True, index=sm.scores.index), metric, "variant").reindex(preds).idxmax()
    pw = pairwise_win_proportion(ranks_df, preds)
    wpt, wlo, whi, _ = kendalls_w_ci(sm, metric, gene_groups, evidence_masks, n_boot=n_boot_w, seed=seed)
    pb, _ = probability_best(sm, metric="auroc", weighting="variant", n_boot=n_boot_prob, seed=seed)
    boot1 = pb.set_index("predictor")["prob_best"]
    n_specs = len(ranks_df)
    retained = int((ranks_df[base_top] == 1).sum())
    genes = sm.table[COL_GENE].nunique(dropna=True)
    return RankReliabilityAudit(
        n_variants=len(sm.scores), n_genes=int(genes), preds=preds, n_specs=n_specs,
        w_point=wpt, w_lo=wlo, w_hi=whi, winner=base_top, winner_retained=retained,
        winner_retention=round(retained / n_specs, 3) if n_specs else 0.0,
        rank_dist=dist, entropy=ent, magnitude=mag, prop_rank1=dist["prop_rank1"],
        boot_prob1=boot1, design_sensitivity=sens, design_share=share,
        objective_agreement=oagree, selection_aware=sar, pairwise=pw,
        n_gene_clusters=int(genes))
