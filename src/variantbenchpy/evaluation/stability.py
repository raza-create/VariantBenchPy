"""Ranking-fragility engine — the signature analysis.

Conventional benchmark tables report one ranking. But a ranking is the product of many
defensible choices: which variants (all vs shared), how they are weighted (variant vs
equal-gene), which metric (AUROC/AUPRC/MCC), which gene groups, which review levels.
This module runs the whole grid and asks how stable the answer to "who is best?" is.

Stability index (defined, not arbitrary): for predictor p let q_r be the fraction of
configurations in which p attains rank r. Stability(p) = 1 - H(q)/log(M) lies in [0, 1],
where 1 means p always holds the same rank and 0 means its rank is maximally spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from variantbenchpy.coverage.analysis import shared_subset_mask
from variantbenchpy.evaluation.weighting import config_scores, gene_weighted_metric
from variantbenchpy.harmonize.schema import COL_GENE, COL_VARIANT_ID
from variantbenchpy.metrics.classification import auprc, binary_metrics
from variantbenchpy.metrics.fast import fast_auroc
from variantbenchpy.predictors.base import ScoreMatrix


def benchmark_configurations(
    sm: ScoreMatrix,
    gene_groups: pd.Series | None = None,
    extra_masks: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(scores_by_config, ranks_by_config)`` over a default design battery."""
    shared = shared_subset_mask(sm)
    rows: dict[str, pd.Series] = {}
    rows["all / AUROC / variant"] = config_scores(sm, None, "auroc", "variant")
    rows["shared / AUROC / variant"] = config_scores(sm, shared, "auroc", "variant")
    rows["shared / AUROC / equal-gene"] = config_scores(sm, shared, "auroc", "gene")
    rows["shared / AUPRC / variant"] = config_scores(sm, shared, "auprc", "variant")
    rows["shared / AUPRC / equal-gene"] = config_scores(sm, shared, "auprc", "gene")
    rows["shared / MCC / variant"] = config_scores(sm, shared, "mcc", "variant")

    if gene_groups is not None:
        gg = gene_groups.reindex(sm.scores.index)
        for grp in pd.unique(gg.dropna()):
            m = shared & (gg == grp)
            rows[f"group={grp} / AUROC / variant"] = config_scores(sm, m, "auroc", "variant")

    if extra_masks:
        for label, m in extra_masks.items():
            rows[label] = config_scores(sm, m.reindex(sm.scores.index).fillna(False) & shared,
                                        "auroc", "variant")

    scores_df = pd.DataFrame(rows).T
    ranks_df = scores_df.apply(lambda r: r.rank(ascending=False, method="min"), axis=1)
    return scores_df, ranks_df


def _normalised_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0 or counts.size <= 1:
        return 0.0
    p = counts[counts > 0] / total
    H = -(p * np.log(p)).sum()
    return float(H / np.log(counts.size))


def kendalls_w(ranks_df: pd.DataFrame) -> float:
    """Kendall's coefficient of concordance across configurations (1 = perfect)."""
    K = len(ranks_df)
    M = ranks_df.shape[1]
    R = ranks_df.sum(axis=0).to_numpy()
    S = ((R - R.mean()) ** 2).sum()
    denom = K ** 2 * (M ** 3 - M)
    return float(12 * S / denom) if denom > 0 else np.nan


def ranking_fragility(ranks_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-predictor rank travel + Stability index, plus overall fragility summary."""
    n_config = len(ranks_df)
    preds = list(ranks_df.columns)
    M = len(preds)
    top_each = ranks_df.idxmin(axis=1)
    modal_top = top_each.mode().iloc[0]

    rows = []
    for p in preds:
        r = ranks_df[p]
        counts = r.value_counts().reindex(range(1, M + 1), fill_value=0).to_numpy().astype(float)
        stability = 1.0 - _normalised_entropy(counts)
        times1 = int((r == 1).sum())
        rows.append({"predictor": p, "mean_rank": round(float(r.mean()), 2),
                     "best_rank": int(r.min()), "worst_rank": int(r.max()),
                     "modal_rank": int(r.mode().iloc[0]),
                     "times_rank1": times1, "pct_rank1": round(100 * times1 / n_config, 1),
                     "stability": round(stability, 3)})
    frag = pd.DataFrame(rows).sort_values("mean_rank", ignore_index=True)
    summary = {
        "n_configs": n_config,
        "modal_top_predictor": modal_top,
        "top_changed_fraction": round(float((top_each != modal_top).mean()), 3),
        "n_distinct_winners": int(top_each.nunique()),
        "kendalls_w": round(kendalls_w(ranks_df), 3),
    }
    return frag, summary


def _metric_value(y, s, g, metric, weighting):
    if weighting == "gene":
        v, _ = gene_weighted_metric(y, s, g, metric=metric)
        return v
    if metric == "auroc":
        return fast_auroc(y, s)
    if metric == "auprc":
        return auprc(y, s)
    return binary_metrics(y, s)["mcc"]


def probability_best(sm: ScoreMatrix, mask: pd.Series | None = None, metric: str = "auroc",
                     weighting: str = "variant", n_boot: int = 1000, seed: int = 0
                     ) -> tuple[pd.DataFrame, int]:
    """Gene-clustered bootstrap probability that each predictor is the true #1."""
    labels = sm.labels
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    base = shared_subset_mask(sm) if mask is None else mask
    sub = base.reindex(sm.scores.index).fillna(False) & labels.notna()
    y = labels[sub].to_numpy()
    g = genes[sub].to_numpy()
    preds = sm.predictor_names
    S = {p: sm.scores[p][sub].to_numpy() for p in preds}
    uniq = np.unique(g)
    idx_by = {gr: np.where(g == gr)[0] for gr in uniq}
    rng = np.random.default_rng(seed)

    win = np.zeros(len(preds)); top2 = np.zeros(len(preds)); rank_sum = np.zeros(len(preds))
    valid = 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        ix = np.concatenate([idx_by[c] for c in chosen])
        yy = y[ix]; gg = g[ix]
        if np.unique(yy).size < 2:
            continue
        vals = np.array([_metric_value(yy, S[p][ix], gg, metric, weighting) for p in preds])
        if np.all(np.isnan(vals)):
            continue
        order = pd.Series(vals, index=preds).rank(ascending=False, method="min").to_numpy()
        win[int(np.nanargmax(vals))] += 1
        top2 += (order <= 2).astype(float)
        rank_sum += order
        valid += 1
    df = pd.DataFrame({"predictor": preds,
                       "prob_best": (win / valid).round(3),
                       "prob_top2": (top2 / valid).round(3),
                       "expected_rank": (rank_sum / valid).round(2)})
    return df.sort_values("prob_best", ascending=False, ignore_index=True), valid


def _bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, "float64")
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return np.clip(adj, 0, 1)


def pairwise_delta_auroc(sm: ScoreMatrix, mask: pd.Series | None = None,
                         n_boot: int = 1000, seed: int = 0, ci: float = 0.95,
                         alpha: float = 0.05) -> pd.DataFrame:
    """Paired gene-clustered DeltaAUROC for every predictor pair, BH-corrected."""
    labels = sm.labels
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    base = shared_subset_mask(sm) if mask is None else mask
    sub = base.reindex(sm.scores.index).fillna(False) & labels.notna()
    y = labels[sub].to_numpy(); g = genes[sub].to_numpy()
    preds = sm.predictor_names
    S = {p: sm.scores[p][sub].to_numpy() for p in preds}
    uniq = np.unique(g)
    idx_by = {gr: np.where(g == gr)[0] for gr in uniq}
    rng = np.random.default_rng(seed)

    boot_idx = [np.concatenate([idx_by[c] for c in rng.choice(uniq, size=uniq.size, replace=True)])
                for _ in range(n_boot)]

    recs = []
    for i, a in enumerate(preds):
        for b in preds[i + 1:]:
            point = fast_auroc(y, S[a]) - fast_auroc(y, S[b])
            deltas = np.empty(n_boot)
            for k, ix in enumerate(boot_idx):
                deltas[k] = fast_auroc(y[ix], S[a][ix]) - fast_auroc(y[ix], S[b][ix])
            deltas = deltas[~np.isnan(deltas)]
            lo, hi = np.percentile(deltas, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
            prob = float((deltas > 0).mean())
            p_val = 2 * min(prob, 1 - prob)
            p_val = max(p_val, 1.0 / (len(deltas) + 1))
            recs.append({"A": a, "B": b, "delta_auroc": round(point, 4),
                         "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                         "prob_A_gt_B": round(prob, 3), "p_value": p_val})
    df = pd.DataFrame(recs)
    df["p_adj"] = _bh(df["p_value"].to_numpy()).round(4)
    df["verdict"] = np.where((df["delta_auroc"] > 0) & (df["p_adj"] < alpha), "A > B",
                     np.where((df["delta_auroc"] < 0) & (df["p_adj"] < alpha), "B > A",
                              "indistinguishable"))
    df["p_value"] = df["p_value"].round(4)
    return df
