"""Ranking Fragility & Uncertainty — the focused Step 9 deliverable.

One report that answers not "who ranks first?" but "how stable is that answer?". It:

  1. runs the benchmark across a large battery of *defensible* configurations spanning
     variant-set (all vs shared), weighting (variant vs equal-gene), metric
     (AUROC/AUPRC/MCC) and subset (all genes, gene groups, review levels);
  2. summarises each predictor's rank distribution (median / best / worst);
  3. estimates P(rank #1) by gene-clustered resampling (honest "who is best");
  4. gives an overall STABLE / MODERATE / FRAGILE verdict and counts how often the
     winner changes; and
  5. attributes rank movement to the benchmark *choices* that cause it, by measuring
     the mean rank shift when each design dimension is toggled with the others fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from variantbenchpy.coverage.analysis import shared_subset_mask
from variantbenchpy.evaluation.stability import kendalls_w, probability_best
from variantbenchpy.evaluation.weighting import config_scores
from variantbenchpy.predictors.base import ScoreMatrix

_DIMENSIONS = ["variant_set", "subset", "metric", "weighting"]
_METRIC_WEIGHTINGS = [("auroc", "variant"), ("auroc", "gene"),
                      ("auprc", "variant"), ("auprc", "gene"),
                      ("mcc", "variant")]


def _build_battery(sm, gene_groups=None, subset_masks=None):
    idx = sm.scores.index
    shared = shared_subset_mask(sm)
    variant_sets = {"all": pd.Series(True, index=idx), "shared": shared}

    subsets = {"all-genes": pd.Series(True, index=idx)}
    if gene_groups is not None:
        gg = gene_groups.reindex(idx)
        for grp in sorted(pd.unique(gg.dropna())):
            subsets[f"genes:{grp}"] = (gg == grp)
    if subset_masks:
        for name, m in subset_masks.items():
            subsets[name] = m.reindex(idx).fillna(False)

    configs = []
    for vs_name, vmask in variant_sets.items():
        for su_name, smask in subsets.items():
            for metric, weighting in _METRIC_WEIGHTINGS:
                s = config_scores(sm, vmask & smask, metric, weighting)
                configs.append({"config": f"{vs_name} | {su_name} | {metric}/{weighting}",
                                "variant_set": vs_name, "subset": su_name,
                                "metric": metric, "weighting": weighting, "scores": s})
    return configs


def _attribution(config_df: pd.DataFrame, preds: list[str]) -> pd.Series:
    """Mean rank shift caused by toggling each dimension, others held fixed."""
    result = {}
    for D in _DIMENSIONS:
        others = [d for d in _DIMENSIONS if d != D]
        shifts = []
        for _, grp in config_df.groupby(others):
            if grp[D].nunique() < 2:
                continue
            R = grp[preds].to_numpy(dtype="float64")
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    shifts.append(np.nanmean(np.abs(R[i] - R[j])))
        result[D] = float(np.mean(shifts)) if shifts else 0.0
    return pd.Series(result).sort_values(ascending=False)


@dataclass
class RankingFragilityReport:
    table: pd.DataFrame
    ranks_by_config: pd.DataFrame
    config_dims: pd.DataFrame
    n_configs: int
    stability_label: str
    kendalls_w: float
    top_changed: int
    modal_top: str
    attribution: pd.Series
    prob_valid: int

    def format(self) -> str:
        lines = ["RANKING FRAGILITY & UNCERTAINTY", "=" * 31,
                 f"Configurations tested: {self.n_configs}", ""]
        hdr = f"{'Predictor':<16}{'Median rank':>12}{'Best':>7}{'Worst':>7}{'P(rank #1)':>12}"
        lines.append(hdr)
        for _, r in self.table.iterrows():
            lines.append(f"{r['predictor']:<16}{r['median_rank']:>12.0f}"
                         f"{int(r['best']):>7}{int(r['worst']):>7}{r['prob_best']:>12.2f}")
        lines += ["", f"Overall ranking stability: {self.stability_label} "
                      f"(Kendall's W = {self.kendalls_w:.2f})",
                  f"Top-ranked predictor changed in {self.top_changed}/{self.n_configs} "
                  f"configurations.", "",
                  "Benchmark choices driving rank changes (mean rank shift):"]
        for dim, val in self.attribution.items():
            lines.append(f"  {dim:<14}{val:>6.2f}")
        return "\n".join(lines)


def ranking_fragility_report(sm: ScoreMatrix, gene_groups=None, subset_masks=None,
                             n_boot: int = 1000, seed: int = 0) -> RankingFragilityReport:
    preds = sm.predictor_names
    configs = _build_battery(sm, gene_groups=gene_groups, subset_masks=subset_masks)

    kept = []
    for c in configs:
        s = c["scores"]
        if s.reindex(preds).notna().all():
            row = {k: c[k] for k in ("config", *_DIMENSIONS)}
            ranks = s.reindex(preds).rank(ascending=False, method="min")
            for p in preds:
                row[p] = ranks[p]
            kept.append(row)
    config_df = pd.DataFrame(kept)
    ranks_by_config = config_df.set_index("config")[preds]
    n_configs = len(config_df)

    tbl = pd.DataFrame({
        "predictor": preds,
        "median_rank": [ranks_by_config[p].median() for p in preds],
        "best": [int(ranks_by_config[p].min()) for p in preds],
        "worst": [int(ranks_by_config[p].max()) for p in preds],
    })

    pb, valid = probability_best(sm, metric="auroc", weighting="variant",
                                 n_boot=n_boot, seed=seed)
    tbl = tbl.merge(pb[["predictor", "prob_best"]], on="predictor")
    tbl = tbl.sort_values(["median_rank", "prob_best"],
                          ascending=[True, False], ignore_index=True)

    W = kendalls_w(ranks_by_config)
    label = "STABLE" if W >= 0.7 else ("MODERATE" if W >= 0.4 else "FRAGILE")
    top_each = ranks_by_config.idxmin(axis=1)
    modal_top = top_each.mode().iloc[0]
    top_changed = int((top_each != modal_top).sum())

    attribution = _attribution(config_df, preds)

    return RankingFragilityReport(
        table=tbl, ranks_by_config=ranks_by_config,
        config_dims=config_df[["config", *_DIMENSIONS]],
        n_configs=n_configs, stability_label=label, kendalls_w=float(W),
        top_changed=top_changed, modal_top=modal_top,
        attribution=attribution, prob_valid=valid)
