"""Weighting and metric choices that define one legitimate benchmark configuration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from variantbenchpy.realdata._fast_gene_auroc import gene_macro_auroc

from variantbenchpy.harmonize.schema import COL_GENE, COL_VARIANT_ID
from variantbenchpy.metrics.classification import auprc, binary_metrics
from variantbenchpy.metrics.fast import fast_auroc
from variantbenchpy.predictors.base import ScoreMatrix


def gene_weighted_metric(y, s, genes, metric: str = "auroc", min_n: int = 1
                        ) -> tuple[float, int]:
    """Macro-average of per-gene AUROC: mean over genes with BOTH classes.

    M = (1/G) * sum_g AUROC_g, no further weighting or normalization.
    Hard invariants enforced: every per-gene AUROC in [0,1]; result in [0,1].
    """
    y = np.asarray(y); s = np.asarray(s, dtype="float64"); genes = np.asarray(genes)
    keep = ~np.isnan(s)
    y, s, genes = y[keep], s[keep], genes[keep]
    if y.size == 0:
        return np.nan, 0

    if metric != "auroc":
        vals = []
        for g in pd.unique(genes):
            m = genes == g
            yg, sg = y[m], s[m]
            if yg.size >= min_n and np.unique(yg).size == 2:
                vals.append(auprc(yg, sg))
        vals = np.array(vals, "float64"); vals = vals[~np.isnan(vals)]
        return (float(vals.mean()) if vals.size else np.nan), int(vals.size)

    df = pd.DataFrame({"y": np.asarray(y, dtype="int64"), "s": s, "g": genes})
    # rank of score WITHIN gene (ties -> average), then Mann-Whitney identity
    df["r"] = df.groupby("g", sort=False)["s"].rank(method="average")
    df["rpos"] = df["r"].where(df["y"] == 1, 0.0)
    agg = df.groupby("g", sort=False).agg(
        npos=("y", "sum"), n=("y", "size"), srp=("rpos", "sum"))
    agg["nneg"] = agg["n"] - agg["npos"]
    elig = agg[(agg["npos"] > 0) & (agg["nneg"] > 0) & (agg["n"] >= min_n)]
    if not len(elig):
        return np.nan, 0
    auc = ((elig["srp"] - elig["npos"] * (elig["npos"] + 1) / 2.0)
           / (elig["npos"] * elig["nneg"]))
    auc = auc.to_numpy(dtype="float64")
    if np.any(auc < -1e-9) or np.any(auc > 1 + 1e-9):
        bad = int(((auc < 0) | (auc > 1)).sum())
        raise ValueError(f"gene_weighted_metric: {bad} per-gene AUROC outside [0,1]")
    auc = np.clip(auc, 0.0, 1.0)
    m = float(auc.mean())
    if not (0.0 - 1e-9 <= m <= 1.0 + 1e-9):
        raise ValueError(f"gene_weighted_metric: macro AUROC {m} outside [0,1]")
    return m, int(auc.size)


def config_scores(sm: ScoreMatrix, mask: pd.Series | None = None,
                  metric: str = "auroc", weighting: str = "variant") -> pd.Series:
    """One metric value per predictor for a given (mask, metric, weighting) design."""
    labels = sm.labels
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    idx = sm.scores.index
    base = pd.Series(True, index=idx) if mask is None else mask.reindex(idx).fillna(False)
    out = {}
    for name in sm.predictor_names:
        s = sm.scores[name]
        use = base & s.notna() & labels.notna()
        y = labels[use].to_numpy(); sc = s[use].to_numpy(); g = genes[use].to_numpy()
        if weighting == "gene":
            val, _ = gene_weighted_metric(y, sc, g, metric=metric)
        elif metric == "auroc":
            val = fast_auroc(y, sc)
        elif metric == "auprc":
            val = auprc(y, sc)
        else:
            val = binary_metrics(y, sc)["mcc"]
        out[name] = val
    return pd.Series(out, name=f"{metric}/{weighting}")
