"""Evaluate a whole score matrix into a metrics leaderboard (optionally with CIs)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from variantbenchpy.harmonize.schema import COL_GENE, COL_VARIANT_ID
from variantbenchpy.metrics.bootstrap import cluster_bootstrap_ci
from variantbenchpy.metrics.classification import auroc, binary_metrics
from variantbenchpy.predictors.base import ScoreMatrix


def evaluate_all(
    sm: ScoreMatrix,
    mask: pd.Series | None = None,
    threshold_mode: str = "optimal",
    bootstrap: bool = False,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Metrics for every predictor, ranked by AUROC."""
    labels = sm.labels
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    idx = sm.scores.index
    base = pd.Series(True, index=idx) if mask is None else mask.reindex(idx).fillna(False)

    rows = []
    for name, predictor in sm.predictors.items():
        s = sm.scores[name]
        use = base & s.notna() & labels.notna()
        y = labels[use].to_numpy()
        sc = s[use].to_numpy()
        if threshold_mode == "predictor":
            thr = predictor.oriented_threshold
        elif threshold_mode == "optimal":
            thr = None
        else:
            thr = float(threshold_mode)
        m = binary_metrics(y, sc, threshold=thr)
        row = {"predictor": name, **m}
        if bootstrap:
            g = genes[use].to_numpy()
            ci = cluster_bootstrap_ci(y, sc, g, statistic=auroc, n_boot=n_boot, seed=seed)
            row["auroc_lo"] = ci.lo
            row["auroc_hi"] = ci.hi
            row["auroc_se"] = ci.se
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("auroc", ascending=False, ignore_index=True)
    df.insert(1, "rank", np.arange(1, len(df) + 1))
    return df
