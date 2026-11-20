"""Complete-case bias: how much does forcing a shared subset move the results?

Every fair head-to-head restricts to variants all predictors can score. That is the
right thing to do statistically, but it silently discards data and can change both
the scores and the ranking. This module runs the leaderboard twice — on all-available
data and on the strict shared subset — and reports the shift per predictor, so the
cost of fairness is visible rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from variantbenchpy.coverage.analysis import shared_subset_mask
from variantbenchpy.metrics.evaluate import evaluate_all
from variantbenchpy.predictors.base import ScoreMatrix


def complete_case_bias(sm: ScoreMatrix, threshold_mode: str = "optimal") -> pd.DataFrame:
    """Per-predictor comparison of all-available vs strict-shared-subset metrics."""
    all_b = (evaluate_all(sm, threshold_mode=threshold_mode)[["predictor", "rank", "auroc", "n"]]
             .rename(columns={"rank": "rank_all", "auroc": "auroc_all", "n": "n_all"}))
    mask = shared_subset_mask(sm)
    sh = (evaluate_all(sm, mask=mask, threshold_mode=threshold_mode)[["predictor", "rank", "auroc", "n"]]
          .rename(columns={"rank": "rank_shared", "auroc": "auroc_shared", "n": "n_shared"}))
    m = all_b.merge(sh, on="predictor")
    m["delta_auroc"] = (m["auroc_shared"] - m["auroc_all"]).round(4)
    m["rank_change"] = m["rank_all"] - m["rank_shared"]
    m["n_dropped"] = m["n_all"] - m["n_shared"]
    return m.sort_values("rank_all", ignore_index=True)


def complete_case_summary(sm: ScoreMatrix, threshold_mode: str = "optimal") -> dict:
    """Headline numbers: shared-subset size, fraction dropped, whether ranking changed."""
    mask = shared_subset_mask(sm)
    n_labeled = int(sm.labels.notna().sum())
    n_shared = int(mask.sum())
    cc = complete_case_bias(sm, threshold_mode=threshold_mode)
    ranking_changed = bool((cc["rank_change"] != 0).any())
    return {
        "n_labeled": n_labeled,
        "n_shared": n_shared,
        "pct_dropped": round(100 * (n_labeled - n_shared) / n_labeled, 2) if n_labeled else 0.0,
        "ranking_changed": ranking_changed,
        "max_abs_delta_auroc": float(cc["delta_auroc"].abs().max()),
        "table": cc,
    }
