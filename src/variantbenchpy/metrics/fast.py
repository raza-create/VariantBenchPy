"""Fast AUROC via the Mann-Whitney rank identity, for tight bootstrap loops."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def fast_auroc(y, s) -> float:
    y = np.asarray(y, "float64")
    s = np.asarray(s, "float64")
    keep = ~(np.isnan(y) | np.isnan(s))
    y = y[keep]; s = s[keep]
    n_pos = y.sum()
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
