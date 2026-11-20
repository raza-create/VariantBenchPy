"""Bootstrap confidence intervals that respect gene clustering.

Variants are not independent: those in the same gene share evolutionary
conservation, structural context and often a skewed benign/pathogenic ratio.
The ordinary bootstrap, which resamples individual variants, treats them as
independent and therefore *underestimates* variance — producing confidence
intervals that are too narrow and rankings that look more certain than they are.

The cluster (block) bootstrap resamples whole genes with replacement and takes all
their variants together. This is the standard remedy for clustered data (Field &
Welsh 2007; Cameron & Miller 2015) and typically widens intervals substantially.
Both estimators are provided so the widening can be shown rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from variantbenchpy.metrics.classification import auroc

Statistic = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    se: float
    n_boot_valid: int
    method: str

    def __str__(self) -> str:
        return (f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}] "
                f"(SE={self.se:.4f}, {self.method}, {self.n_boot_valid} reps)")


def _clean3(y, s, g):
    y = np.asarray(y, "float64"); s = np.asarray(s, "float64"); g = np.asarray(g)
    keep = ~(np.isnan(y) | np.isnan(s))
    return y[keep], s[keep], g[keep]


def _percentile_ci(reps: np.ndarray, point: float, ci: float, method: str) -> BootstrapCI:
    reps = reps[~np.isnan(reps)]
    if reps.size == 0:
        return BootstrapCI(point, np.nan, np.nan, np.nan, 0, method)
    lo, hi = np.percentile(reps, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    se = float(reps.std(ddof=1)) if reps.size > 1 else np.nan
    return BootstrapCI(float(point), float(lo), float(hi), se, int(reps.size), method)


def iid_bootstrap_ci(y, s, statistic: Statistic = auroc,
                     n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> BootstrapCI:
    """Ordinary bootstrap: resample individual variants (assumes independence)."""
    y, s, _ = _clean3(y, s, np.zeros(len(y)))
    rng = np.random.default_rng(seed)
    n = y.size
    point = statistic(y, s)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[b] = statistic(y[idx], s[idx])
    return _percentile_ci(reps, point, ci, "iid")


def cluster_bootstrap_ci(y, s, groups, statistic: Statistic = auroc,
                         n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> BootstrapCI:
    """Gene-clustered bootstrap: resample whole genes with replacement."""
    y, s, g = _clean3(y, s, groups)
    rng = np.random.default_rng(seed)
    point = statistic(y, s)
    uniq = np.unique(g)
    idx_by = {gr: np.where(g == gr)[0] for gr in uniq}
    reps = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by[c] for c in chosen])
        reps[b] = statistic(y[idx], s[idx])
    return _percentile_ci(reps, point, ci, "gene-clustered")


@dataclass
class PairedComparison:
    """Paired cluster-bootstrap comparison of two predictors (A minus B)."""

    delta_point: float
    delta_lo: float
    delta_hi: float
    prob_A_gt_B: float
    n_boot_valid: int

    def __str__(self) -> str:
        return (f"delta={self.delta_point:+.4f} [{self.delta_lo:+.4f}, {self.delta_hi:+.4f}], "
                f"P(A>B)={self.prob_A_gt_B:.3f} ({self.n_boot_valid} reps)")


def cluster_bootstrap_compare(y, s_a, s_b, groups, statistic: Statistic = auroc,
                              n_boot: int = 1000, ci: float = 0.95,
                              seed: int = 0) -> PairedComparison:
    """Is predictor A really better than B, accounting for gene clustering?

    Both predictors are evaluated on the *same* resampled genes each replicate, so
    the difference is paired and the shared sampling noise cancels. Returns the
    delta CI and the fraction of replicates in which A beats B.
    """
    y = np.asarray(y, "float64"); a = np.asarray(s_a, "float64")
    b = np.asarray(s_b, "float64"); g = np.asarray(groups)
    keep = ~(np.isnan(y) | np.isnan(a) | np.isnan(b))
    y, a, b, g = y[keep], a[keep], b[keep], g[keep]
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    idx_by = {gr: np.where(g == gr)[0] for gr in uniq}
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by[c] for c in chosen])
        deltas[i] = statistic(y[idx], a[idx]) - statistic(y[idx], b[idx])
    deltas = deltas[~np.isnan(deltas)]
    point = statistic(y, a) - statistic(y, b)
    lo, hi = np.percentile(deltas, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return PairedComparison(float(point), float(lo), float(hi),
                            float((deltas > 0).mean()), int(deltas.size))
