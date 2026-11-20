"""Threshold-free and threshold-based classification metrics.

All functions accept raw arrays and defend against the two failure modes that make
variant benchmarks silently wrong:
  - missing values (a score or a label is NaN) are dropped pairwise, never imputed;
  - a subset containing only one class yields NaN rather than a bogus 0.5.

Scores are assumed already oriented so that HIGHER = more damaging (the predictor
layer guarantees this), so a variant is called "damaging" when score >= threshold.

The threshold search is fully vectorised (O(n log n)) so metrics stay fast on
ClinVar-scale inputs and inside thousands of bootstrap replicates.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


def _clean(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype="float64")
    s = np.asarray(s, dtype="float64")
    keep = ~(np.isnan(y) | np.isnan(s))
    return y[keep], s[keep]


def auroc(y, s) -> float:
    y, s = _clean(y, s)
    if y.size == 0 or np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def auprc(y, s) -> float:
    """Average precision (area under precision-recall), robust to class imbalance."""
    y, s = _clean(y, s)
    if y.size == 0 or np.unique(y).size < 2:
        return np.nan
    return float(average_precision_score(y, s))


def _sweep(y: np.ndarray, s: np.ndarray):
    """Vectorised confusion-matrix sweep over all thresholds.

    Sorting scores descending and taking cumulative sums gives (tp, fp, fn, tn) at
    every possible cut in one pass. Returns the arrays plus the threshold at each cut.
    """
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    P = y_sorted.sum()
    N = y_sorted.size - P
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    fn = P - tp
    tn = N - fp
    return tp, fp, fn, tn, s_sorted


def optimal_threshold(y, s, metric: str = "mcc") -> float:
    """Score cut-off maximising MCC (default) or F1, computed in O(n log n)."""
    y, s = _clean(y, s)
    if y.size == 0 or np.unique(y).size < 2:
        return np.nan
    tp, fp, fn, tn, s_sorted = _sweep(y, s)
    if metric == "mcc":
        denom = np.sqrt(np.float64(tp + fp) * np.float64(tp + fn) * np.float64(tn + fp) * np.float64(tn + fn))
        with np.errstate(invalid="ignore", divide="ignore"):
            crit = np.where(denom > 0, (tp * tn - fp * fn) / denom, -1.0)
    else:  # f1
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
            rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
            crit = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(s_sorted[int(np.argmax(crit))])


def threshold_metrics(y, s, threshold: float) -> dict[str, float]:
    """MCC, F1, accuracy, sensitivity, specificity at a given oriented threshold."""
    y, s = _clean(y, s)
    if y.size == 0:
        return {k: np.nan for k in ("mcc", "f1", "accuracy", "sensitivity", "specificity")}
    pred = (s >= threshold).astype(int)
    yi = y.astype(int)
    tp = int(((pred == 1) & (yi == 1)).sum())
    tn = int(((pred == 0) & (yi == 0)).sum())
    fp = int(((pred == 1) & (yi == 0)).sum())
    fn = int(((pred == 0) & (yi == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    denom = np.sqrt(np.float64(tp + fp) * np.float64(tp + fn) * np.float64(tn + fp) * np.float64(tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "mcc": float(mcc) if mcc == mcc else np.nan,
        "f1": float(f1),
        "accuracy": float((pred == yi).mean()),
        "sensitivity": float(sens) if sens == sens else np.nan,
        "specificity": float(spec) if spec == spec else np.nan,
    }


def binary_metrics(y, s, threshold: float | None = None,
                   threshold_metric: str = "mcc") -> dict[str, float]:
    """Full metric bundle. If ``threshold`` is None, an optimal one is chosen."""
    yc, sc = _clean(y, s)
    n = int(yc.size)
    n_pos = int((yc == 1).sum())
    n_neg = int((yc == 0).sum())
    thr = optimal_threshold(yc, sc, threshold_metric) if threshold is None else float(threshold)
    out = {"n": n, "n_pos": n_pos, "n_neg": n_neg,
           "auroc": auroc(yc, sc), "auprc": auprc(yc, sc), "threshold": thr}
    out.update(threshold_metrics(yc, sc, thr) if thr == thr else
               {k: np.nan for k in ("mcc", "f1", "accuracy", "sensitivity", "specificity")})
    return out
