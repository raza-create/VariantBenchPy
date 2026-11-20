"""Missingness-bias audit: is a predictor's coverage missing-at-random, or not?

Reporting "EVE covers 78%" is not enough. If the missing 22% are disproportionately
pathogenic (or concentrated in long proteins, or in particular genes), then dropping
them — as every complete-case comparison silently does — biases the metrics. This
module tests, per predictor, whether the *pattern* of missingness is associated with:

  - pathogenicity label   (2x2 chi-square / Fisher, with an odds ratio)
  - protein position       (Mann-Whitney U + point-biserial correlation)
  - gene                   (gene x missing chi-square + per-gene spread)

plus any user-supplied covariate (allele frequency, structure availability, ...).
Per predictor the p-values are Holm-corrected across the covariates tested, and a
missingness that is significantly associated with pathogenicity is flagged as the
most dangerous kind (it directly distorts the labels the metric is computed on).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from variantbenchpy.harmonize.schema import COL_AA_POS, COL_GENE, COL_LABEL, COL_VARIANT_ID
from variantbenchpy.predictors.base import ScoreMatrix


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN-safe)."""
    p = np.asarray(pvals, dtype="float64")
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return adj
    sub = p[idx]
    order = np.argsort(sub)
    m = sub.size
    running = 0.0
    out = np.empty(m)
    for rank, j in enumerate(order):
        running = max(running, (m - rank) * sub[j])
        out[j] = min(running, 1.0)
    adj[idx] = out
    return adj


def _chi2_or_fisher(ct: np.ndarray) -> tuple[float, float, float]:
    """Return (statistic, p_value, odds_ratio) for a 2x2 table.

    Table layout (rows=label 0/1, cols=present/missing):
        [[label0_present, label0_missing],
         [label1_present, label1_missing]]
    OR is odds(missing | pathogenic) / odds(missing | benign).
    """
    a, b = float(ct[0, 0]), float(ct[0, 1])
    c, d = float(ct[1, 0]), float(ct[1, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        orat = (a * d) / (b * c) if (b * c) > 0 else np.inf
    if (ct < 5).any():
        _, p = stats.fisher_exact([[a, b], [c, d]])
        return np.nan, float(p), float(orat)
    stat, p, _, _ = stats.chi2_contingency(np.array([[a, b], [c, d]]))
    return float(stat), float(p), float(orat)


def missingness_audit(
    sm: ScoreMatrix,
    alpha: float = 0.05,
    extra_covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tidy audit table: one row per (predictor, covariate) with an adjusted p-value.

    ``extra_covariates`` is an optional variant_id-indexed frame; numeric columns are
    tested with Mann-Whitney U, non-numeric columns with a chi-square contingency.
    """
    idx = sm.scores.index
    tab = sm.table.set_index(COL_VARIANT_ID).reindex(idx)
    label = tab[COL_LABEL].astype("float64")
    pos = tab[COL_AA_POS].astype("float64")
    gene = tab[COL_GENE].astype("object")

    rows: list[dict] = []
    for name in sm.predictor_names:
        missing = sm.scores[name].isna().to_numpy()

        lm = label.notna().to_numpy()
        y = label.to_numpy()[lm].astype(int)
        ml = missing[lm]
        ct = np.array([
            [np.sum((y == 0) & ~ml), np.sum((y == 0) & ml)],
            [np.sum((y == 1) & ~ml), np.sum((y == 1) & ml)],
        ])
        stat, p, orat = _chi2_or_fisher(ct)
        rows.append({"predictor": name, "covariate": "pathogenicity",
                     "test": "chi2/Fisher 2x2", "statistic": stat, "p_value": p,
                     "effect": f"OR(miss|path)={orat:.2f}",
                     "direction": "pathogenic more missing" if orat > 1 else "benign more missing"})

        pv = pos.to_numpy()
        pm = ~np.isnan(pv)
        a = pv[pm & ~missing]
        b = pv[pm & missing]
        if a.size > 1 and b.size > 1:
            u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
            rpb, _ = stats.pointbiserialr(missing[pm].astype(int), pv[pm])
            rows.append({"predictor": name, "covariate": "protein_position",
                         "test": "Mann-Whitney U", "statistic": float(u), "p_value": float(pu),
                         "effect": f"r_pb={rpb:.3f}",
                         "direction": "missing at higher position" if np.nanmean(b) > np.nanmean(a)
                         else "missing at lower position"})

        g = gene.to_numpy()
        gm = pd.notna(g)
        ct_g = pd.crosstab(pd.Series(g[gm]), pd.Series(missing[gm]))
        if ct_g.shape[1] == 2 and ct_g.shape[0] > 1:
            chi2, pg, _, _ = stats.chi2_contingency(ct_g.values)
            per_gene = ct_g.iloc[:, 1] / ct_g.sum(axis=1)
            mean_m = float(np.nanmean(per_gene))
            cv = float(np.nanstd(per_gene) / mean_m) if mean_m > 0 else np.nan
            rows.append({"predictor": name, "covariate": "gene",
                         "test": "chi2 gene x missing", "statistic": float(chi2),
                         "p_value": float(pg), "effect": f"CV(gene miss)={cv:.2f}",
                         "direction": "varies by gene"})

        if extra_covariates is not None:
            ec = extra_covariates.reindex(idx)
            for col in ec.columns:
                v = ec[col]
                if pd.api.types.is_numeric_dtype(v):
                    vv = v.to_numpy(dtype="float64")
                    cm = ~np.isnan(vv)
                    a = vv[cm & ~missing]; b = vv[cm & missing]
                    if a.size > 1 and b.size > 1:
                        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
                        rpb, _ = stats.pointbiserialr(missing[cm].astype(int), vv[cm])
                        rows.append({"predictor": name, "covariate": col,
                                     "test": "Mann-Whitney U", "statistic": float(u),
                                     "p_value": float(pu), "effect": f"r_pb={rpb:.3f}",
                                     "direction": "missing at higher value" if np.nanmean(b) > np.nanmean(a)
                                     else "missing at lower value"})
                else:
                    ct_c = pd.crosstab(v, pd.Series(missing, index=idx))
                    if ct_c.shape[1] == 2 and ct_c.shape[0] > 1:
                        chi2, pc, _, _ = stats.chi2_contingency(ct_c.values)
                        rows.append({"predictor": name, "covariate": col,
                                     "test": "chi2 contingency", "statistic": float(chi2),
                                     "p_value": float(pc), "effect": "categorical",
                                     "direction": "varies by category"})

    df = pd.DataFrame(rows)
    df["p_adj"] = np.nan
    for _, sub in df.groupby("predictor"):
        df.loc[sub.index, "p_adj"] = _holm(sub["p_value"].to_numpy())
    df["significant"] = df["p_adj"] < alpha
    return df


def missingness_verdict(audit: pd.DataFrame) -> pd.DataFrame:
    """Collapse the audit into one MCAR / informative verdict per predictor."""
    rows = []
    for name, sub in audit.groupby("predictor"):
        sig = sub[sub["significant"]]
        path_sig = ((sub["covariate"] == "pathogenicity") & sub["significant"]).any()
        if path_sig:
            verdict = "INFORMATIVE (missingness tied to pathogenicity)"
        elif len(sig):
            verdict = "biased (missingness tied to " + ", ".join(sig["covariate"]) + ")"
        else:
            verdict = "consistent with MCAR"
        rows.append({"predictor": name, "n_significant": int(len(sig)), "verdict": verdict})
    return pd.DataFrame(rows)
