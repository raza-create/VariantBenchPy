"""STEP 14E — per-assay Spearman between predictors and DMS_deleterious.

Assay-aware by design: DMS scales differ across assays, so pooling all rows into a
single Spearman is NOT the primary estimand. No global ranking, no bootstrap here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
PRED6 = PREDS5 + ["EVE"]
MIN_N = 20


def per_assay(D, preds, label):
    rows, excluded = [], set()
    for (assay, protein), sub in D.groupby(["assay", "protein"]):
        for p in preds:
            m = sub[p].notna() & sub["DMS_deleterious"].notna()
            n = int(m.sum())
            if n < MIN_N:
                excluded.add((assay, p))
                continue
            rho, pv = spearmanr(sub.loc[m, p], sub.loc[m, "DMS_deleterious"])
            rows.append({"assay": assay, "protein": protein, "predictor": p,
                         "n_scored": n, "spearman_rho": rho, "p_value": pv})
    R = pd.DataFrame(rows)
    L = [f"{label}:", "",
         "  " + f"{'Predictor':<15}{'assays':>8}{'median_rho':>12}{'Q1':>8}{'Q3':>8}"
         f"{'positive_assays':>17}"]
    for p in preds:
        s = R[R["predictor"] == p]["spearman_rho"]
        if len(s) == 0:
            L.append("  " + f"{p:<15}{0:>8}{'-':>12}{'-':>8}{'-':>8}{'-':>17}")
            continue
        L.append("  " + f"{p:<15}{len(s):>8}{s.median():>12.3f}{s.quantile(.25):>8.3f}"
                 f"{s.quantile(.75):>8.3f}{(s>0).sum():>10}/{len(s):<6}")
    return R, "\n".join(L), excluded


def run(path="data/dms_benchmark_14d.parquet"):
    D = pd.read_parquet(path)
    D["DMS_deleterious"] = pd.to_numeric(D["DMS_deleterious"], errors="coerce")
    sh5 = D[PREDS5].notna().all(axis=1)
    sh6 = D[PRED6].notna().all(axis=1)

    R_av, t_av, ex_av = per_assay(D, PRED6, "Available-case")
    R_5, t_5, _ = per_assay(D[sh5], PREDS5, "5-predictor shared subset")
    R_6, t_6, _ = per_assay(D[sh6], PRED6, "6-predictor shared subset")

    L = ["STEP 14E — DMS PER-ASSAY SPEARMAN", "=" * 33, "",
         f"benchmark: {len(D):,} rows / {D['assay'].nunique()} assays / "
         f"{D['protein'].nunique()} proteins",
         f"minimum assay size: n >= {MIN_N} scored variants", "",
         t_av, "",
         f"  assay x predictor combinations excluded by n<{MIN_N}: {len(ex_av)}", "",
         t_5, "",
         f"  5-pred shared rows: {int(sh5.sum()):,} / assays {D.loc[sh5,'assay'].nunique()}", "",
         t_6, "",
         f"  6-pred shared rows: {int(sh6.sum()):,} / assays {D.loc[sh6,'assay'].nunique()}", ""]

    L += ["Per-assay rho range (available-case):"]
    for p in PRED6:
        s = R_av[R_av["predictor"] == p]["spearman_rho"]
        if len(s):
            L.append(f"  {p:<15} min={s.min():+.3f}  max={s.max():+.3f}  n_assays={len(s)}")

    neg = R_av[R_av["spearman_rho"] < 0].sort_values("spearman_rho")
    L += ["", f"Assays with negative correlations (available-case): {len(neg)}"]
    if len(neg):
        L.append("  " + f"{'assay':<44}{'predictor':<15}{'rho':>8}{'n':>8}")
        for _, r in neg.head(15).iterrows():
            L.append("  " + f"{r['assay'][:44]:<44}{r['predictor']:<15}"
                     f"{r['spearman_rho']:>8.3f}{r['n_scored']:>8,}")

    R_av.to_parquet("data/dms_spearman_available_14e.parquet")
    R_5.to_parquet("data/dms_spearman_shared5_14e.parquet")
    R_6.to_parquet("data/dms_spearman_shared6_14e.parquet")
    L += ["", "-> data/dms_spearman_{available,shared5,shared6}_14e.parquet",
          "NOTE: assay-aware estimand; no pooled Spearman, no global ranking,",
          "  no bootstrap inference at this step."]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
