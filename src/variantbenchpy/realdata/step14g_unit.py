"""STEP 14G — DMS evaluation-unit robustness: assay-weighted vs protein-weighted.

Point estimates only. Same frozen 5-predictor shared data. No bootstrap, no EVE.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]


def run(spearman="data/dms_spearman_shared5_14e.parquet",
        bench="data/dms_benchmark_14d.parquet"):
    R = pd.read_parquet(spearman)
    D = pd.read_parquet(bench)
    sh5 = D[PREDS5].notna().all(axis=1)

    W = R.pivot_table(index=["assay", "protein"], columns="predictor",
                      values="spearman_rho").dropna(subset=PREDS5)

    assay_w = {p: float(W[p].median()) for p in PREDS5}
    per_prot = W.groupby(level="protein")[PREDS5].median()
    prot_w = {p: float(per_prot[p].median()) for p in PREDS5}

    rank_a = sorted(PREDS5, key=lambda p: -assay_w[p])
    rank_p = sorted(PREDS5, key=lambda p: -prot_w[p])
    pos_a = {p: i + 1 for i, p in enumerate(rank_a)}
    pos_p = {p: i + 1 for i, p in enumerate(rank_p)}
    tau, _ = kendalltau([pos_a[p] for p in PREDS5], [pos_p[p] for p in PREDS5])

    multi = W.groupby(level="protein").size()
    L = ["STEP 14G — DMS EVALUATION-UNIT ROBUSTNESS", "=" * 41, "",
         "5-predictor shared:",
         f"  variants: {int(sh5.sum()):,}",
         f"  assays:   {len(W)}",
         f"  proteins: {W.index.get_level_values('protein').nunique()}",
         f"  proteins with >1 assay: {int((multi > 1).sum())} "
         f"({', '.join(multi[multi > 1].index.tolist())})", "",
         "  " + f"{'Predictor':<16}{'ASSAY-WEIGHTED':>18}{'PROTEIN-WEIGHTED':>20}",
         "  " + f"{'':<16}{'median rho':>18}{'median rho':>20}"]
    for p in PREDS5:
        L.append("  " + f"{p:<16}{assay_w[p]:>18.3f}{prot_w[p]:>20.3f}")

    L += ["", "Ranking:",
          "  assay-weighted:    " + " > ".join(rank_a),
          "  protein-weighted:  " + " > ".join(rank_p), "",
          f"Winner changed: {'YES' if rank_a[0] != rank_p[0] else 'NO'}", "",
          "Rank changes:"]
    for p in PREDS5:
        d = pos_p[p] - pos_a[p]
        L.append(f"  {p:<16}#{pos_a[p]} -> #{pos_p[p]}"
                 + ("  (unchanged)" if d == 0 else f"  ({d:+d})"))

    L += ["", f"Kendall rank concordance between the two rankings: {tau:.3f}", "",
          "AlphaMissense - REVEL:",
          f"  assay-weighted difference:    {assay_w['AlphaMissense']-assay_w['REVEL']:+.3f}",
          f"  protein-weighted difference:  {prot_w['AlphaMissense']-prot_w['REVEL']:+.3f}"]

    per_prot.to_parquet("data/dms_protein_weighted_14g.parquet")
    L += ["", "-> data/dms_protein_weighted_14g.parquet",
          "(point estimates only; no bootstrap at this step; EVE excluded)"]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
