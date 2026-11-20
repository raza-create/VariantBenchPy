"""STEP 14F — protein-clustered bootstrap on the frozen 5-predictor shared DMS subset.

Resampling unit = PROTEIN (36 assays span only 32 proteins; assays of the same
protein are not independent). All assays of a sampled protein enter together.
Per-assay Spearman is invariant under protein resampling, so it is computed once
and the bootstrap re-selects assays; medians and paired differences are computed
per replicate on the SAME resampled assay set (paired by construction).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
PAIRS = [("AlphaMissense", "REVEL"), ("AlphaMissense", "CADD"),
         ("AlphaMissense", "SIFT"), ("AlphaMissense", "PolyPhen-2"),
         ("REVEL", "CADD"), ("REVEL", "SIFT"), ("REVEL", "PolyPhen-2")]
N_BOOT = 5000


def run(spearman="data/dms_spearman_shared5_14e.parquet",
        bench="data/dms_benchmark_14d.parquet", n_boot=N_BOOT, seed=1):
    R = pd.read_parquet(spearman)
    D = pd.read_parquet(bench)
    sh5 = D[PREDS5].notna().all(axis=1)
    S = D[sh5]

    # assay x predictor rho matrix; keep assays scored for ALL five predictors
    W = R.pivot_table(index=["assay", "protein"], columns="predictor",
                      values="spearman_rho")
    W = W.dropna(subset=PREDS5)
    proteins = W.index.get_level_values("protein").to_numpy()
    uniq_prot = np.unique(proteins)
    prot_rows = {p: np.where(proteins == p)[0] for p in uniq_prot}
    M = W[PREDS5].to_numpy(dtype="float64")

    obs = {p: float(np.median(M[:, i])) for i, p in enumerate(PREDS5)}
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(PREDS5)), dtype="float64")
    for b in range(n_boot):
        pick = np.concatenate([prot_rows[p] for p in rng.choice(uniq_prot, uniq_prot.size,
                                                               replace=True)])
        draws[b] = np.median(M[pick], axis=0)

    L = ["STEP 14F — DMS PROTEIN-CLUSTERED INFERENCE", "=" * 42, "",
         "5-predictor shared:",
         f"  variants: {int(sh5.sum()):,}",
         f"  assays:   {len(W)}",
         f"  proteins: {len(uniq_prot)}",
         f"  bootstrap replicates: {n_boot:,}",
         "  resampling unit: PROTEIN (all assays of a sampled protein included)", "",
         "  " + f"{'Predictor':<15}{'median rho':>12}{'95% CI':>22}"]
    for i, p in enumerate(PREDS5):
        lo, hi = np.percentile(draws[:, i], [2.5, 97.5])
        L.append("  " + f"{p:<15}{obs[p]:>12.3f}   [{lo:+.3f}, {hi:+.3f}]")

    L += ["", "Paired differences in median assay rho (same resampled assays):",
          "  " + f"{'Comparison':<30}{'difference':>12}{'95% CI':>22}"]
    idx = {p: i for i, p in enumerate(PREDS5)}
    for a, b in PAIRS:
        d_obs = obs[a] - obs[b]
        d_bs = draws[:, idx[a]] - draws[:, idx[b]]
        lo, hi = np.percentile(d_bs, [2.5, 97.5])
        flag = "" if lo <= 0 <= hi else "  *"
        L.append("  " + f"{a + ' - ' + b:<30}{d_obs:>+12.3f}   [{lo:+.3f}, {hi:+.3f}]{flag}")

    win = np.array(PREDS5)[np.argmax(draws, axis=1)]
    vc = pd.Series(win).value_counts(normalize=True)
    L += ["", "Bootstrap winner probability (highest median assay rho):"]
    for p in PREDS5:
        L.append(f"  {p:<15}{vc.get(p, 0.0):>7.3f}")

    L += ["", "* CI excludes zero.",
          "EVE excluded from this inferential comparison: its 6-predictor shared",
          "  benchmark has only 791 variants / 7 assays meeting n>=20.",
          "No pooled all-variant correlation is used at any point."]
    pd.DataFrame(draws, columns=PREDS5).to_parquet("data/dms_bootstrap_14f.parquet")
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
