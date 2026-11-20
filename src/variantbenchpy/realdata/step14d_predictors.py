"""STEP 14D — attach frozen predictors to the MANE-mapped DMS SNV benchmark.

Reuses the exact ClinVar-frozen sources/orientation:
  AlphaMissense (official hg38), REVEL v1.3, dbNSFP 5.3.1a (CADD/SIFT/PolyPhen-2
  via MANE-Select positional transcript rule), EVE (official, protein-keyed).
Orientation: higher = more damaging for every predictor; SIFT = 1 - SIFT_score.
DMS kept in BOTH orientations: DMS_score (higher = fitter) and
DMS_deleterious = -DMS_score (higher = more deleterious).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from variantbenchpy.realdata import predictors_real as pr
from variantbenchpy.realdata.dbnsfp_tabix import extract as dbnsfp_extract

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
PRED6 = PREDS5 + ["EVE"]
RANGES = {"AlphaMissense": (0, 1), "REVEL": (0, 1), "CADD": (0, 99),
          "SIFT": (0, 1), "PolyPhen-2": (0, 1), "EVE": (0, 1)}


def run(dms_parquet="data/dms_mane_snv_14c.parquet",
        am="data/AlphaMissense_hg38.tsv.gz", revel="data/revel.zip",
        bgzf="data/dbnsfp/dbNSFP5.3.1a_grch38.gz",
        eve_scores="data/eve_scores.parquet"):
    D = pd.read_parquet(dms_parquet).copy()
    D["variant_id"] = (D["chrom"].astype(str) + "-" + D["pos"].astype(int).astype(str)
                       + "-" + D["ref"] + "-" + D["alt"])
    D["DMS_deleterious"] = -pd.to_numeric(D["DMS_score"], errors="coerce")

    keys_file = "data/dms_keys.txt"
    with open(keys_file, "w") as fh:
        fh.write("\n".join(sorted(set(D["variant_id"]))))

    # dbNSFP slice for the DMS keys (CADD/SIFT/PolyPhen-2, MANE transcript rule)
    slice_pq = "data/dbnsfp_slice_dms.parquet"
    if not os.path.exists(slice_pq):
        dbnsfp_extract(bgzf, [l.strip() for l in open(keys_file)],
                       slice_pq, "data/dbnsfp_slice_dms.manifest.json")
    S = pd.read_parquet(slice_pq).drop_duplicates("variant_id").set_index("variant_id")

    cv_like = D[["variant_id"]].drop_duplicates()
    am_s = pr.score_alphamissense(cv_like, am)
    rv_s = pr.score_revel(cv_like, revel)

    D = D.set_index("variant_id")
    D["AlphaMissense"] = am_s.reindex(D.index)
    D["REVEL"] = rv_s.reindex(D.index)
    for c in ["CADD", "SIFT", "PolyPhen-2"]:
        D[c] = S[c].reindex(D.index) if c in S.columns else np.nan

    # EVE: protein-keyed; reuse ClinVar EVE table where genomic keys overlap
    if os.path.exists(eve_scores):
        E = pd.read_parquet(eve_scores)
        D["EVE"] = E["EVE"].reindex(D.index) if "EVE" in E.columns else np.nan
    else:
        D["EVE"] = np.nan

    dup = int(D.index.duplicated().sum())
    L = ["STEP 14D — DMS PREDICTOR INTEGRATION", "=" * 36, "",
         f"DMS SNVs: {len(D):,}",
         f"Assays:   {D['assay'].nunique()}",
         f"Proteins: {D['protein'].nunique()}", "",
         "  " + f"{'Predictor':<15}{'scored n':>10}{'coverage':>11}{'assays':>9}{'proteins':>10}"]
    for p in PRED6:
        nn = D[p].notna()
        L.append("  " + f"{p:<15}{int(nn.sum()):>10,}{nn.mean():>11.1%}"
                 f"{D.loc[nn,'assay'].nunique():>9}{D.loc[nn,'protein'].nunique():>10}")

    sh5 = D[PREDS5].notna().all(axis=1)
    sh6 = D[PRED6].notna().all(axis=1)
    L += ["", "5-predictor shared:",
          f"  {int(sh5.sum()):,} variants / {D.loc[sh5,'assay'].nunique()} assays / "
          f"{D.loc[sh5,'protein'].nunique()} proteins",
          "6-predictor shared:",
          f"  {int(sh6.sum()):,} variants / {D.loc[sh6,'assay'].nunique()} assays / "
          f"{D.loc[sh6,'protein'].nunique()} proteins", "",
          "INTEGRITY CHECKS",
          f"  duplicate genomic keys:            {dup}"]
    oob = 0
    for p in PRED6:
        s = pd.to_numeric(D[p], errors="coerce").dropna()
        lo, hi = RANGES[p]
        n = int(((s < lo) | (s > hi)).sum()); oob += n
        L.append(f"  {p:<15} range=[{s.min() if len(s) else float('nan'):.3g},"
                 f"{s.max() if len(s) else float('nan'):.3g}] out_of_range={n}")
    L.append(f"  total out-of-range:                {oob}")

    L += ["", "SCORE DIRECTION CHECK (Spearman sign vs DMS_deleterious; magnitudes NOT reported)"]
    for p in PRED6:
        m = D[p].notna() & D["DMS_deleterious"].notna()
        if m.sum() > 30:
            r = pd.Series(D.loc[m, p]).corr(D.loc[m, "DMS_deleterious"], method="spearman")
            L.append(f"  {p:<15} sign={'+' if r > 0 else '-'}  "
                     f"{'PASS' if r > 0 else 'CHECK'}  (n={int(m.sum()):,})")
        else:
            L.append(f"  {p:<15} insufficient overlap (n={int(m.sum())})")

    L += ["", "PER-ASSAY COVERAGE (5-pred shared fraction):"]
    cov = D.groupby("assay").apply(lambda x: x[PREDS5].notna().all(axis=1).mean())
    for a, v in cov.sort_values(ascending=False).items():
        L.append(f"  {a[:44]:<44}{v:>7.1%}  (n={int((D['assay']==a).sum()):,})")

    D.reset_index().to_parquet("data/dms_benchmark_14d.parquet")
    L += ["", "DMS orientation preserved: DMS_score (higher = fitter) and",
          "  DMS_deleterious = -DMS_score (higher = more deleterious).",
          "", "-> data/dms_benchmark_14d.parquet"]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
