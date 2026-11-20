"""STEP 13D — matched temporal cohorts (common cutoffs, no AUROC)."""
from __future__ import annotations
import pandas as pd

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
PRED6 = PREDS5 + ["EVE"]
COHORTS = [("POST-2021", "2021-10-27", "EVE publication proxy"),
           ("POST-2023 — PRIMARY", "2023-09-22", "AlphaMissense publication proxy")]


def run(path="data/benchmark_with_temporal.parquet"):
    df = pd.read_parquet(path)
    df["first_in_clinvar"] = pd.to_datetime(df["germline_date_created"], errors="coerce")
    gcol = "primary_gene_10A"

    L = ["STEP 13D — MATCHED TEMPORAL COHORTS", "=" * 35, "",
         f"temporal field: germline First-in-ClinVar (validated 20/20)",
         f"provenance available: {int(df['first_in_clinvar'].notna().sum()):,}  "
         f"missing: {int(df['first_in_clinvar'].isna().sum()):,}"]

    for name, cut, basis in COHORTS:
        m = df["first_in_clinvar"] > pd.Timestamp(cut)
        s = df[m]
        sh5 = s[PREDS5].notna().all(axis=1)
        sh6 = s[PRED6].notna().all(axis=1)
        L += ["", "=" * 45, f"{name}   (> {cut}; {basis})", "=" * 45,
              f"  variants:   {len(s):,}",
              f"  genes:      {s[gcol].nunique():,}",
              f"  P/LP %:     {s['label'].mean():.1%}",
              f"  >=1-star %: {(s['stars'] >= 1).mean():.1%}",
              f"  >=2-star %: {(s['stars'] >= 2).mean():.1%}",
              "", "  Coverage:"]
        for p in PRED6:
            L.append(f"    {p:<15}{s[p].notna().mean():>7.1%}  ({int(s[p].notna().sum()):,})")
        L += ["",
              f"  5-predictor shared: {int(sh5.sum()):,} variants / "
              f"{s.loc[sh5, gcol].nunique():,} genes",
              f"  6-predictor shared: {int(sh6.sum()):,} variants / "
              f"{s.loc[sh6, gcol].nunique():,} genes"]

    full5 = df[PREDS5].notna().all(axis=1)
    full6 = df[PRED6].notna().all(axis=1)
    L += ["", "=" * 45, "FULL COHORT REFERENCE (no temporal restriction)", "=" * 45,
          f"  variants: {len(df):,} / genes: {df[gcol].nunique():,} / P/LP {df['label'].mean():.1%}",
          f"  5-predictor shared: {int(full5.sum()):,} / {df.loc[full5, gcol].nunique():,} genes",
          f"  6-predictor shared: {int(full6.sum()):,} / {df.loc[full6, gcol].nunique():,} genes"]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
