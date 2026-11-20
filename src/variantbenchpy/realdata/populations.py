"""Prespecified real-data analysis populations (connects Step 9 framework to ClinVar)."""
from __future__ import annotations
import pandas as pd
from variantbenchpy.realdata.clinvar_ingest import parse_clinvar_missense


def build_populations(clinvar_vcf, scores_parquet):
    cv = parse_clinvar_missense(clinvar_vcf).drop_duplicates("variant_id").set_index("variant_id")
    S = pd.read_parquet(scores_parquet)
    preds = list(S.columns)
    shared = S[preds].notna().all(axis=1)
    L = ["REAL-DATA ANALYSIS POPULATIONS", "=" * 30, "",
         f"Variant-level (available-case): {len(cv):,} variants / {cv['gene'].nunique():,} genes",
         f"Complete shared ({len(preds)}-pred):    {int(shared.sum()):,} / {cv.loc[shared,'gene'].nunique():,} genes",
         f">=1-star:                       {int((cv['stars']>=1).sum()):,}",
         f">=2-star:                       {int((cv['stars']>=2).sum()):,}",
         "",
         "Coverage association with class (missingness signal for Step 7 audit):"]
    isP = cv["label"] == 1
    for p in preds:
        nn = S[p].notna()
        d = nn[isP].mean() - nn[~isP].mean()
        flag = "  <-- imbalance" if abs(d) > 0.03 else ""
        L.append(f"  {p:<14} P/LP={nn[isP].mean():.1%} B/LB={nn[~isP].mean():.1%} delta={d:+.1%}{flag}")
    return cv, S, "\n".join(L)


if __name__ == "__main__":
    _, _, rep = build_populations("data/clinvar.vcf.gz", "data/benchmark_scores_5pred.parquet")
    print(rep)
