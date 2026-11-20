"""EVE mapping audit + six-predictor shared-subset comparison."""
from __future__ import annotations
import pandas as pd
from variantbenchpy.realdata.clinvar_ingest import parse_clinvar_missense


def run(clinvar_vcf, scores5_parquet, eve_parquet):
    cv = parse_clinvar_missense(clinvar_vcf).drop_duplicates("variant_id").set_index("variant_id")
    S5 = pd.read_parquet(scores5_parquet)
    eve = pd.read_parquet(eve_parquet)
    S = S5.join(eve["EVE"])
    isP = cv["label"] == 1
    s1 = cv["stars"] >= 1
    s2 = cv["stars"] >= 2
    st = eve["EVE_state"].value_counts()

    L = ["EVE MAPPING AUDIT", "=" * 17, ""]
    order = ["EVE_exact_scored", "EVE_exact_score_missing", "EVE_protein_not_covered",
             "no_protein_mapping", "EVE_AA_or_alt_mismatch", "EVE_position_absent"]
    for k in order:
        L.append(f"  {k:<28}{int(st.get(k, 0)):>9,}")
    L += ["",
          "EVE COVERAGE (usable numeric score, of 225,208)",
          f"  overall   {S['EVE'].notna().mean():>7.1%}  ({int(S['EVE'].notna().sum()):,})",
          f"  P/LP      {S['EVE'][isP].notna().mean():>7.1%}",
          f"  B/LB      {S['EVE'][~isP].notna().mean():>7.1%}",
          f"  >=1-star  {S['EVE'][s1].notna().mean():>7.1%}",
          f"  >=2-star  {S['EVE'][s2].notna().mean():>7.1%}"]

    preds5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
    sh5 = S[preds5].notna().all(axis=1)
    sh6 = S[preds5 + ["EVE"]].notna().all(axis=1)
    L += ["",
          "SHARED-SUBSET COMPARISON",
          f"  available-case:        {len(cv):,} variants / {cv['gene'].nunique():,} genes",
          f"  5-predictor shared:    {int(sh5.sum()):,} / {cv.loc[sh5,'gene'].nunique():,} genes",
          f"  6-predictor shared:    {int(sh6.sum()):,} / {cv.loc[sh6,'gene'].nunique():,} genes",
          f"  variants lost adding EVE: {int(sh5.sum()-sh6.sum()):,}",
          f"  genes lost adding EVE:    {cv.loc[sh5,'gene'].nunique()-cv.loc[sh6,'gene'].nunique():,}",
          "",
          "CLASS BALANCE",
          f"  available-case:  P/LP {isP.mean():.1%} / B/LB {(~isP).mean():.1%}",
          f"  6-pred shared:   P/LP {cv.loc[sh6,'label'].mean():.1%} / "
          f"B/LB {(1-cv.loc[sh6,'label'].mean()):.1%}"]

    S.to_parquet("data/benchmark_scores_6pred.parquet")
    return "\n".join(L)


if __name__ == "__main__":
    print(run("data/clinvar.vcf.gz", "data/benchmark_scores_5pred.parquet", "data/eve_scores.parquet"))
    print("\nsaved -> data/benchmark_scores_6pred.parquet")
