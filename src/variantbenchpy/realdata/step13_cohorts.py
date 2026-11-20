"""STEP 13C — temporal cohort construction (frozen germline First-in-ClinVar)."""
from __future__ import annotations
import pandas as pd
from variantbenchpy.realdata.clinvar_ingest import parse_clinvar_missense
from variantbenchpy.realdata.step13_chronology import CHRONOLOGY

PREDS = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2", "EVE"]


def run(clinvar_vcf="data/clinvar.vcf.gz",
        scores="data/benchmark_scores_6pred.parquet",
        prov="data/clinvar_temporal_provenance.parquet"):
    cv = parse_clinvar_missense(clinvar_vcf).drop_duplicates("variant_id").set_index("variant_id")
    gmap = pd.read_parquet("data/primary_gene_10A.parquet")
    cv = cv.join(gmap["primary_gene_10A"])
    S = pd.read_parquet(scores)
    P = pd.read_parquet(prov).set_index("variant_id")
    df = cv.join(S).join(P[["germline_date_created"]])
    df["first_in_clinvar"] = pd.to_datetime(df["germline_date_created"], errors="coerce")

    L = ["STEP 13C — TEMPORAL COHORT CONSTRUCTION", "=" * 39, "",
         f"Temporal provenance available: {int(df['first_in_clinvar'].notna().sum()):,}",
         f"Missing: {int(df['first_in_clinvar'].isna().sum()):,}", "",
         "  " + f"{'Predictor':<15}{'Cutoff':<13}{'Post n':>9}{'Genes':>8}{'P/LP%':>8}"
         f"{'>=1star%':>10}{'>=2star%':>10}{'Coverage':>10}"]
    for p, meta in CHRONOLOGY.items():
        if not meta["strict_eligible"]:
            continue
        cut = pd.Timestamp(meta["cutoff"])
        m = df["first_in_clinvar"] > cut
        sub = df[m]
        cov = sub[p].notna().mean() if len(sub) else float("nan")
        L.append("  " + f"{p:<15}{meta['cutoff']:<13}{len(sub):>9,}"
                 f"{sub['primary_gene_10A'].nunique():>8,}{sub['label'].mean():>8.1%}"
                 f"{(sub['stars']>=1).mean():>10.1%}{(sub['stars']>=2).mean():>10.1%}{cov:>10.1%}")
    L += ["", "Descriptive only (cutoff unknown, no strict cohort): " +
          ", ".join(k for k, v in CHRONOLOGY.items() if not v["strict_eligible"]),
          "", "Full cohort reference:",
          f"  {len(df):,} variants / {df['primary_gene_10A'].nunique():,} genes / "
          f"P/LP {df['label'].mean():.1%}"]
    df.to_parquet("data/benchmark_with_temporal.parquet")
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
