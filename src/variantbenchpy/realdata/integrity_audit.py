"""STEP 10 — real data ingestion + integrity audit (no scoring/AUROC yet)."""
from __future__ import annotations
import pandas as pd
from variantbenchpy.realdata.clinvar_ingest import parse_clinvar_missense
from variantbenchpy.realdata import predictors_real as pr


def build(clinvar_vcf, alphamissense=None, revel_zip=None, release="unknown",
          min_gene_variants=1):
    cv = parse_clinvar_missense(clinvar_vcf)
    scores = {}
    if alphamissense:
        scores["AlphaMissense"] = pr.score_alphamissense(cv, alphamissense)
    if revel_zip:
        scores["REVEL"] = pr.score_revel(cv, revel_zip)
    sdf = pd.DataFrame(index=cv["variant_id"])
    for name, s in scores.items():
        sdf[name] = s.reindex(sdf.index)
    cv = cv.set_index("variant_id")

    L = ["STEP 10 — REAL DATA INGESTION", "=" * 29, "",
         f"ClinVar release: {release}", "Genome build: GRCh38", "",
         f"Input labeled missense variants: {len(cv):,}",
         f"Genes: {cv['gene'].nunique():,}", "",
         "Clinical labels:",
         f"  Pathogenic/LP: {(cv.label==1).sum():,}",
         f"  Benign/LB:     {(cv.label==0).sum():,}", "",
         "Review status (stars):"]
    for s, n in cv["stars"].value_counts().sort_index().items():
        L.append(f"  {s}-star: {n:,}")
    L += ["", "Predictor coverage (of labeled missense variants):"]
    for name in scores:
        cov = sdf[name].notna().mean()
        L.append(f"  {name:<14}{cov:6.1%}  ({sdf[name].notna().sum():,})")
    L += ["", f"Mapping failures / missing predictions per predictor shown above.",
          "", "STATUS:", "  Ready for real-world reliability audit "
          "(pending CADD/SIFT/PolyPhen/EVE via dbNSFP)"]
    return cv, sdf, "\n".join(L)


if __name__ == "__main__":
    import sys
    cv, sdf, report = build(sys.argv[1],
                            alphamissense=sys.argv[2] if len(sys.argv) > 2 else None,
                            revel_zip=sys.argv[3] if len(sys.argv) > 3 else None,
                            release=sys.argv[4] if len(sys.argv) > 4 else "2025-08 (GRCh38)")
    print(report)
    cv.join(sdf).to_parquet("data/clinvar_missense_scored.parquet")
    print("\nsaved -> data/clinvar_missense_scored.parquet")
