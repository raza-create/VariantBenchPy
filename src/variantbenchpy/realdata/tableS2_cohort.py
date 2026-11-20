"""Table S2 — frozen ClinVar benchmark cohort integrity (Step 10A)."""
from __future__ import annotations
import os
import pandas as pd

ROWS = [
    ("Source", "ClinVar release", "2026-08-16", ""),
    ("Source", "Genome build", "GRCh38", ""),
    ("Source", "Variant class", "missense_variant (MC field)", ""),
    ("Source", "Label source", "ClinVar germline aggregate classification (CLNSIG)",
     "somatic and oncogenicity assertions excluded from labels"),

    ("Cohort", "Labeled P/LP or B/LB missense variants", "225,208", ""),
    ("Cohort", "Unique genomic keys (chr:pos:ref:alt)", "225,208", "duplicates = 0"),
    ("Cohort", "Unique ClinVar AlleleIDs", "225,208", ""),
    ("Cohort", "Unique ClinVar VariationIDs", "225,208", ""),
    ("Cohort", "Variants mapping to >1 gene", "24,271", "10.8% of cohort"),

    ("Exclusions", "VUS / not provided", "2,139,383", ""),
    ("Exclusions", "Conflicting P/B classifications", "110,991", ""),
    ("Exclusions", "Missing classification", "57,382", ""),
    ("Exclusions", "Other (risk factor / drug response / association)", "2,310", ""),
    ("Exclusions", "Total excluded records", "2,310,066", ""),

    ("Class balance", "Pathogenic / Likely pathogenic", "71,926 (31.9%)", ""),
    ("Class balance", "Benign / Likely benign", "153,282 (68.1%)", ""),

    ("Review status", "0-star", "17,459", ""),
    ("Review status", "1-star", "142,306", ""),
    ("Review status", "2-star", "60,554", ""),
    ("Review status", "3-star", "4,881", ""),
    ("Review status", "4-star", "8", ""),

    ("Gene assignment", "Primary-gene rule", "MANE-informed, gene-name level",
     "ClinVar VCF exposes no MANE transcript field; assignment is by MANE Select GENE NAME, "
     "not variant-level MANE transcript consequence"),
    ("Gene assignment", "MANE-informed assignment", "224,855 (99.8%)", ""),
    ("Gene assignment", "GENEINFO fallback", "353 (0.2%)", ""),
    ("Gene assignment", "Unresolved primary gene", "0", ""),
    ("Gene assignment", "Frozen gene clusters", "15,249",
     "used for all gene-clustered inference in Steps 11-15"),

    ("Somatic / oncogenicity", "ONC (ONCDN) annotated", "259", ""),
    ("Somatic / oncogenicity", "SCI (SCIDN) annotated", "319", ""),
    ("Somatic / oncogenicity", "Both ONC and SCI", "98", ""),
    ("Somatic / oncogenicity", "Unique ONC or SCI", "480",
     "single definition: ONCDN and/or SCIDN; no double counting"),

    ("Gene eligibility", "Genes with >=5 variants", "8,332", ""),
    ("Gene eligibility", "Genes with >=10 variants", "4,763", ""),
    ("Gene eligibility", "Genes with >=20 variants", "2,335", ""),
    ("Gene eligibility", "Genes with both classes", "4,202", ""),
    ("Gene eligibility", "Genes with both classes AND >=10 variants", "2,973",
     "within-gene AUROC eligible population"),

    ("Analysis populations", "Variant-level (available-case)", "225,208 / 15,249 genes", ""),
    ("Analysis populations", "Gene-aware clustered", "225,208 / 15,249 gene clusters", ""),
    ("Analysis populations", "Within-gene AUROC (>=10 var, both classes)", "2,973 genes", ""),
    ("Analysis populations", ">=1-star", "207,749 variants", ""),
    ("Analysis populations", ">=2-star", "65,443 variants", ""),
    ("Analysis populations", "Multi-gene excluded", "200,937 variants", "excludes 24,271"),
    ("Analysis populations", "Somatic/onco excluded", "224,728 variants", "excludes 480"),
    ("Analysis populations", "GENEINFO-fallback excluded", "224,855 variants", "excludes 353"),
]
COLS = ["Section", "Item", "Value", "Note"]


def run():
    df = pd.DataFrame(ROWS, columns=COLS)
    os.makedirs("tables", exist_ok=True)
    path = "tables/TableS2_clinvar_cohort_integrity.csv"
    df.to_csv(path, index=False)

    print("TABLE S2 — GENERATION QC")
    print("=" * 24)
    print(f"rows: {len(df)}   sections: {df['Section'].nunique()}\n")
    for sec in df["Section"].unique():
        sub = df[df["Section"] == sec]
        print(f"{sec} ({len(sub)} rows)")
        for _, r in sub.iterrows():
            print(f"   {r['Item']:<46}{r['Value']}")
        print()
    # internal consistency checks
    checks = [
        ("class balance sums to cohort", 71926 + 153282 == 225208),
        ("review stars sum to cohort", 17459 + 142306 + 60554 + 4881 + 8 == 225208),
        ("gene assignment sums to cohort", 224855 + 353 == 225208),
        ("somatic union consistent (259+319-98)", 259 + 319 - 98 == 480),
        ("exclusions total", 2139383 + 110991 + 57382 + 2310 == 2310066),
    ]
    print("internal consistency checks:")
    for name, ok in checks:
        print(f"   {name:<44}{'PASS' if ok else 'FAIL'}")
    print(f"\nall checks: {'PASS' if all(o for _, o in checks) else 'FAIL'}")
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    run()
