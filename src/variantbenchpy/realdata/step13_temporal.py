"""STEP 13B — ClinVar temporal provenance for the frozen benchmark cohort.

Source: variant_summary.txt.gz (GRCh38 rows only).
Columns (1-based): 7 ClinicalSignificance, 9 LastEvaluated, 17 Assembly,
19 Chromosome, 31 VariationID, 32 PositionVCF, 33 ReferenceAlleleVCF, 34 AlternateAlleleVCF.
DateLastEvaluated is the date the submitted classification was last evaluated -- it is
NOT necessarily first appearance; treated as 'label_last_evaluated' and documented as such.
"""
from __future__ import annotations
import gzip
import pandas as pd

C = {"clnsig": 6, "lasteval": 8, "assembly": 16, "chrom": 18,
     "varid": 30, "pos": 31, "ref": 32, "alt": 33}  # 0-based


def extract(variant_summary_gz, keys_file, out_parquet):
    keys = set(l.strip() for l in open(keys_file) if l.strip())
    rows = []
    with gzip.open(variant_summary_gz, "rt") as fh:
        fh.readline()  # header
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) <= C["alt"]:
                continue
            if c[C["assembly"]] != "GRCh38":
                continue
            key = f"{c[C['chrom']]}-{c[C['pos']]}-{c[C['ref']]}-{c[C['alt']]}"
            if key not in keys:
                continue
            rows.append({"variant_id": key, "variation_id": c[C["varid"]],
                         "last_evaluated": c[C["lasteval"]],
                         "clnsig_vs": c[C["clnsig"]]})
    df = pd.DataFrame(rows)
    df["label_date"] = pd.to_datetime(df["last_evaluated"], errors="coerce", format="mixed")
    # one row per variant: earliest evaluated date across submissions in the summary
    df = (df.sort_values("label_date")
            .drop_duplicates("variant_id", keep="first")
            .set_index("variant_id"))
    df.to_parquet(out_parquet)
    print(f"matched {len(df):,} / {len(keys):,} benchmark variants")
    print(f"with parseable date: {int(df['label_date'].notna().sum()):,} "
          f"({df['label_date'].notna().mean():.1%})")
    if df["label_date"].notna().any():
        print(f"date range: {df['label_date'].min().date()} -> {df['label_date'].max().date()}")
        print("\nvariants by year (label_last_evaluated):")
        print(df["label_date"].dt.year.value_counts().sort_index().to_string())
    return df


if __name__ == "__main__":
    extract("data/variant_summary.txt.gz", "data/clinvar_keys.txt",
            "data/clinvar_temporal.parquet")
