"""STEP 14D — duplicate genomic key audit (within-assay, cross-assay consistency)."""
from __future__ import annotations
import numpy as np
import pandas as pd

PRED6 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2", "EVE"]
KEY = ["chrom", "pos", "ref", "alt"]


def run(path="data/dms_benchmark_14d.parquet", tol=1e-9):
    D = pd.read_parquet(path).copy()
    D["gkey"] = (D["chrom"].astype(str) + ":" + D["pos"].astype(int).astype(str)
                 + ":" + D["ref"].astype(str) + ":" + D["alt"].astype(str))

    n_rows = len(D)
    n_uniq = D["gkey"].nunique()
    within = int(D.duplicated(["assay", "gkey"]).sum())
    per_key = D.groupby("gkey")["assay"].nunique()
    repeated = per_key[per_key > 1]

    incons = {}
    for p in PRED6:
        g = D.dropna(subset=[p]).groupby("gkey")[p]
        spread = g.max() - g.min()
        incons[p] = int((spread > tol).sum())

    aa = D.groupby("gkey")["mutant"].nunique()
    aa_bad = int((aa > 1).sum())

    L = ["STEP 14D — DUPLICATE GENOMIC KEY AUDIT", "=" * 38, "",
         f"Assay-level DMS rows:                 {n_rows:,}",
         f"Unique genomic SNVs:                  {n_uniq:,}",
         f"Repeated assay-level observations:    {n_rows - n_uniq:,}", "",
         f"Within-assay duplicate genomic keys:  {within}", "",
         f"Repeated genomic SNVs across assays:  {len(repeated):,}", "",
         "Predictor consistency across repeated SNVs:"]
    for p in PRED6:
        L.append(f"  {p:<15} inconsistent: {incons[p]}")
    L += ["", f"Same genomic SNV -> different AA mutation: {aa_bad}"]

    if aa_bad:
        bad = aa[aa > 1].head(5).index.tolist()
        L.append("  offending keys: " + ", ".join(bad))

    L += ["", "Examples of legitimate repeated variants:",
          "  " + f"{'genomic SNV':<24}{'AA':<10}{'#assays':>8}  assays"]
    for gk in repeated.sort_values(ascending=False).head(6).index:
        sub = D[D["gkey"] == gk]
        L.append("  " + f"{gk:<24}{sub['mutant'].iloc[0]:<10}"
                 f"{sub['assay'].nunique():>8}  "
                 + ", ".join(sorted(sub["assay"].unique())[:3])[:60])

    ok = (within == 0) and (aa_bad == 0) and all(v == 0 for v in incons.values())
    L += ["", f"STATUS: {'PASS' if ok else 'FAIL'}"]
    if not ok:
        L.append("  NOTE: nonzero counts above must be resolved before freezing 14D.")
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
