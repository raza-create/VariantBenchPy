"""Table S3 — the 36 reliability-audit specifications (5-predictor primary layer).

Exports the SAME grid that rank_reliability_audit() builds, so the per-spec AUROCs
reconcile exactly with Figure 1 (retention 24/36) and Figure S6 (20/36).
Grid = coverage(2) x evaluation_unit(2) x composition(3) x evidence(3) = 36.
The auxiliary data/step12_specgrid.parquet (18 rows/layer) is a DIFFERENT grid and
is not used here.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from variantbenchpy.realdata.step12_reliability import _layer_sm, PREDS
from variantbenchpy.evaluation.reliability import build_design_grid
from variantbenchpy.harmonize.schema import COL_GENE, COL_VARIANT_ID

PREDS5 = PREDS[:5]
OUTCSV = "tables/TableS3_benchmark_specifications.csv"
OUTPQ = "data/step12_reliability_grid_5pred.parquet"
COLS = ["Specification_ID", "Coverage", "Evaluation_unit", "Composition", "Evidence",
        "N_variants", "N_gene_clusters", "N_eligible_genes",
        "AlphaMissense_AUROC", "REVEL_AUROC", "CADD_AUROC", "SIFT_AUROC",
        "PolyPhen2_AUROC", "Winner"]


def run():
    sm, gg, ev = _layer_sm("data/clinvar.vcf.gz", "data/benchmark_scores_6pred.parquet", PREDS5)
    configs = build_design_grid(sm, "auroc", gg, ev)
    genes = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    labels = sm.labels

    rows = []
    for i, c in enumerate(configs, 1):
        mask = c["mask"].reindex(sm.scores.index).fillna(False)
        s = c["scores"].reindex(PREDS5)
        sub_g = genes[mask]
        # eligible genes = genes with both classes present under this mask
        yb = labels[mask]
        elig = int((pd.DataFrame({"y": yb.to_numpy(), "g": sub_g.to_numpy()})
                    .groupby("g")["y"].nunique() == 2).sum())
        rows.append({
            "Specification_ID": f"R{i:02d}",
            "Coverage": c["coverage"], "Evaluation_unit": c["evaluation_unit"],
            "Composition": c["composition"], "Evidence": c["evidence"],
            "N_variants": int(mask.sum()), "N_gene_clusters": int(sub_g.nunique()),
            "N_eligible_genes": elig if c["evaluation_unit"] != "variant" else "NA",
            "AlphaMissense_AUROC": round(float(s["AlphaMissense"]), 5),
            "REVEL_AUROC": round(float(s["REVEL"]), 5),
            "CADD_AUROC": round(float(s["CADD"]), 5),
            "SIFT_AUROC": round(float(s["SIFT"]), 5),
            "PolyPhen2_AUROC": round(float(s["PolyPhen-2"]), 5),
            "Winner": s.idxmax()})
    df = pd.DataFrame(rows)[COLS]
    os.makedirs("tables", exist_ok=True)
    df.to_csv(OUTCSV, index=False)          # "NA" preserved for variant-unit rows
    pq = df.copy()                          # parquet needs a single dtype per column
    pq["N_eligible_genes"] = pd.to_numeric(pq["N_eligible_genes"],
                                           errors="coerce").astype("Int64")
    pq.to_parquet(OUTPQ)

    auc_cols = [c for c in COLS if c.endswith("_AUROC")]
    vals = df[auc_cols].to_numpy(dtype="float64")
    ok = bool(np.all((vals >= 0) & (vals <= 1)))
    wc = df["Winner"].value_counts()

    print("TABLE S3 — GENERATION QC")
    print("=" * 24)
    print(f"rows: {len(df)}")
    print(f"coverage levels: {df['Coverage'].nunique()}  {sorted(df['Coverage'].unique())}")
    print(f"evaluation units: {df['Evaluation_unit'].nunique()}  "
          f"{sorted(df['Evaluation_unit'].unique())}")
    print(f"composition levels: {df['Composition'].nunique()}  "
          f"{sorted(df['Composition'].unique())}")
    print(f"evidence levels: {df['Evidence'].nunique()}  {sorted(df['Evidence'].unique())}")
    print(f"product: {df['Coverage'].nunique()} x {df['Evaluation_unit'].nunique()} x "
          f"{df['Composition'].nunique()} x {df['Evidence'].nunique()} = "
          f"{df['Coverage'].nunique()*df['Evaluation_unit'].nunique()*df['Composition'].nunique()*df['Evidence'].nunique()}")
    print(f"unique specifications: "
          f"{df[['Coverage','Evaluation_unit','Composition','Evidence']].drop_duplicates().shape[0]}")
    print(f"winner counts: {wc.to_dict()}")
    ref = df[(df.Coverage == "available") & (df.Evaluation_unit == "variant")
             & (df.Composition == "all-genes") & (df.Evidence == "all")]
    rw = ref["Winner"].iloc[0] if len(ref) else "NA"
    print(f"reference spec winner: {rw}; retained #1 in "
          f"{int((df['Winner'] == rw).sum())}/{len(df)}  "
          f"(matches audit 24/36: {int((df['Winner'] == rw).sum()) == 24})")
    print(f"AUROC range check: {'PASS' if ok else 'FAIL'}  "
          f"[min {vals.min():.5f}, max {vals.max():.5f}]")
    print(f"\nsaved: {OUTCSV}")
    print(f"       {OUTPQ}")


if __name__ == "__main__":
    run()
