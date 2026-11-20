"""Table S4 — DMS assay-level results (frozen Step 14E, 5-predictor shared).

Reads data/dms_spearman_shared5_14e.parquet only; no correlation is recomputed.
EVE excluded; no pooled Spearman anywhere.
"""
from __future__ import annotations
import os
import pandas as pd

SRC = "data/dms_spearman_shared5_14e.parquet"
META = "data/proteingym_v1.3/DMS_substitutions.csv"
PREDS5 = ["AlphaMissense", "REVEL", "SIFT", "CADD", "PolyPhen-2"]
RHO_COLS = {"AlphaMissense": "AlphaMissense_rho", "REVEL": "REVEL_rho",
            "SIFT": "SIFT_rho", "CADD": "CADD_rho", "PolyPhen-2": "PolyPhen2_rho"}
OUT = "tables/TableS4_DMS_assay_results.csv"
COLS = ["ProteinGym_assay", "Protein", "Coarse_selection_type", "N_shared",
        "AlphaMissense_rho", "REVEL_rho", "SIFT_rho", "CADD_rho", "PolyPhen2_rho"]


def run():
    R = pd.read_parquet(SRC)
    W = R.pivot_table(index=["assay", "protein"], columns="predictor",
                      values="spearman_rho")
    N = R.pivot_table(index=["assay", "protein"], columns="predictor", values="n_scored")
    W = W.dropna(subset=PREDS5)
    df = W.reset_index().rename(columns={"assay": "ProteinGym_assay",
                                         "protein": "Protein", **RHO_COLS})
    # N_shared: identical across predictors within the shared subset
    nsh = N.loc[W.index, PREDS5]
    df["N_shared"] = nsh.max(axis=1).to_numpy().astype(int)
    consistent = bool((nsh.nunique(axis=1) == 1).all())

    cat = pd.read_csv(META)[["DMS_id", "coarse_selection_type"]]
    df = df.merge(cat, left_on="ProteinGym_assay", right_on="DMS_id", how="left")
    df["Coarse_selection_type"] = df["coarse_selection_type"].fillna("NA")
    df = df[COLS].sort_values("ProteinGym_assay").reset_index(drop=True)
    for c in RHO_COLS.values():
        df[c] = df[c].round(4)

    os.makedirs("tables", exist_ok=True)
    df.to_csv(OUT, index=False)

    rho_cols = list(RHO_COLS.values())
    miss = int(df[rho_cols].isna().sum().sum())
    print("TABLE S4 — GENERATION QC")
    print("=" * 24)
    print(f"rows: {len(df)}")
    print(f"unique assays: {df['ProteinGym_assay'].nunique()}")
    print(f"unique proteins: {df['Protein'].nunique()}")
    vc = df["Coarse_selection_type"].value_counts()
    print(f"assay categories: {vc.to_dict()}")
    print(f"N_shared range: {df['N_shared'].min():,} – {df['N_shared'].max():,}  "
          f"(total {df['N_shared'].sum():,}; per-predictor n identical within assay: "
          f"{consistent})")
    print("rho range by predictor:")
    for p, c in RHO_COLS.items():
        print(f"   {p:<15}min={df[c].min():+.4f}  median={df[c].median():+.4f}  "
              f"max={df[c].max():+.4f}")
    print(f"missing rho values: {miss}")
    print("EVE excluded; values copied from frozen Step 14E file; no recomputation.")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    run()
