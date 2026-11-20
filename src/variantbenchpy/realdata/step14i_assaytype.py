"""STEP 14I — DMS assay-type robustness using ProteinGym's own coarse_selection_type.

Categories come from the ProteinGym reference file only; no manual relabelling.
Protein-weighted estimand within each category. No bootstrap, no EVE.
"""
from __future__ import annotations
import pandas as pd

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
MIN_ASSAYS, MIN_PROTEINS = 3, 3


def run(spearman="data/dms_spearman_shared5_14e.parquet",
        base="data/proteingym_v1.3"):
    R = pd.read_parquet(spearman)
    W = R.pivot_table(index=["assay", "protein"], columns="predictor",
                      values="spearman_rho").dropna(subset=PREDS5).reset_index()

    meta = pd.read_csv(f"{base}/DMS_substitutions.csv")[
        ["DMS_id", "coarse_selection_type", "selection_assay"]]
    W = W.merge(meta, left_on="assay", right_on="DMS_id", how="left")
    W["category"] = W["coarse_selection_type"].fillna("UNSPECIFIED")

    cat = W.groupby("category").agg(assays=("assay", "nunique"),
                                    proteins=("protein", "nunique"))
    eligible = cat[(cat["assays"] >= MIN_ASSAYS) & (cat["proteins"] >= MIN_PROTEINS)]

    L = ["STEP 14I — DMS ASSAY-TYPE ROBUSTNESS", "=" * 36, "",
         "5-predictor shared:",
         f"  assays: {W['assay'].nunique()}",
         f"  proteins: {W['protein'].nunique()}", "",
         "ProteinGym assay categories (coarse_selection_type):",
         "  " + f"{'Category':<24}{'assays':>8}{'proteins':>10}"]
    for c, r in cat.sort_values("assays", ascending=False).iterrows():
        L.append("  " + f"{c:<24}{int(r['assays']):>8}{int(r['proteins']):>10}")
    L += ["", f"Categories eligible (>={MIN_ASSAYS} assays AND >={MIN_PROTEINS} proteins): "
          f"{len(eligible)}  ({', '.join(eligible.index.tolist())})", "",
          "Per-category protein-weighted median rho:"]

    winners, diffs = {}, []
    for c in eligible.index:
        sub = W[W["category"] == c]
        per_prot = sub.groupby("protein")[PREDS5].median()
        med = {p: float(per_prot[p].median()) for p in PREDS5}
        rank = sorted(PREDS5, key=lambda p: -med[p])
        winners[c] = rank[0]
        diffs.append({"category": c, "diff": med["AlphaMissense"] - med["REVEL"],
                      "assays": int(eligible.loc[c, "assays"]),
                      "proteins": int(eligible.loc[c, "proteins"])})
        L += ["", f"Category: {c}  ({int(eligible.loc[c,'assays'])} assays / "
              f"{int(eligible.loc[c,'proteins'])} proteins)"]
        for p in PREDS5:
            L.append(f"  {p:<15}{med[p]:>8.3f}")
        L.append("  Ranking: " + " > ".join(rank))

    wc = pd.Series(list(winners.values())).value_counts()
    L += ["", "Winner by eligible category:"]
    for p in PREDS5:
        L.append(f"  {p:<15}{int(wc.get(p, 0)):>4} / {len(eligible)}")

    Dd = pd.DataFrame(diffs)
    L += ["", "AlphaMissense - REVEL:",
          "  " + f"{'Category':<24}{'difference':>12}"]
    for _, r in Dd.iterrows():
        L.append("  " + f"{r['category']:<24}{r['diff']:>+12.3f}")
    neg = Dd[Dd["diff"] <= 0]
    L += ["", f"Categories where AM-REVEL <= 0: {len(neg)}"
          + (f"  ({', '.join(neg['category'])})" if len(neg) else "")]

    ok = int(wc.get("AlphaMissense", 0)) == len(eligible)
    L += ["", f"STATUS: {'PASS' if ok else 'MIXED'}",
          "  (PASS = AlphaMissense ranks #1 in every eligible ProteinGym assay category)",
          "", "Categories from ProteinGym coarse_selection_type; no manual labels.",
          "No bootstrap at this step; EVE excluded."]
    W.to_parquet("data/dms_assaytype_14i.parquet")
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
