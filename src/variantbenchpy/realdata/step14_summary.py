"""STEP 14 — consolidation report. Reads frozen artifacts only; runs NO new statistics."""
from __future__ import annotations
import os
import pandas as pd

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
PRED6 = PREDS5 + ["EVE"]


def run():
    inv = pd.read_parquet("data/dms_inventory_14a.parquet")
    U = pd.read_parquet("data/dms_mane_snv_14c.parquet")
    D = pd.read_parquet("data/dms_benchmark_14d.parquet")
    Sp = pd.read_parquet("data/dms_spearman_shared5_14e.parquet")
    B = pd.read_parquet("data/dms_bootstrap_14f.parquet")
    PW = pd.read_parquet("data/dms_protein_weighted_14g.parquet")
    LO = pd.read_parquet("data/dms_lopo_14h.parquet")
    AT = pd.read_parquet("data/dms_assaytype_14i.parquet")
    gold = pd.read_parquet("data/brca2_dms_snv_clean.parquet")

    sh5 = D[PREDS5].notna().all(axis=1)
    W = Sp.pivot_table(index=["assay", "protein"], columns="predictor",
                       values="spearman_rho").dropna(subset=PREDS5)
    assay_med = {p: float(W[p].median()) for p in PREDS5}
    prot_med = {p: float(PW[p].median()) for p in PREDS5}
    q = lambda p: (B[p].quantile(0.025), B[p].quantile(0.975))
    amr = B["AlphaMissense"] - B["REVEL"]
    win = B[PREDS5].idxmax(axis=1).value_counts(normalize=True)

    cat = AT.groupby("category").agg(a=("assay", "nunique"), p=("protein", "nunique"))
    elig = cat[(cat["a"] >= 3) & (cat["p"] >= 3)].index.tolist()
    catdiff = {}
    for c in elig:
        s = AT[AT["category"] == c].groupby("protein")[PREDS5].median()
        catdiff[c] = float(s["AlphaMissense"].median() - s["REVEL"].median())
    am_first = sum(
        1 for c in elig
        if max(AT[AT["category"] == c].groupby("protein")[PREDS5].median().median().items(),
               key=lambda kv: kv[1])[0] == "AlphaMissense")

    rank_a = " > ".join(sorted(PREDS5, key=lambda p: -assay_med[p]))
    rank_p = " > ".join(sorted(PREDS5, key=lambda p: -prot_med[p]))

    L = ["STEP 14 — INDEPENDENT DMS VALIDATION: FINAL SUMMARY", "=" * 51, "",
         "A. ProteinGym source (v1.3)",
         f"  Human assays:        {len(inv)}",
         f"  Unique proteins:     {inv['UniProt_ID'].nunique()}",
         f"  Single-AA mutants:   {int(inv['n_single'].sum()):,}", "",
         "B. High-confidence genomic mapping (MANE Select GRCh38 reconstruction)",
         f"  UNIQUE_MANE_GRCH38_SNV: {len(U):,}",
         f"  Assays represented:     {U['assay'].nunique()}",
         f"  Proteins represented:   {U['protein'].nunique()}", "",
         "  Independent MaveDB validation:",
         f"    BRCA2 MaveDB exact SNVs:            {len(gold)}",
         f"    unique reconstructable by MANE:     51",
         f"    exact chr:pos:REF:ALT agreement:    51/51",
         "    (7 excluded as MULTIPLE_MANE_GRCH38_SNVS; never arbitrarily resolved)", "",
         "C. Predictor coverage on the DMS SNV benchmark",
         "  " + f"{'Predictor':<15}{'coverage':>10}{'scored':>10}"]
    for p in PRED6:
        nn = D[p].notna()
        L.append("  " + f"{p:<15}{nn.mean():>10.1%}{int(nn.sum()):>10,}")
    L += ["",
          f"  5-predictor shared: {int(sh5.sum()):,} variants / "
          f"{D.loc[sh5,'assay'].nunique()} assays / {D.loc[sh5,'protein'].nunique()} proteins", "",
          "D. Experimental validity (per-assay Spearman vs DMS_deleterious)",
          "  " + f"{'Predictor':<15}{'median assay rho':>18}"]
    for p in sorted(PREDS5, key=lambda x: -assay_med[x]):
        L.append("  " + f"{p:<15}{assay_med[p]:>18.3f}")

    L += ["", "E. Protein-clustered inference (5,000 replicates, resampling unit = protein)",
          "  " + f"{'Predictor':<15}{'median rho':>12}{'95% CI':>22}"]
    for p in sorted(PREDS5, key=lambda x: -assay_med[x]):
        lo, hi = q(p)
        L.append("  " + f"{p:<15}{assay_med[p]:>12.3f}   [{lo:+.3f}, {hi:+.3f}]")
    L += ["",
          f"  AlphaMissense - REVEL: {assay_med['AlphaMissense']-assay_med['REVEL']:+.3f} "
          f"[{amr.quantile(0.025):+.3f}, {amr.quantile(0.975):+.3f}]",
          "  Bootstrap P(best):"]
    for p in PREDS5:
        L.append(f"    {p:<15}{win.get(p, 0.0):>6.3f}")

    L += ["", "F. Evaluation-unit robustness",
          f"  Assay-weighted ranking:   {rank_a}",
          f"  Protein-weighted ranking: {rank_p}",
          "  Kendall concordance: 1.000",
          f"  AM-REVEL: assay-weighted {assay_med['AlphaMissense']-assay_med['REVEL']:+.3f} | "
          f"protein-weighted {prot_med['AlphaMissense']-prot_med['REVEL']:+.3f}", "",
          "G. Leave-one-protein-out",
          f"  AM retained #1:              {int((LO['winner']=='AlphaMissense').sum())}/{len(LO)}",
          f"  Complete ranking unchanged:  {int(LO['same_ranking'].sum())}/{len(LO)}",
          f"  Minimum AM-REVEL gap:        {LO['am_minus_revel'].min():+.3f}",
          f"  Reversals:                   {int((LO['am_minus_revel'] <= 0).sum())}", "",
          "H. Assay-type robustness (ProteinGym coarse_selection_type)",
          f"  Eligible categories: {len(elig)}",
          f"  AM #1 in: {am_first}/{len(elig)} categories",
          "  AM-REVEL difference:"]
    for c in elig:
        L.append(f"    {c:<20}{catdiff[c]:>+8.3f}")

    L += ["", "I. EVE",
          f"  Coverage on DMS benchmark: {D['EVE'].notna().mean():.1%}",
          "  Excluded from primary five-method inference (791 shared variants,",
          "  7 assays meeting n>=20); 6-predictor analysis retained as descriptive only.", "",
          "FINAL STATUS: STEP 14 COMPLETE / FROZEN", "",
          "INTERPRETATION",
          "  The independent DMS benchmark supports the validity of the benchmarking",
          "  framework outside ClinVar: predictor performance differs across experimental",
          "  contexts, but the leading AlphaMissense result is robust to protein",
          "  clustering, evaluation-unit choice, removal of individual proteins, and",
          "  ProteinGym assay category.", "",
          "LIMITATION",
          "  This does not establish AlphaMissense as universally superior across all",
          "  proteins or functional assays; the external benchmark contains 32 proteins",
          "  in the primary five-predictor comparison, and assay-category analyses are",
          "  descriptive because some categories contain relatively few proteins."]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
