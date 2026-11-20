"""STEP 12 — Real-data benchmark reliability audit via the frozen Step 9 framework."""
from __future__ import annotations
import numpy as np
import pandas as pd
from variantbenchpy.predictors.base import ScoreMatrix
from variantbenchpy.evaluation import rank_reliability_audit
from variantbenchpy.realdata.clinvar_ingest import parse_clinvar_missense

PREDS = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2", "EVE"]


def build_scorematrix(clinvar_vcf, scores_parquet):
    cv = parse_clinvar_missense(clinvar_vcf).drop_duplicates("variant_id").set_index("variant_id")
    S = pd.read_parquet(scores_parquet)
    # authoritative clustering = FROZEN Step-10A primary gene (not regenerated here)
    gmap = pd.read_parquet("data/primary_gene_10A.parquet")
    cv = cv.join(gmap["primary_gene_10A"])
    assert cv["primary_gene_10A"].notna().all(), "variant missing frozen 10A primary gene"
    idx = cv.index
    scores = S.reindex(idx)[PREDS].copy()
    # hard pipeline assertions
    assert cv.index.is_unique, "duplicate variant_id"
    assert int(cv["label"].isin([0, 1]).all()) == 1, "non-binary label"
    assert scores.index.equals(idx), "score/label index mismatch"
    table = pd.DataFrame({"variant_id": idx,
                          "label": cv["label"].to_numpy(),
                          "gene": cv["primary_gene_10A"].to_numpy(),
                          "stars": cv["stars"].to_numpy()})
    table["review_status"] = table["stars"].map(lambda s: f"{int(s)}-star")
    sm = ScoreMatrix(table=table, scores=scores, predictors={})
    gg = cv["primary_gene_10A"].reindex(scores.index)
    ev = {"review>=1star": (cv["stars"] >= 1).reindex(scores.index),
          "review>=2star": (cv["stars"] >= 2).reindex(scores.index)}
    return sm, gg, ev


def _layer_sm(clinvar_vcf, scores_parquet, preds):
    import pandas as pd
    cv = parse_clinvar_missense(clinvar_vcf).drop_duplicates("variant_id").set_index("variant_id")
    gmap = pd.read_parquet("data/primary_gene_10A.parquet")
    cv = cv.join(gmap["primary_gene_10A"])
    S = pd.read_parquet(scores_parquet)
    idx = cv.index
    scores = S.reindex(idx)[preds].copy()
    # restrict to the layer's complete-case shared subset
    shared = scores.notna().all(axis=1)
    scores = scores[shared]
    table = pd.DataFrame({"variant_id": scores.index,
                          "label": cv.loc[shared, "label"].to_numpy(),
                          "gene": cv.loc[shared, "primary_gene_10A"].to_numpy(),
                          "stars": cv.loc[shared, "stars"].to_numpy()})
    table["review_status"] = table["stars"].map(lambda s: f"{int(s)}-star")
    sm = ScoreMatrix(table=table, scores=scores, predictors={})
    # composition perturbation = prespecified {primary=all, excl_multigene}.
    # NOTE: gene_groups here defines COMPOSITION levels for the design grid; the
    # gene-clustering unit for weighting/bootstrap is sm.table["gene"] (real gene).
    n_genes_per_var = cv.loc[shared, "n_genes"] if "n_genes" in cv.columns else pd.Series(1, index=scores.index)
    gg = pd.Series(np.where(n_genes_per_var.to_numpy() <= 1, "single_gene", "multi_gene"),
                   index=scores.index)
    ev = {"review>=1star": (cv.loc[shared, "stars"] >= 1).set_axis(scores.index),
          "review>=2star": (cv.loc[shared, "stars"] >= 2).set_axis(scores.index)}
    return sm, gg, ev


def run_split(clinvar_vcf, scores_parquet, n_boot_w=200, n_boot_prob=500, seed=1):
    out = []
    for layer, preds in [("PRIMARY - 5 PREDICTORS", PREDS[:5]),
                         ("SECONDARY - 6 PREDICTORS", PREDS)]:
        sm, gg, ev = _layer_sm(clinvar_vcf, scores_parquet, preds)
        a = rank_reliability_audit(sm, gene_groups=gg, evidence_masks=ev,
                                   n_boot_w=n_boot_w, n_boot_prob=n_boot_prob, seed=seed)
        ngenes = sm.table["gene"].nunique()
        refname = "5pred_shared" if len(preds) == 5 else "6pred_shared"
        bar = "=" * 60
        head = (
            bar + chr(10)
            + f"{layer} ({len(preds)} preds, {sm.scores.shape[0]:,} variants "
              f"/ {ngenes:,} gene clusters)" + chr(10)
            + f"REFERENCE: {refname} / all evidence / pooled_variant / primary composition" + chr(10)
            + f"GRID PROVENANCE: prespecified grid for the {len(preds)}-predictor layer" + chr(10)
            + bar + chr(10)
        )
        out.append(head + a.format())
    return (chr(10) * 2).join(out)


def run(clinvar_vcf, scores_parquet, n_boot_w=200, n_boot_prob=500, seed=1):
    sm, gg, ev = build_scorematrix(clinvar_vcf, scores_parquet)
    audit = rank_reliability_audit(sm, gene_groups=gg, evidence_masks=ev,
                                   n_boot_w=n_boot_w, n_boot_prob=n_boot_prob, seed=seed)
    return audit.format()


if __name__ == "__main__":
    print(run("data/clinvar.vcf.gz", "data/benchmark_scores_6pred.parquet"))
