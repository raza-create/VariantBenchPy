"""STEP 12 — frozen specification grid + per-spec rank matrix (pre-audit view)."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from variantbenchpy.realdata.clinvar_ingest import parse_clinvar_missense
from variantbenchpy.metrics.fast import fast_auroc

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
PRED6 = PREDS5 + ["EVE"]

COVERAGE = ["5pred_shared", "6pred_shared"]
EVIDENCE = ["all", ">=1star", ">=2star"]
UNIT = ["pooled", "equal_gene", "within_gene"]
COMPOSITION = ["primary", "excl_multigene"]


def _load(clinvar_vcf, scores_parquet):
    cv = parse_clinvar_missense(clinvar_vcf).drop_duplicates("variant_id").set_index("variant_id")
    gmap = pd.read_parquet("data/primary_gene_10A.parquet")
    cv = cv.join(gmap["primary_gene_10A"])
    S = pd.read_parquet(scores_parquet)
    df = cv.join(S)
    df["gene"] = df["primary_gene_10A"]
    if "n_genes" not in df:
        df["n_genes"] = 1
    return df


WITHIN_MIN = 10  # frozen eligibility: gene needs >=10 variants AND both classes

def _eligible_genes(y, g, min_n):
    dfg = pd.DataFrame({"y": y, "g": g})
    grp = dfg.groupby("g")["y"]
    ok = (grp.nunique() == 2) & (grp.size() >= min_n)
    return set(ok[ok].index)

def _auroc(unit, y, s, g):
    """pooled: one AUROC on all variants.
    equal_gene: unweighted MEAN of per-gene AUROCs over genes with both classes (>=2 var).
    within_gene: same as equal_gene but restricted to frozen-eligible genes (>=10 var, both classes)."""
    m = ~np.isnan(s)
    y, s, g = y[m], s[m], g[m]
    if np.unique(y).size < 2:
        return np.nan, 0
    if unit == "pooled":
        return fast_auroc(y, s), np.unique(g).size
    min_n = WITHIN_MIN if unit == "within_gene" else 2
    elig = _eligible_genes(y, g, min_n)
    dfg = pd.DataFrame({"y": y, "s": s, "g": g})
    aucs = [fast_auroc(sub["y"].to_numpy(), sub["s"].to_numpy())
            for gg, sub in dfg.groupby("g") if gg in elig]
    return (float(np.mean(aucs)) if aucs else np.nan), len(elig)


def build_specs(df):
    specs = []
    sid = 0
    for cov, ev, unit, comp in itertools.product(COVERAGE, EVIDENCE, UNIT, COMPOSITION):
        preds = PRED6 if cov == "6pred_shared" else PREDS5
        mask = pd.Series(True, index=df.index)
        if cov == "5pred_shared":
            mask &= df[PREDS5].notna().all(axis=1)
        else:
            mask &= df[PRED6].notna().all(axis=1)
        if ev == ">=1star":
            mask &= df["stars"] >= 1
        elif ev == ">=2star":
            mask &= df["stars"] >= 2
        if comp == "excl_multigene":
            mask &= df["n_genes"] <= 1
        sid += 1
        specs.append({"id": f"S{sid:02d}", "coverage": cov, "evidence": ev,
                      "unit": unit, "composition": comp, "preds": preds, "mask": mask})
    return specs


def run(clinvar_vcf, scores_parquet):
    df = _load(clinvar_vcf, scores_parquet)
    specs = build_specs(df)
    rows = []
    for sp in specs:
        sub = df[sp["mask"]]
        y = sub["label"].to_numpy(); g = sub["gene"].to_numpy()
        aucs, elign = {}, 0
        for p in sp["preds"]:
            a, en = _auroc(sp["unit"], y, sub[p].to_numpy(), g)
            aucs[p] = a; elign = max(elign, en)
        rank = pd.Series(aucs).rank(ascending=False, method="min")
        winner = rank.idxmin() if rank.notna().any() else "NA"
        rows.append({**{k: sp[k] for k in ["id", "coverage", "evidence", "unit", "composition"]},
                     "n_var": int(sp["mask"].sum()), "n_gene": sub["gene"].nunique(),
                     "n_elig_gene": elign, "winner": winner,
                     **{f"auc_{p}": aucs.get(p, np.nan) for p in PRED6}})
    R = pd.DataFrame(rows)

    L = ["STEP 12 — REAL-DATA BENCHMARK RELIABILITY (spec grid)", "=" * 52, "",
         f"Input: {len(df):,} variants / {df['gene'].nunique():,} genes / 6 predictors",
         f"N specifications: {len(specs)}  (coverage x evidence x unit x composition)", "",
         "DESIGN SPECIFICATIONS + WINNER"]
    L.append("  " + f"{'ID':<5}{'Coverage':<14}{'Evidence':<9}{'Unit':<13}{'Comp':<15}"
             f"{'nVar':>9}{'nGene':>7}{'eligG':>7}  Winner")
    for _, r in R.iterrows():
        eg = f"{int(r['n_elig_gene']):,}" if r['unit'] != 'pooled' else "-"
        L.append("  " + f"{r['id']:<5}{r['coverage']:<14}{r['evidence']:<9}{r['unit']:<13}"
                 f"{r['composition']:<15}{r['n_var']:>9,}{r['n_gene']:>7,}{eg:>7}  {r['winner']}")

    R5 = R[R.coverage == "5pred_shared"]; R6 = R[R.coverage == "6pred_shared"]
    ref5 = R5[(R5.evidence=="all") & (R5.unit=="pooled") & (R5.composition=="primary")].iloc[0]
    ref6 = R6[(R6.evidence=="all") & (R6.unit=="pooled") & (R6.composition=="primary")].iloc[0]
    L += ["", "WINNER RETENTION (split; reference = all/pooled/primary within each layer)",
          f"  5-predictor PRIMARY: ref winner {ref5['winner']}; retained "
          f"{int((R5['winner']==ref5['winner']).sum())}/{len(R5)} "
          f"({(R5['winner']==ref5['winner']).mean():.0%})",
          f"  6-predictor SECONDARY: ref winner {ref6['winner']}; retained "
          f"{int((R6['winner']==ref6['winner']).sum())}/{len(R6)} "
          f"({(R6['winner']==ref6['winner']).mean():.0%})",
          "",
          "  winner by unit (across all specs):",
          f"    pooled:      {R[R.unit=='pooled']['winner'].value_counts().to_dict()}",
          f"    equal_gene:  {R[R.unit=='equal_gene']['winner'].value_counts().to_dict()}",
          f"    within_gene: {R[R.unit=='within_gene']['winner'].value_counts().to_dict()}",
          "",
          "  UNIT DEFINITIONS (frozen):",
          "    pooled      = single AUROC over all variants",
          "    equal_gene  = unweighted mean of per-gene AUROCs (genes with both classes, >=2 var)",
          f"    within_gene = mean of per-gene AUROCs over eligible genes (>=%d var + both classes)" % WITHIN_MIN]
    R.to_parquet("data/step12_specgrid.parquet")
    return "\n".join(L)


if __name__ == "__main__":
    print(run("data/clinvar.vcf.gz", "data/benchmark_scores_6pred.parquet"))
