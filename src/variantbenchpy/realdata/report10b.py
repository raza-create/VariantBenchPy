"""STEP 10B — full predictor integration & score-integrity report (PASS criteria A-E)."""
from __future__ import annotations
import subprocess
import numpy as np
import pandas as pd
from variantbenchpy.realdata.clinvar_ingest import parse_clinvar_missense

PROVENANCE = {
    "ClinVar": "GRCh38 release 2026-08-16",
    "AlphaMissense": "official hg38 (dm_alphamissense, 2023)",
    "REVEL": "v1.3 (grch38)",
    "dbNSFP": "5.3.1a academic, GRCh38 BGZF",
    "CADD": "dbNSFP5.3.1a CADD_phred",
    "SIFT": "dbNSFP5.3.1a SIFT_score (oriented 1-SIFT), MANE-Select transcript",
    "PolyPhen-2": "dbNSFP5.3.1a Polyphen2_HDIV_score, MANE-Select transcript",
    "EVE": "official EVE resource (separate; NOT popEVE) — pending",
    "transcript_rule": "MANE Select positional index -> first-numeric fallback; never max/min",
}


def unmatched_audit(cv, matched_keys, bgzf):
    un = cv[~cv["variant_id"].isin(matched_keys)]
    cats = {"absent_position": 0, "allele_mismatch": 0, "other": 0}
    per = {1: dict(cats), 0: dict(cats)}
    for _, r in un.iterrows():
        q = subprocess.run(["tabix", bgzf, f"{r.chrom}:{r.pos}-{r.pos}"],
                           capture_output=True, text=True).stdout
        if not q.strip():
            k = "absent_position"
        else:
            alleles = {(l.split("\t")[2], l.split("\t")[3]) for l in q.splitlines()}
            k = "other" if (r.ref, r.alt) in alleles else "allele_mismatch"
        cats[k] += 1
        per[int(r.label)][k] += 1
    return len(un), cats, per


def build(clinvar_vcf, slice_parquet, am_series, revel_series, bgzf):
    cv = parse_clinvar_missense(clinvar_vcf)
    cv = cv.drop_duplicates("variant_id").set_index("variant_id")
    sl = pd.read_parquet(slice_parquet).drop_duplicates("variant_id").set_index("variant_id")

    S = pd.DataFrame(index=cv.index)
    S["AlphaMissense"] = am_series.reindex(cv.index)
    S["REVEL"] = revel_series.reindex(cv.index)
    S["CADD"] = sl["CADD"].reindex(cv.index)
    S["SIFT"] = sl["SIFT"].reindex(cv.index)
    S["PolyPhen-2"] = sl["PolyPhen-2"].reindex(cv.index)
    preds = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]
    isP = cv["label"] == 1

    L = ["STEP 10B — PREDICTOR INTEGRATION & SCORE INTEGRITY", "=" * 51, "",
         f"ClinVar variants: {len(cv):,}", ""]

    # A. matching
    matched = set(sl.index)
    n_un, cats, per = unmatched_audit(cv.reset_index(), matched, bgzf)
    L += ["A. VARIANT MATCHING",
          f"  dbNSFP exact genomic matches: {len(matched & set(cv.index)):,} "
          f"({len(matched & set(cv.index))/len(cv):.2%})",
          f"  Unmatched: {n_un:,}",
          f"    absent from dbNSFP position: {cats['absent_position']:,}",
          f"    position present, allele mismatch: {cats['allele_mismatch']:,}",
          f"    other: {cats['other']:,}",
          f"    (P/LP absent={per[1]['absent_position']:,} allele_mm={per[1]['allele_mismatch']:,} | "
          f"B/LB absent={per[0]['absent_position']:,} allele_mm={per[0]['allele_mismatch']:,})", ""]

    # B. transcript resolution (from slice reasons)
    L += ["B. TRANSCRIPT RESOLUTION (SIFT | PolyPhen-2)"]
    for cat in ["MANE_Select", "MANE_index_no_score->fallback", "no_MANE->fallback", "no_usable_score"]:
        s_n = int((sl["SIFT_reason"] == cat).sum()); p_n = int((sl["PolyPhen2_reason"] == cat).sum())
        L.append(f"    {cat:<32}{s_n:>8,} | {p_n:>8,}")
    L.append("    rule: MANE-Select positional index; NO arbitrary first-transcript")

    # C. score integrity
    L += ["", "C. SCORE INTEGRITY"]
    ranges = {"AlphaMissense": (0, 1), "REVEL": (0, 1), "CADD": (0, 99),
              "SIFT": (0, 1), "PolyPhen-2": (0, 1)}
    invalid = 0
    for p in preds:
        s = pd.to_numeric(S[p], errors="coerce").dropna()
        lo, hi = ranges[p]
        oob = int(((s < lo) | (s > hi)).sum())
        invalid += oob
        mp, mb = S[p][isP].mean(), S[p][~isP].mean()
        L.append(f"    {p:<14} range[{s.min():.3g},{s.max():.3g}] out_of_bounds={oob} "
                 f"dir={'PASS' if mp > mb else 'CHECK'} (P={mp:.3f} B={mb:.3f})")
    dup = int(S.index.duplicated().sum())
    L += [f"    invalid/out-of-range total: {invalid}",
          f"    duplicate variant keys: {dup}"]

    # D. missingness / coverage
    L += ["", "D. COVERAGE (overall | P/LP | B/LB | >=1star | >=2star)"]
    s1 = cv["stars"] >= 1; s2 = cv["stars"] >= 2
    for p in preds:
        nn = S[p].notna()
        L.append(f"    {p:<14}{nn.mean():>7.1%}{nn[isP].mean():>8.1%}{nn[~isP].mean():>8.1%}"
                 f"{nn[s1].mean():>9.1%}{nn[s2].mean():>9.1%}")
    # gene-stratified median coverage
    L += ["  gene-stratified median coverage:"]
    for p in preds:
        g = S[p].notna().groupby(cv["gene"]).mean()
        L.append(f"    {p:<14} median_gene_cov={g.median():.1%} genes<50%={int((g<0.5).sum()):,}")

    # shared subset
    shared = S[preds].notna().all(axis=1)
    L += ["", "SHARED 5-PREDICTOR SET (AlphaMissense+REVEL+CADD+SIFT+PolyPhen-2)",
          f"    variants: {int(shared.sum()):,} ({shared.mean():.1%})",
          f"    genes: {cv.loc[shared, 'gene'].nunique():,}",
          "    (EVE pending -> becomes 6-predictor set when added)"]

    # E. provenance
    L += ["", "E. PROVENANCE (frozen)"]
    for k, v in PROVENANCE.items():
        L.append(f"    {k}: {v}")

    warn = invalid > 0 or dup > 0 or any("CHECK" in x for x in L)
    L += ["", f"STATUS: {'PASS WITH WARNINGS' if warn else 'PASS'} "
          "(EVE integration pending before final 6-predictor freeze)"]

    S.to_parquet("data/benchmark_scores_5pred.parquet")
    return "\n".join(L)
