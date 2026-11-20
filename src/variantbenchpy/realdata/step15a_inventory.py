"""STEP 15A — master results inventory. Frozen numbers only; no new analysis."""
from __future__ import annotations

INVENTORY = [
    dict(
        n="RESULT 1",
        q="How different are sampling uncertainty under a fixed benchmark and "
          "benchmark-design uncertainty across defensible specifications?",
        pop="ClinVar GRCh38 missense, 5-predictor shared: 187,155 variants / 13,599 gene "
            "clusters (Step 12 primary layer)",
        est="Bootstrap P(#1) under a fixed reference design vs. proportion of prespecified "
            "design configurations in which the reference winner is #1",
        nums=[
            "REVEL bootstrap P(#1) = 1.00 under the fixed reference design",
            "REVEL retained #1 in only 24/36 configurations (retention 0.67)",
            "Kendall's W = 0.91 [95% CI 0.89, 0.93] — whole-ranking concordance is high",
            "Rank distribution: REVEL and AlphaMissense occupy only ranks 1-2; CADD always #3",
            "Design sensitivity share: evaluation_unit 74%, evidence 21%, composition 5%",
            "Selection-aware inference: no supported relative-performance change "
            "under any prespecified perturbation",
        ],
        concl="Sampling uncertainty and benchmark-design uncertainty are distinct: the "
              "winner appears certain under one fixed design (P(#1)=1.00) yet changes "
              "across a third of defensible designs, with instability concentrated at "
              "the top of an otherwise highly concordant ranking.",
    ),
    dict(
        n="RESULT 2",
        q="Does requiring complete predictor coverage change the benchmark population and "
          "apparent performance?",
        pop="ClinVar: 5-predictor shared (187,155 variants / 13,599 genes) -> "
            "EVE-constrained 6-predictor shared (53,552 variants / 1,981 genes); "
            "frozen Step-10A MANE-informed gene mapping throughout",
        est="Matched ΔAUROC (paired gene-clustered bootstrap) and pairwise ΔΔAUROC between "
            "the two nested populations",
        nums=[
            "Population: 187,155 -> 53,552 variants; 13,599 -> 1,981 genes "
            "(1,981 of 15,249 frozen ClinVar gene clusters, 13.0%)",
            "P/LP prevalence 31.9% -> 56.9%; effective genes N_eff = 282 of 1,981 nominal",
            "ΔAUROC all five predictors negative: CADD -0.041 [-0.054, -0.029], "
            "PolyPhen-2 -0.036, SIFT -0.028, AlphaMissense -0.015, REVEL -0.013",
            "AlphaMissense-CADD ΔΔAUROC +0.026 [+0.017, +0.035], Holm p<0.001",
            "EVE availability ~ label: OR 4.21 [4.12, 4.29]; within-gene control "
            "OR 2.16 [1.94, 2.40] (cluster-robust)",
            "Effect attenuates at higher evidence: CADD ΔAUROC -0.059 (1-star) vs "
            "-0.012 (>=2-star), Holm p<0.001",
            "Stable under exclusion of multi-gene, GENEINFO-fallback, somatic/onco variants",
        ],
        concl="Complete-case restriction materially changes benchmark composition and "
              "lowers estimated AUROC for every comparator, altering relative performance "
              "gaps while leaving the ordinal ranking intact.",
    ),
    dict(
        n="RESULT 3",
        q="Are predictor rankings robust to defensible benchmark-design choices?",
        pop="Step 12: 5-predictor primary (187,155 / 13,599 clusters) and 6-predictor "
            "secondary (53,552 / 1,981 clusters), 36 prespecified specifications each",
        est="Winner retention, Kendall's W with gene-clustered CI, rank distribution, "
            "descriptive design-sensitivity shares, selection-aware ΔΔAUROC",
        nums=[
            "5-predictor: winner retention 0.67; W = 0.91 [0.89, 0.93]",
            "6-predictor: winner retention 0.56; W = 0.94 [0.89, 0.95]",
            "Winner by evaluation unit: pooled -> REVEL (12/12); macro-gene -> "
            "AlphaMissense (12/12)",
            "EVE occupies rank 3 in 89% of 6-predictor configurations; PolyPhen-2 "
            "rank 6 in 94%",
            "Objective sensitivity: AUROC/AUPRC/MCC rank-correlation 1.00, same #1",
            "Selection-aware: 6-pred composition numerical reversal ΔΔAUROC +0.007 "
            "[-0.024, +0.038], p_adj 0.258 -> NOT supported",
        ],
        concl="The benchmark is not globally fragile — whole-ranking concordance is high — "
              "but the identity of the top method is design-dependent, driven mainly by "
              "the evaluation unit (pooled-variant vs gene-macro).",
    ),
    dict(
        n="RESULT 4",
        q="Does temporal exposure-resistant evaluation alter conclusions?",
        pop="Temporal exposure-resistant subset: germline First-in-ClinVar after a "
            "predictor publication-date PROXY (2023-09-22 = AlphaMissense publication "
            "date, NOT a verified training cutoff); post-2023 70,661 / 11,768 genes "
            "vs full 5-pred shared 187,155 / 13,599 genes",
        est="Matched ΔAUROC and pairwise ΔΔAUROC, paired gene-clustered bootstrap on "
            "nested populations",
        nums=[
            "Temporal field validated 20/20 against ClinVar web 'First in ClinVar'",
            "All five predictors improve post-2023: PolyPhen-2 +0.030 [+0.027, +0.032], "
            "SIFT +0.023, CADD +0.021, REVEL +0.011, AlphaMissense +0.010",
            "AlphaMissense-CADD ΔΔAUROC -0.012 [-0.014, -0.010], Holm p<0.001",
            "REVEL vs AlphaMissense ΔΔAUROC +0.001 [-0.001, +0.003], p=0.282 (unchanged)",
            "Composition standardization retains most of the effect "
            "(e.g. CADD +0.021 -> +0.011)",
            "Post-2021 landmark agrees in direction for all five predictors",
        ],
        concl="Under temporal exposure-resistant evaluation (never described as "
              "leakage-free or unexposed), restricting to variants first public after "
              "the predictor publication proxy does not "
              "degrade measured performance — it improves it for all five methods — and "
              "leaves the leading predictor unchanged, though relative gaps shift.",
    ),
    dict(
        n="RESULT 5",
        q="Do the findings generalize to independent experimental DMS data?",
        pop="ProteinGym v1.3 human assays mapped to GRCh38 by MANE Select reconstruction; "
            "5-predictor shared 37,197 variants / 36 assays / 32 proteins",
        est="Per-assay Spearman vs DMS_deleterious; protein-clustered bootstrap of the "
            "median assay rho (5,000 replicates)",
        nums=[
            "Mapper validated against MaveDB: BRCA2 51/51 exact chr:pos:REF:ALT agreement",
            "Median assay rho: AlphaMissense 0.519, REVEL 0.440, SIFT 0.409, CADD 0.391, "
            "PolyPhen-2 0.349",
            "AlphaMissense - REVEL +0.079 [+0.006, +0.142]; bootstrap P(best) 0.982",
            "Assay- vs protein-weighted rankings identical (Kendall concordance 1.000)",
            "Leave-one-protein-out: AlphaMissense #1 in 32/32; minimum gap +0.034",
            "AlphaMissense ranks #1 in 5/5 ProteinGym assay categories",
            "EVE coverage only 2.1% -> excluded from primary inference",
        ],
        concl="On an independent experimental benchmark the framework reproduces a stable "
              "leader: AlphaMissense ranks first and the result survives protein "
              "clustering, evaluation-unit choice, protein removal, and assay category.",
    ),
]


def run():
    L = ["STEP 15A — FINAL RESULTS INVENTORY", "=" * 34, "",
         "Sources: Step 9 (reliability framework), Step 10 (ClinVar cohort + predictor",
         "integration), Step 11 (coverage/missingness selection), Step 12 (design",
         "reliability), Step 13 (temporal exposure), Step 14 (independent DMS).",
         "Frozen numbers only; no new analysis in this step."]
    for r in INVENTORY:
        L += ["", "=" * 70, r["n"], "=" * 70,
              "Question:", f"  {r['q']}", "",
              "Dataset / population:", f"  {r['pop']}", "",
              "Primary estimand:", f"  {r['est']}", "",
              "Key numbers:"]
        L += [f"  - {x}" for x in r["nums"]]
        L += ["", "Conclusion:", f"  {r['concl']}"]
    L += ["", "=" * 70,
          "CROSS-CUTTING METHODOLOGICAL CLAIM",
          "  VariantBenchPy distinguishes numerical benchmark instability from",
          "  reproducible, statistically supported design-induced changes in relative",
          "  predictor performance via gene-aware, selection-aware inference.",
          "", "Validated operating characteristics (Step 9 simulation):",
          "  Winner retention by scenario (verified rerun, 50 reps, 120 genes):",
          "    Stable                    0.99 [0.92, 1.00]",
          "    Dominant                  1.00 [1.00, 1.00]",
          "    Gene-dependent            0.48 [0.09, 0.84]",
          "    Informative-missingness   0.56 [0.23, 0.85]",
          "    Close-performance         0.45 [0.13, 0.80]",
          "  Selection-aware null calibration (200 reps/cell, Wilson CI):",
          "    40 genes:   naive 0.260 [0.204, 0.325] | crossfit 0.070 [0.042, 0.114]",
          "    80 genes:   naive 0.195 [0.146, 0.255] | crossfit 0.050 [0.027, 0.090]",
          "    160 genes:  naive 0.100 [0.066, 0.149] | crossfit 0.045 [0.024, 0.083]",
          "  Power for ΔΔAUROC ~ 0.05 (crossfit):",
          "    40 genes:   18.5% [13.7, 24.5]",
          "    80 genes:   36.0% [29.7, 42.9]",
          "    160 genes:  75.5% [69.1, 80.9]",
          "  These validated power estimates apply only to the simulated gene counts",
          "    and should not be directly extrapolated to the 13,599-gene ClinVar",
          "    benchmark; power depends on effect size, gene dependence, benchmark",
          "    composition, and analysis design.",
          "  NOTE: simulation rates establish that the framework behaves as intended;",
          "    they are not headline manuscript results.",
          "",
          "  Provenance: all Step 9 simulation values above are VERIFIED reruns of the",
          "  frozen datasim/validation code (grid: data/step9_verification_grid.csv).",
          "  Earlier unreproduced draft values (retentions 1.00/0.57/0.11/0.18/0.20;",
          "  naive Type-I 0.23/0.13/0.07; power 18%/43%/73%) are discarded."]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
