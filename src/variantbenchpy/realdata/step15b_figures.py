"""STEP 15B — final figure/table plan. Specification only; no analysis, no plotting."""
from __future__ import annotations

MAIN = [
    dict(
        id="Figure 1", result="RESULT 3 (+ RESULT 1)",
        title="Benchmark-design uncertainty is distinct from sampling uncertainty",
        panels=[
            "A. Rank distribution heatmap: predictors x ranks #1-#5, cell = proportion of "
            "36 prespecified specifications (5-predictor primary layer).",
            "B. Winner retention vs Kendall's W: retention 0.67 with W = 0.91 [0.89, 0.93], "
            "annotated to show high whole-ranking concordance with an unstable top rank.",
            "C. Sampling vs design uncertainty: REVEL bootstrap P(#1) = 1.00 under the "
            "fixed reference design against 24/36 configuration retention.",
            "D. Design-sensitivity shares (descriptive): evaluation_unit 74%, evidence 21%, "
            "composition 5%, coverage 0%.",
        ],
        source="data/step12_audit.txt; step12_specgrid.parquet",
        message="The ranking is globally concordant; instability is concentrated at rank 1 "
                "and driven by the evaluation unit.",
    ),
    dict(
        id="Figure 2", result="RESULT 2",
        title="Complete-case predictor coverage reshapes the benchmark population",
        panels=[
            "A. Population funnel: 225,208 -> 187,155 (5-pred) -> 53,552 (6-pred) variants; "
            "15,249 -> 13,599 -> 1,981 gene clusters.",
            "B. Composition shift: P/LP prevalence 31.9% -> 56.9%; gene concentration "
            "(top10/50/100 = 12.1/32.2/44.9%, N_eff = 282 of 1,981).",
            "C. Matched ΔAUROC forest plot, 5-pred shared -> EVE-constrained, with "
            "gene-clustered CIs (CADD -0.041 [-0.054, -0.029] largest).",
            "D. Pairwise ΔΔAUROC: AlphaMissense-CADD +0.026 [+0.017, +0.035], Holm p<0.001.",
        ],
        source="step11_selection.py output; benchmark_scores_6pred.parquet",
        message="Requiring all predictors changes who is being evaluated, not just how "
                "well they score.",
    ),
    dict(
        id="Figure 3", result="RESULT 2 (missingness mechanism)",
        title="Predictor availability is associated with clinical class",
        panels=[
            "A. Coverage by class per predictor (P/LP vs B/LB): EVE +29.3pp, "
            "PolyPhen-2 +6.1pp, others <=2.3pp.",
            "B. EVE availability model: unadjusted OR 4.21 [4.12, 4.29] vs within-gene "
            "cluster-robust OR 2.16 [1.94, 2.40].",
            "C. Evidence-stratum heterogeneity: matched ΔAUROC at exactly-1-star vs "
            ">=2-star per predictor, with Holm-adjusted difference CIs.",
        ],
        source="step11_selection.py; eve_audit.py",
        message="Missingness is informative and its downstream effect attenuates at higher "
                "ClinVar evidence levels (associational, not causal).",
    ),
    dict(
        id="Figure 4", result="RESULT 4",
        title="Temporal exposure-resistant evaluation",
        panels=[
            "A. ClinVar germline First-in-ClinVar year distribution with the 2021-10-27 and "
            "2023-09-22 publication-proxy landmarks marked.",
            "B. Matched ΔAUROC (full -> post-2023), paired gene-clustered CIs; all five "
            "predictors positive (PolyPhen-2 +0.030 to AlphaMissense +0.010).",
            "C. Observed vs composition-standardized ΔAUROC per predictor.",
            "D. Post-2021 sensitivity: same direction for all five predictors.",
        ],
        source="step13e_temporal_perf.py; step13g_post2021.py",
        message="Performance does not degrade on variants first public after predictor "
                "release; the leader is unchanged while relative gaps shift.",
    ),
    dict(
        id="Figure 5", result="RESULT 5",
        title="Independent experimental (DMS) validation",
        panels=[
            "A. Mapping provenance: ProteinGym human single-AA mutants -> "
            "UNIQUE_MANE_GRCH38_SNV 38,517, with the BRCA2 MaveDB check 51/51.",
            "B. Per-assay Spearman distributions per predictor across 36 assays "
            "(median assay rho: AlphaMissense 0.519 ... PolyPhen-2 0.349).",
            "C. Protein-clustered bootstrap: median rho with 95% CI, plus "
            "AlphaMissense-REVEL +0.079 [+0.006, +0.142] and P(best) 0.982.",
            "D. Robustness strip: assay- vs protein-weighted ranking (Kendall 1.000), "
            "LOPO 32/32, assay category 5/5.",
        ],
        source="step14e/f/g/h/i outputs; step14_final_summary.txt",
        message="The leading experimental result is robust to evaluation-unit choice, "
                "leave-one-protein-out analysis, and ProteinGym assay category.",
    ),
]

SUPP_FIGURES = [
    ("Fig S1", "Step 9 scenario suite: winner retention by simulated scenario "
               "(stable 0.99 ... close 0.45), 50 reps / 120 genes."),
    ("Fig S2", "Step 9 operating characteristics: null calibration (naive vs crossfit at "
               "40/80/160 genes) and power at ΔΔAUROC ~ 0.05, Wilson CIs."),
    ("Fig S3", "ClinVar cohort construction: exclusions (VUS 2,139,383; conflicting "
               "110,991), star distribution, prespecified analysis populations."),
    ("Fig S4", "dbNSFP transcript resolution: MANE-Select vs fallback vs no-usable-score "
               "for SIFT and PolyPhen-2."),
    ("Fig S5", "EVE mapping cascade: six terminal states over 222,152 dbNSFP-matched "
               "variants; coverage by class and evidence."),
    ("Fig S6", "Step 12 six-predictor secondary layer: rank distribution, entropy, "
               "magnitude sensitivity."),
    ("Fig S7", "ClinVar gene-level performance heterogeneity: per-gene AUROC "
               "distributions, tie-safe unique/fractional win counts, and genes with "
               "AUROC <0.60 for the five-predictor shared benchmark."),
]

SUPP_TABLES = [
    ("Table S1", "Predictor provenance: source file, version, release/download date, "
                 "checksums, score orientation, transformation."),
    ("Table S2", "Frozen ClinVar cohort: 225,208 variants; integrity audit incl. "
                 "MANE-informed primary-gene rule and somatic/onco reconciliation."),
    ("Table S3", "Step 10B integrity: matching (222,152 exact; 1,556 absent; 1,500 allele "
                 "mismatch), score validity, coverage by class/evidence/gene."),
    ("Table S4", "Step 12 specification grid: all 36 specifications per layer with per-spec "
                 "AUROC, winner, n variants, n gene clusters, eligible genes."),
    ("Table S5", "Step 11 sensitivity populations: matched ΔAUROC under multi-gene, "
                 "GENEINFO-fallback, somatic/onco, >=1-star, >=2-star exclusions."),
    ("Table S6", "Step 13 predictor chronology: publication-date proxies, strict-eligibility "
                 "flags, and the 20/20 First-in-ClinVar semantic validation."),
    ("Table S7", "ProteinGym crosswalk: 96 human assays, MaveDB match status, one-to-one "
                 "QC, and MANE mapping-state counts per assay."),
    ("Table S8", "Per-assay DMS Spearman for all predictors (available-case, 5-pred shared, "
                 "6-pred shared) with n_scored and p-values."),
]

RULES = [
    "Every figure/table draws only on frozen artifacts; no new estimation in Step 15B.",
    "All AUROC differences carry gene-clustered (ClinVar) or protein-clustered (DMS) CIs.",
    "Descriptive panels (design-sensitivity shares, rank distributions) are labelled "
    "DESCRIPTIVE and never presented as inferential.",
    "Temporal panels use 'temporal exposure-resistant', never 'leakage-free'.",
    "EVE appears only in coverage/secondary panels, never in primary DMS inference.",
    "Simulation panels (Fig S1-S2) are supplementary; they establish framework behaviour, "
    "not headline results.",
]


def run():
    L = ["STEP 15B — FIGURE AND TABLE PLAN", "=" * 32, "",
         f"Main figures: {len(MAIN)}   Supplementary figures: {len(SUPP_FIGURES)}   "
         f"Supplementary tables: {len(SUPP_TABLES)}", ""]
    for f in MAIN:
        L += ["=" * 70, f"{f['id']}  [{f['result']}]", f"  {f['title']}", "", "  Panels:"]
        L += [f"    {p}" for p in f["panels"]]
        L += ["", f"  Source: {f['source']}", f"  Message: {f['message']}", ""]
    L += ["=" * 70, "SUPPLEMENTARY FIGURES", "=" * 70]
    L += [f"  {i}: {d}" for i, d in SUPP_FIGURES]
    L += ["", "=" * 70, "SUPPLEMENTARY TABLES", "=" * 70]
    L += [f"  {i}: {d}" for i, d in SUPP_TABLES]
    L += ["", "=" * 70, "PRESENTATION RULES", "=" * 70]
    L += [f"  - {r}" for r in RULES]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
