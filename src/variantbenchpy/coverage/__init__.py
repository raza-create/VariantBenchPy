"""Coverage analysis: per-predictor / per-gene coverage and shared-subset construction."""
from __future__ import annotations

from variantbenchpy.coverage.analysis import (
    CoverageResult,
    analyze_coverage,
    coverage_depth,
    gene_coverage,
    pairwise_overlap,
    per_predictor_coverage,
    shared_subset_mask,
)

__all__ = [
    "CoverageResult", "analyze_coverage", "per_predictor_coverage",
    "coverage_depth", "pairwise_overlap", "shared_subset_mask", "gene_coverage",
]
