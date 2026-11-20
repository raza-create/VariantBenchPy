"""Ranking-stability / fragility analysis and equal-gene-weighted evaluation."""
from __future__ import annotations

from variantbenchpy.evaluation.weighting import config_scores, gene_weighted_metric
from variantbenchpy.evaluation.stability import (
    benchmark_configurations, kendalls_w, pairwise_delta_auroc,
    probability_best, ranking_fragility,
)
from variantbenchpy.evaluation.fragility import (
    RankingFragilityReport, ranking_fragility_report,
)
from variantbenchpy.evaluation.reliability import (
    RankReliabilityAudit, rank_reliability_audit,
)
from variantbenchpy.evaluation.validation import (
    structural_power, naive_vs_selection, negative_controls,
    realized_effect, method_comparison,
)

__all__ = [
    "config_scores", "gene_weighted_metric", "benchmark_configurations",
    "kendalls_w", "ranking_fragility", "probability_best", "pairwise_delta_auroc",
    "RankingFragilityReport", "ranking_fragility_report",
    "RankReliabilityAudit", "rank_reliability_audit",
    "structural_power", "naive_vs_selection", "negative_controls",
    "realized_effect", "method_comparison",
]
