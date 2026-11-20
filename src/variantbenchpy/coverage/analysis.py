"""Coverage analysis: how much of the benchmark can each predictor actually score?

A raw AUROC leaderboard hides the fact that predictors are evaluated on different
variants. This module measures that directly and constructs the *shared subset* —
the variants every (or a chosen set of) predictor can score — so a later stage can
recompute metrics on identical data and expose how much the ranking moved.

Everything here is derived from the boolean missingness matrix; nothing depends on
the labels, so coverage is a property of the predictors, not of the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from variantbenchpy.harmonize.schema import COL_GENE, COL_VARIANT_ID
from variantbenchpy.predictors.base import ScoreMatrix


def per_predictor_coverage(sm: ScoreMatrix) -> pd.DataFrame:
    """Variant- and gene-level coverage for each predictor."""
    n_var = len(sm.scores)
    labeled = sm.labels.notna()
    n_lab = int(labeled.sum())
    gene_by_id = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    n_genes_total = gene_by_id.nunique(dropna=True)

    rows = []
    for name in sm.predictor_names:
        has = sm.scores[name].notna()
        genes_covered = gene_by_id[has].nunique(dropna=True)
        rows.append({
            "predictor": name,
            "n_scored": int(has.sum()),
            "pct_variants": round(100 * has.sum() / n_var, 2) if n_var else 0.0,
            "n_scored_labeled": int((has & labeled).sum()),
            "pct_labeled": round(100 * (has & labeled).sum() / n_lab, 2) if n_lab else 0.0,
            "n_genes_covered": int(genes_covered),
            "pct_genes": round(100 * genes_covered / n_genes_total, 2) if n_genes_total else 0.0,
        })
    return pd.DataFrame(rows)


def coverage_depth(sm: ScoreMatrix) -> pd.DataFrame:
    """Distribution of how many predictors score each variant."""
    n_pred = len(sm.predictor_names)
    depth = sm.scores.notna().sum(axis=1)
    counts = depth.value_counts().reindex(range(0, n_pred + 1), fill_value=0).sort_index()
    df = counts.rename("n_variants").rename_axis("n_predictors").reset_index()
    total = len(sm.scores)
    df["pct_variants"] = (100 * df["n_variants"] / total).round(2) if total else 0.0
    return df


def pairwise_overlap(sm: ScoreMatrix) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(shared_counts, jaccard)`` matrices between predictors."""
    present = sm.scores.notna()
    names = sm.predictor_names
    counts = pd.DataFrame(index=names, columns=names, dtype="int64")
    jacc = pd.DataFrame(index=names, columns=names, dtype="float64")
    for a in names:
        for b in names:
            inter = int((present[a] & present[b]).sum())
            union = int((present[a] | present[b]).sum())
            counts.loc[a, b] = inter
            jacc.loc[a, b] = round(inter / union, 3) if union else np.nan
    return counts, jacc


def shared_subset_mask(
    sm: ScoreMatrix,
    predictors: list[str] | None = None,
    min_predictors: int | None = None,
    labeled_only: bool = True,
) -> pd.Series:
    """Boolean mask of variants usable for a *fair* comparison."""
    cols = predictors or sm.predictor_names
    present = sm.scores[cols].notna()
    need = len(cols) if min_predictors is None else min_predictors
    mask = present.sum(axis=1) >= need
    if labeled_only:
        mask = mask & sm.labels.notna()
    mask.name = "in_shared_subset"
    return mask


def gene_coverage(sm: ScoreMatrix) -> pd.DataFrame:
    """Gene x predictor table of how many variants each predictor scores per gene."""
    gene_by_id = sm.table.set_index(COL_VARIANT_ID)[COL_GENE].reindex(sm.scores.index)
    present = sm.scores.notna()
    out = present.groupby(gene_by_id.values).sum()
    out.index.name = COL_GENE
    return out.astype("int64")


@dataclass
class CoverageResult:
    """Bundle of every coverage view, plus the strict shared-subset mask."""

    per_predictor: pd.DataFrame
    depth: pd.DataFrame
    overlap_counts: pd.DataFrame
    overlap_jaccard: pd.DataFrame
    gene_coverage: pd.DataFrame
    shared_all_mask: pd.Series

    @property
    def n_shared_all(self) -> int:
        return int(self.shared_all_mask.sum())

    def summary_text(self) -> str:
        lines = ["CoverageResult",
                 f"  predictors        : {list(self.per_predictor['predictor'])}",
                 f"  strict shared set : {self.n_shared_all} labeled variants "
                 f"scored by ALL predictors"]
        return "\n".join(lines)


def analyze_coverage(sm: ScoreMatrix) -> CoverageResult:
    """Run every coverage view over a score matrix."""
    counts, jacc = pairwise_overlap(sm)
    return CoverageResult(
        per_predictor=per_predictor_coverage(sm),
        depth=coverage_depth(sm),
        overlap_counts=counts,
        overlap_jaccard=jacc,
        gene_coverage=gene_coverage(sm),
        shared_all_mask=shared_subset_mask(sm, labeled_only=True),
    )
