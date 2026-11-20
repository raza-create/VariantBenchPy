"""Common interface that makes heterogeneous predictors comparable.

Predictors disagree on almost everything: the key they address variants by
(genomic vs protein), the column their score lives in, the score's range, and —
critically — its *orientation* (SIFT calls a variant damaging when its score is
LOW; AlphaMissense when it is HIGH). If those differences leak into the metrics,
AUROC is silently computed on inconsistent conventions.

This module reduces every predictor to the same convention:

    a single oriented score per variant where HIGHER ALWAYS MEANS MORE DAMAGING,

plus an explicit record of *which variants each predictor could not score*. That
missingness record is not a nuisance to be dropped — it is the raw material for
the coverage and missingness-bias analyses that follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from variantbenchpy.harmonize.harmonizer import VariantHarmonizer
from variantbenchpy.harmonize.schema import CanonicalKey, COL_SOURCE_ROW, COL_VARIANT_ID


class ScoreOrientation(str, Enum):
    """Direction in which a predictor's raw score signals pathogenicity."""

    HIGHER_IS_DAMAGING = "higher_is_damaging"   # AlphaMissense, EVE, CADD, REVEL, PolyPhen-2
    LOWER_IS_DAMAGING = "lower_is_damaging"      # SIFT, many PLM log-likelihood ratios


@dataclass
class Predictor:
    """Adapter describing how to read one predictor's scores."""

    name: str
    key: CanonicalKey
    orientation: ScoreOrientation
    score_column: str
    column_map: dict[str, str]
    decision_threshold: float | None = None
    raw_range: tuple[float, float] | None = None
    notes: str = ""

    def orient(self, raw_score: pd.Series) -> pd.Series:
        """Map raw scores into the shared convention (higher = more damaging)."""
        s = pd.to_numeric(raw_score, errors="coerce")
        if self.orientation == ScoreOrientation.LOWER_IS_DAMAGING:
            return -s
        return s

    @property
    def oriented_threshold(self) -> float | None:
        """The decision threshold expressed in oriented (higher=damaging) space."""
        if self.decision_threshold is None:
            return None
        if self.orientation == ScoreOrientation.LOWER_IS_DAMAGING:
            return -self.decision_threshold
        return self.decision_threshold

    def build_scores(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Harmonize a raw predictor table to ``(variant_id, <name>)`` oriented scores."""
        harmonizer = VariantHarmonizer(key=self.key)
        htab, _ = harmonizer.harmonize(raw, self.column_map)
        raw_reset = raw.reset_index(drop=True)
        if self.score_column not in raw_reset.columns:
            raise KeyError(
                f"score_column {self.score_column!r} not found in raw table for "
                f"predictor {self.name!r}"
            )
        raw_scores = pd.to_numeric(raw_reset[self.score_column], errors="coerce")
        htab = htab.assign(_raw=htab[COL_SOURCE_ROW].map(raw_scores))
        oriented = self.orient(htab["_raw"])
        out = pd.DataFrame({COL_VARIANT_ID: htab[COL_VARIANT_ID].values, self.name: oriented.values})
        out = out.sort_values(self.name, ascending=False).drop_duplicates(
            subset=[COL_VARIANT_ID], keep="first"
        )
        return out.dropna(subset=[self.name]).reset_index(drop=True)


@dataclass
class ScoreMatrix:
    """Benchmark variants aligned against every predictor's oriented score."""

    table: pd.DataFrame
    scores: pd.DataFrame
    predictors: dict[str, Predictor] = field(default_factory=dict)

    @property
    def predictor_names(self) -> list[str]:
        return list(self.scores.columns)

    @property
    def missing_mask(self) -> pd.DataFrame:
        """Boolean matrix: True where a predictor has no score for a variant."""
        return self.scores.isna()

    @property
    def labels(self) -> pd.Series:
        """The 0/1 label aligned to the score matrix index (NaN where unlabeled)."""
        lab = self.table.set_index(COL_VARIANT_ID)["label"]
        return lab.reindex(self.scores.index)

    def summary(self) -> pd.DataFrame:
        """Per-predictor coverage headline (n scored, % of variants, % of labeled)."""
        n = len(self.scores)
        labeled = self.labels.notna()
        rows = []
        for name in self.predictor_names:
            has = self.scores[name].notna()
            rows.append({
                "predictor": name,
                "n_scored": int(has.sum()),
                "pct_variants": round(100 * has.sum() / n, 1) if n else 0.0,
                "n_scored_labeled": int((has & labeled).sum()),
                "pct_labeled": round(100 * (has & labeled).sum() / labeled.sum(), 1)
                if labeled.sum() else 0.0,
            })
        return pd.DataFrame(rows)


def build_score_matrix(
    canonical_table: pd.DataFrame,
    predictor_inputs: dict[str, tuple[Predictor, pd.DataFrame]],
) -> ScoreMatrix:
    """Align many predictors onto a harmonized benchmark table."""
    ids = canonical_table[COL_VARIANT_ID]
    scores = pd.DataFrame(index=pd.Index(ids, name=COL_VARIANT_ID))
    used: dict[str, Predictor] = {}
    for label, (predictor, raw) in predictor_inputs.items():
        built = predictor.build_scores(raw).set_index(COL_VARIANT_ID)[predictor.name]
        scores[label] = built.reindex(scores.index)
        used[label] = predictor
    scores = scores.astype("float64")
    return ScoreMatrix(table=canonical_table.reset_index(drop=True), scores=scores, predictors=used)
