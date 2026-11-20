"""Predictor integration: raw predictor files -> one oriented score matrix."""
from __future__ import annotations

from variantbenchpy.predictors.base import (
    Predictor,
    ScoreMatrix,
    ScoreOrientation,
    build_score_matrix,
)
from variantbenchpy.predictors import builtin

__all__ = ["Predictor", "ScoreMatrix", "ScoreOrientation", "build_score_matrix", "builtin"]
