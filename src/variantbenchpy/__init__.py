"""VariantBenchPy — coverage- and bias-aware benchmarking for variant-effect predictors."""
from __future__ import annotations

__version__ = "0.1.0"

from variantbenchpy.harmonize import (
    CanonicalKey, HarmonizationReport, VariantHarmonizer, harmonize,
)
from variantbenchpy.variantset import VariantSet, ValidationReport

__all__ = [
    "__version__", "CanonicalKey", "VariantHarmonizer", "HarmonizationReport",
    "harmonize", "VariantSet", "ValidationReport",
]
