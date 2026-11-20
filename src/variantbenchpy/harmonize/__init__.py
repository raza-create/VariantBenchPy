"""Variant harmonization: heterogeneous inputs -> one canonical variant table."""
from __future__ import annotations

from variantbenchpy.harmonize.schema import CanonicalKey
from variantbenchpy.harmonize.harmonizer import (
    HarmonizationReport,
    VariantHarmonizer,
)


def harmonize(data, column_map, key=CanonicalKey.PROTEIN):
    """Convenience wrapper: harmonize ``data`` and return ``(table, report)``."""
    return VariantHarmonizer(key=key).harmonize(data, column_map)


__all__ = ["CanonicalKey", "VariantHarmonizer", "HarmonizationReport", "harmonize"]
