"""Turn heterogeneous variant inputs into one canonical, deduplicated table.

The harmonizer never silently discards data: everything it drops (unparseable
notation, incomplete join key, missing label, duplicates) is counted in a
:class:`HarmonizationReport` so a benchmark can be audited rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from variantbenchpy.harmonize.parsers import (
    classify_clinsig,
    parse_hgvs_genomic,
    parse_protein_sub,
)
from variantbenchpy.harmonize.schema import (
    CANONICAL_COLUMNS,
    CanonicalKey,
    COL_AA_ALT,
    COL_AA_POS,
    COL_AA_REF,
    COL_ALT,
    COL_BUILD,
    COL_CHROM,
    COL_CLNSIG,
    COL_GENE,
    COL_LABEL,
    COL_POS,
    COL_PROTEIN_ID,
    COL_REF,
    COL_SOURCE_ROW,
    COL_TRANSCRIPT,
    COL_VARIANT_ID,
    GENOMIC_KEY_COLS,
    PROTEIN_KEY_COLS,
    normalize_chrom,
)
from variantbenchpy.harmonize.parsers import _aa_token_to_one

DIRECTIVE_PROTEIN_SUB = "protein_sub"
DIRECTIVE_HGVS_G = "hgvs_g"

_DIRECT_COLUMNS = {
    COL_CHROM, COL_POS, COL_REF, COL_ALT, COL_BUILD,
    COL_PROTEIN_ID, COL_AA_REF, COL_AA_POS, COL_AA_ALT, COL_TRANSCRIPT,
    COL_GENE, COL_LABEL, COL_CLNSIG,
}


@dataclass
class HarmonizationReport:
    """Counts describing exactly what harmonization did to the input."""

    key: str
    n_input: int = 0
    n_key_complete: int = 0
    n_dropped_incomplete_key: int = 0
    n_labeled: int = 0
    n_pathogenic: int = 0
    n_benign: int = 0
    n_unlabeled: int = 0
    n_duplicates_removed: int = 0
    n_output: int = 0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "HarmonizationReport",
            f"  join key                 : {self.key}",
            f"  input rows               : {self.n_input}",
            f"  key-complete rows        : {self.n_key_complete}",
            f"  dropped (incomplete key) : {self.n_dropped_incomplete_key}",
            f"  duplicates removed       : {self.n_duplicates_removed}",
            f"  output variants          : {self.n_output}",
            f"    labeled                : {self.n_labeled} "
            f"(pathogenic={self.n_pathogenic}, benign={self.n_benign})",
            f"    unlabeled              : {self.n_unlabeled}",
        ]
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


class VariantHarmonizer:
    """Standardise variant tables onto the canonical schema."""

    def __init__(self, key: CanonicalKey = CanonicalKey.PROTEIN) -> None:
        self.key = CanonicalKey(key)

    def harmonize(
        self,
        data: pd.DataFrame,
        column_map: dict[str, str],
    ) -> tuple[pd.DataFrame, HarmonizationReport]:
        """Return ``(canonical_table, report)`` for ``data``.

        ``column_map`` maps a canonical column name (e.g. ``"gene"``) or a parse
        directive (``"protein_sub"``, ``"hgvs_g"``) to the source column holding
        that information.
        """
        df = data.reset_index(drop=True)
        report = HarmonizationReport(key=self.key.value, n_input=len(df))
        canon: dict[str, pd.Series] = {}

        for canon_col in _DIRECT_COLUMNS:
            src = column_map.get(canon_col)
            if src is not None and src in df.columns:
                canon[canon_col] = df[src]

        if DIRECTIVE_PROTEIN_SUB in column_map:
            src = column_map[DIRECTIVE_PROTEIN_SUB]
            parsed = df[src].map(parse_protein_sub)
            self._merge(canon, COL_AA_REF, parsed.map(lambda p: p.aa_ref if p else pd.NA))
            self._merge(canon, COL_AA_POS, parsed.map(lambda p: p.aa_pos if p else pd.NA))
            self._merge(canon, COL_AA_ALT, parsed.map(lambda p: p.aa_alt if p else pd.NA))

        if DIRECTIVE_HGVS_G in column_map:
            src = column_map[DIRECTIVE_HGVS_G]
            parsed = df[src].map(parse_hgvs_genomic)
            self._merge(canon, COL_CHROM, parsed.map(lambda g: g.chrom if g else pd.NA))
            self._merge(canon, COL_POS, parsed.map(lambda g: g.pos if g else pd.NA))
            self._merge(canon, COL_REF, parsed.map(lambda g: g.ref if g else pd.NA))
            self._merge(canon, COL_ALT, parsed.map(lambda g: g.alt if g else pd.NA))

        if COL_CHROM in canon:
            canon[COL_CHROM] = canon[COL_CHROM].map(normalize_chrom)
        for aa_col in (COL_AA_REF, COL_AA_ALT):
            if aa_col in canon:
                canon[aa_col] = canon[aa_col].map(
                    lambda v: _aa_token_to_one(str(v)) if pd.notna(v) else pd.NA
                )

        if COL_LABEL not in canon and COL_CLNSIG in canon:
            canon[COL_LABEL] = canon[COL_CLNSIG].map(classify_clinsig)

        out = pd.DataFrame(canon)
        out[COL_SOURCE_ROW] = df.index

        for pos_col in (COL_POS, COL_AA_POS):
            if pos_col in out.columns:
                out[pos_col] = pd.to_numeric(out[pos_col], errors="coerce").astype("Int64")
        if COL_LABEL in out.columns:
            out[COL_LABEL] = pd.to_numeric(out[COL_LABEL], errors="coerce")

        key_cols = PROTEIN_KEY_COLS if self.key == CanonicalKey.PROTEIN else GENOMIC_KEY_COLS
        for kc in key_cols:
            if kc not in out.columns:
                out[kc] = pd.NA
        complete = out[list(key_cols)].notna().all(axis=1)
        report.n_key_complete = int(complete.sum())
        report.n_dropped_incomplete_key = int((~complete).sum())
        out = out[complete].copy()

        out[COL_VARIANT_ID] = self._build_ids(out)

        before = len(out)
        out = out.drop_duplicates(subset=[COL_VARIANT_ID], keep="first").reset_index(drop=True)
        report.n_duplicates_removed = before - len(out)

        for col in CANONICAL_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        out = out[list(CANONICAL_COLUMNS)]

        if COL_LABEL in out.columns:
            labeled = out[COL_LABEL].notna()
            report.n_labeled = int(labeled.sum())
            report.n_pathogenic = int((out[COL_LABEL] == 1).sum())
            report.n_benign = int((out[COL_LABEL] == 0).sum())
            report.n_unlabeled = int((~labeled).sum())
        report.n_output = len(out)

        return out, report

    @staticmethod
    def _merge(canon: dict[str, pd.Series], col: str, values: pd.Series) -> None:
        """Fill ``col`` with ``values`` where it is currently absent/NA."""
        if col in canon:
            canon[col] = canon[col].combine_first(values)
        else:
            canon[col] = values

    def _build_ids(self, out: pd.DataFrame) -> pd.Series:
        if self.key == CanonicalKey.PROTEIN:
            return (
                out[COL_PROTEIN_ID].astype(str) + ":"
                + out[COL_AA_REF].astype(str)
                + out[COL_AA_POS].astype(str)
                + out[COL_AA_ALT].astype(str)
            )
        return (
            out[COL_CHROM].astype(str) + "-"
            + out[COL_POS].astype(str) + "-"
            + out[COL_REF].astype(str) + "-"
            + out[COL_ALT].astype(str)
        )
