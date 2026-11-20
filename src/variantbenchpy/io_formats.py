"""Loaders for the two formats users have first: delimited text and VCF.

Kept dependency-free (no pysam) so a VariantSet can be built anywhere. The VCF reader
understands the ClinVar-style INFO encoding well enough to pull clinical significance,
gene, allele frequency and any requested predictor fields out of the INFO column.
"""

from __future__ import annotations

import gzip

import pandas as pd


def read_table(path: str, sep: str | None = None) -> pd.DataFrame:
    """Read CSV/TSV (sep auto-detected from extension when not given)."""
    if sep is None:
        sep = "\t" if path.endswith((".tsv", ".tab", ".tsv.gz")) else ","
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return pd.read_csv(fh, sep=sep, dtype=str, comment=None)


def _parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if info in (".", ""):
        return out
    for field in info.split(";"):
        if "=" in field:
            k, v = field.split("=", 1)
            out[k] = v
        else:
            out[field] = "True"
    return out


def read_vcf(path: str, info_fields: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a VCF into a flat DataFrame.

    ``info_fields`` maps an output column name to an INFO key to extract, e.g.
    ``{"clinical_significance": "CLNSIG", "gene": "GENEINFO", "allele_frequency": "AF"}``.
    CHROM/POS/ID/REF/ALT are always returned.
    """
    info_fields = info_fields or {}
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, vid, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            info = _parse_info(parts[7])
            rec = {"chrom": chrom, "pos": pos, "id": vid, "ref": ref, "alt": alt}
            for out_col, key in info_fields.items():
                rec[out_col] = info.get(key)
            rows.append(rec)
    df = pd.DataFrame(rows)
    if "gene" in df.columns:
        df["gene"] = df["gene"].str.split(":").str[0].str.split("|").str[0]
    return df
