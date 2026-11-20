"""Canonical variant schema shared across the whole package.

Every downstream stage (predictors, metrics, bias, evaluation) assumes variants
have already been reduced to this schema. Keeping the vocabulary in one place
means a predictor adapter never has to guess how a chromosome or amino acid is
spelled.
"""

from __future__ import annotations

from enum import Enum


class CanonicalKey(str, Enum):
    """The identity space a benchmark joins predictors on.

    Predictors disagree on how they address a variant: CADD/REVEL are genomic,
    EVE/SIFT/PolyPhen are protein-level, AlphaMissense provides both. The join
    key decides which columns must be present and non-null for a variant to be
    matched against a predictor.
    """

    GENOMIC = "genomic"   # (chrom, pos, ref, alt) on a given build
    PROTEIN = "protein"   # (protein_id, aa_ref, aa_pos, aa_alt)


# --- Canonical columns -------------------------------------------------------
# Genomic coordinates (1-based, VCF-style ref/alt).
COL_CHROM = "chrom"
COL_POS = "pos"
COL_REF = "ref"
COL_ALT = "alt"
COL_BUILD = "genome_build"

# Protein coordinates (1-based residue index, single-letter amino acids).
COL_PROTEIN_ID = "protein_id"   # UniProt accession preferred
COL_AA_REF = "aa_ref"
COL_AA_POS = "aa_pos"
COL_AA_ALT = "aa_alt"
COL_TRANSCRIPT = "transcript"

# Biological grouping + labels.
COL_GENE = "gene"                        # HGNC symbol, the gene-level unit
COL_LABEL = "label"                      # 1 = pathogenic, 0 = benign, NA = unusable
COL_CLNSIG = "clinical_significance"     # raw ClinVar string, kept for provenance

# Bookkeeping.
COL_VARIANT_ID = "variant_id"   # stable canonical string id
COL_SOURCE_ROW = "source_row"   # index into the caller's original input

GENOMIC_KEY_COLS = (COL_CHROM, COL_POS, COL_REF, COL_ALT)
PROTEIN_KEY_COLS = (COL_PROTEIN_ID, COL_AA_REF, COL_AA_POS, COL_AA_ALT)

CANONICAL_COLUMNS = (
    COL_VARIANT_ID,
    COL_CHROM, COL_POS, COL_REF, COL_ALT, COL_BUILD,
    COL_PROTEIN_ID, COL_AA_REF, COL_AA_POS, COL_AA_ALT, COL_TRANSCRIPT,
    COL_GENE, COL_LABEL, COL_CLNSIG, COL_SOURCE_ROW,
)


# --- Amino-acid vocabulary ---------------------------------------------------
# Three-letter -> one-letter, including the stop spellings HGVS allows.
AA_THREE_TO_ONE = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C",
    "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
    "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P",
    "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
    "sec": "U", "pyl": "O", "ter": "*", "stop": "*", "xaa": "X",
}
VALID_AA_ONE = set("ACDEFGHIKLMNPQRSTVWY") | {"*", "U", "O", "X"}


# --- ClinVar clinical significance -> binary label ---------------------------
# Only high-confidence assertions become 0/1; everything else stays NA so that
# VUS / conflicting variants never silently pollute the metrics.
CLNSIG_PATHOGENIC = {
    "pathogenic",
    "likely pathogenic",
    "pathogenic/likely pathogenic",
    "pathogenic, low penetrance",
    "likely pathogenic, low penetrance",
}
CLNSIG_BENIGN = {
    "benign",
    "likely benign",
    "benign/likely benign",
}
CLNSIG_AMBIGUOUS = {
    "uncertain significance",
    "conflicting interpretations of pathogenicity",
    "conflicting classifications of pathogenicity",
    "not provided",
    "other",
    "drug response",
    "risk factor",
    "association",
    "protective",
    "affects",
}


def normalize_chrom(value: object) -> str | None:
    """Normalise a chromosome label to bare form: ``chr7`` -> ``7``, ``M`` -> ``MT``."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "na", "."}:
        return None
    if s.lower().startswith("chr"):
        s = s[3:]
    s = s.upper()
    if s in {"M", "MT"}:
        return "MT"
    return s
