"""Small, dependency-free parsers for the notations found in variant inputs.

Each parser returns either a structured result or ``None`` (never raises on a
malformed cell) so the harmonizer can count failures and report them instead of
crashing on one bad row in a million.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from variantbenchpy.harmonize.schema import (
    AA_THREE_TO_ONE,
    CLNSIG_AMBIGUOUS,
    CLNSIG_BENIGN,
    CLNSIG_PATHOGENIC,
    VALID_AA_ONE,
    normalize_chrom,
)


@dataclass(frozen=True)
class ProteinSub:
    """A parsed protein substitution: ``aa_ref`` @ ``aa_pos`` -> ``aa_alt``."""

    aa_ref: str
    aa_pos: int
    aa_alt: str


@dataclass(frozen=True)
class GenomicVar:
    """A parsed genomic SNV/indel in canonical (bare-chrom, 1-based) form."""

    chrom: str
    pos: int
    ref: str
    alt: str


def _aa_token_to_one(token: str) -> str | None:
    """Map a single amino-acid token (1- or 3-letter, any case) to one-letter."""
    t = token.strip()
    if not t:
        return None
    if len(t) == 1:
        u = t.upper()
        return u if u in VALID_AA_ONE else None
    if t == "*":
        return "*"
    if t == "=":
        return "="
    return AA_THREE_TO_ONE.get(t.lower())


# Matches e.g. p.Arg273His, p.(R273H), Arg273Ter, R273*, p.Gly12=
_PROT_RE = re.compile(
    r"""^
    (?:p\.)?                      # optional p. prefix
    \(?                           # optional opening paren
    (?P<ref>[A-Za-z]{1,3})        # reference residue (1- or 3-letter)
    (?P<pos>\d+)                  # 1-based residue position
    (?P<alt>[A-Za-z]{1,3}|\*|=)   # alt residue, stop (*) or synonymous (=)
    \)?                           # optional closing paren
    $""",
    re.VERBOSE,
)


def parse_protein_sub(text: object) -> ProteinSub | None:
    """Parse ``p.Arg273His`` / ``R273H`` / ``p.(Gly12Asp)`` into a ProteinSub.

    Synonymous notation (``p.Gly12=``) resolves to ``aa_alt == aa_ref``. Returns
    ``None`` for anything that is not a clean single-residue substitution
    (frameshifts, deletions, duplications are intentionally not handled here —
    they are not missense and do not belong in a missense benchmark).
    """
    if text is None:
        return None
    m = _PROT_RE.match(str(text).strip())
    if not m:
        return None
    ref = _aa_token_to_one(m.group("ref"))
    alt_tok = m.group("alt")
    pos = int(m.group("pos"))
    if ref is None:
        return None
    if alt_tok == "=":
        alt = ref  # synonymous
    else:
        alt = _aa_token_to_one(alt_tok)
    if alt is None:
        return None
    return ProteinSub(aa_ref=ref, aa_pos=pos, aa_alt=alt)


# Matches e.g. chr7:g.140453136A>T, 7:g.140453136A>T, NC_000007.14:g.140753336A>T
_HGVS_G_RE = re.compile(
    r"""^
    (?P<chrom>[^:]+):
    g\.
    (?P<pos>\d+)
    (?P<ref>[ACGTN]+)
    >
    (?P<alt>[ACGTN]+)
    $""",
    re.VERBOSE | re.IGNORECASE,
)


def parse_hgvs_genomic(text: object) -> GenomicVar | None:
    """Parse a genomic HGVS substitution ``chr7:g.140453136A>T``.

    RefSeq accessions like ``NC_000007.14`` are passed through as the chromosome
    field unchanged; mapping accession -> chromosome number is a separate,
    annotation-dependent concern handled later.
    """
    if text is None:
        return None
    m = _HGVS_G_RE.match(str(text).strip())
    if not m:
        return None
    chrom = normalize_chrom(m.group("chrom"))
    if chrom is None:
        return None
    return GenomicVar(
        chrom=chrom,
        pos=int(m.group("pos")),
        ref=m.group("ref").upper(),
        alt=m.group("alt").upper(),
    )


def classify_clinsig(text: object) -> float | None:
    """Map a ClinVar clinical-significance string to a binary label.

    Returns ``1.0`` (pathogenic side), ``0.0`` (benign side) or ``None`` when the
    assertion is ambiguous / unusable. Matching is case-insensitive and tolerant
    of surrounding whitespace.
    """
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s or s in {"nan", "na"}:
        return None
    if s in CLNSIG_PATHOGENIC:
        return 1.0
    if s in CLNSIG_BENIGN:
        return 0.0
    if s in CLNSIG_AMBIGUOUS:
        return None
    has_path = "pathogenic" in s
    has_benign = "benign" in s
    if has_path and not has_benign and "conflict" not in s:
        return 1.0
    if has_benign and not has_path and "conflict" not in s:
        return 0.0
    return None
