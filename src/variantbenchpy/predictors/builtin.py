"""Ready-made adapters for the predictors named in the benchmark design.

Orientation is the subtle part and the reason this registry exists:
  - AlphaMissense, EVE, REVEL, CADD, PolyPhen-2  -> HIGHER is damaging
  - SIFT                                          -> LOWER is damaging
  - PLM log-likelihood-ratio scores              -> LOWER is damaging (configurable)
"""

from __future__ import annotations

import pandas as pd

from variantbenchpy.harmonize.schema import CanonicalKey
from variantbenchpy.predictors.base import Predictor, ScoreOrientation


def alphamissense(score_column="am_pathogenicity", protein_id_column="uniprot_id",
                  protein_sub_column="protein_variant") -> Predictor:
    """AlphaMissense (Cheng et al. 2023). Protein key, higher = more pathogenic."""
    return Predictor(
        name="AlphaMissense", key=CanonicalKey.PROTEIN,
        orientation=ScoreOrientation.HIGHER_IS_DAMAGING, score_column=score_column,
        column_map={"protein_id": protein_id_column, "protein_sub": protein_sub_column},
        decision_threshold=0.564, raw_range=(0.0, 1.0), notes="Cheng et al., Science 2023.")


def eve(score_column="EVE_scores", protein_id_column="uniprot_id",
        protein_sub_column="mutations") -> Predictor:
    """EVE (Frazer et al. 2021). Protein key, higher = more pathogenic."""
    return Predictor(
        name="EVE", key=CanonicalKey.PROTEIN,
        orientation=ScoreOrientation.HIGHER_IS_DAMAGING, score_column=score_column,
        column_map={"protein_id": protein_id_column, "protein_sub": protein_sub_column},
        decision_threshold=0.5, raw_range=(0.0, 1.0), notes="Frazer et al., Nature 2021.")


def revel(score_column="REVEL", chrom_column="chrom", pos_column="pos",
          ref_column="ref", alt_column="alt") -> Predictor:
    """REVEL (Ioannidis et al. 2016). Genomic key, higher = more pathogenic."""
    return Predictor(
        name="REVEL", key=CanonicalKey.GENOMIC,
        orientation=ScoreOrientation.HIGHER_IS_DAMAGING, score_column=score_column,
        column_map={"chrom": chrom_column, "pos": pos_column, "ref": ref_column, "alt": alt_column},
        decision_threshold=0.5, raw_range=(0.0, 1.0), notes="Ioannidis et al., AJHG 2016.")


def cadd(score_column="PHRED", chrom_column="chrom", pos_column="pos",
         ref_column="ref", alt_column="alt") -> Predictor:
    """CADD (Rentzsch et al. 2019), PHRED-scaled. Genomic key, higher = more damaging."""
    return Predictor(
        name="CADD", key=CanonicalKey.GENOMIC,
        orientation=ScoreOrientation.HIGHER_IS_DAMAGING, score_column=score_column,
        column_map={"chrom": chrom_column, "pos": pos_column, "ref": ref_column, "alt": alt_column},
        decision_threshold=20.0, raw_range=(0.0, 99.0), notes="Rentzsch et al., NAR 2019.")


def sift(score_column="SIFT_score", protein_id_column="uniprot_id",
         protein_sub_column="protein_variant") -> Predictor:
    """SIFT (Ng & Henikoff 2003). Protein key, LOWER = more damaging (flipped here)."""
    return Predictor(
        name="SIFT", key=CanonicalKey.PROTEIN,
        orientation=ScoreOrientation.LOWER_IS_DAMAGING, score_column=score_column,
        column_map={"protein_id": protein_id_column, "protein_sub": protein_sub_column},
        decision_threshold=0.05, raw_range=(0.0, 1.0),
        notes="Ng & Henikoff, NAR 2003. Lower is more damaging.")


def polyphen2(score_column="pph2_prob", protein_id_column="uniprot_id",
              protein_sub_column="protein_variant") -> Predictor:
    """PolyPhen-2 (Adzhubei et al. 2010). Protein key, higher = more damaging."""
    return Predictor(
        name="PolyPhen-2", key=CanonicalKey.PROTEIN,
        orientation=ScoreOrientation.HIGHER_IS_DAMAGING, score_column=score_column,
        column_map={"protein_id": protein_id_column, "protein_sub": protein_sub_column},
        decision_threshold=0.5, raw_range=(0.0, 1.0), notes="Adzhubei et al., Nat Methods 2010.")


def plm(name, score_column, protein_id_column="uniprot_id", protein_sub_column="protein_variant",
        orientation=ScoreOrientation.LOWER_IS_DAMAGING, decision_threshold=None) -> Predictor:
    """Generic protein-language-model adapter (ESM, ESM-1v, Tranception, etc.)."""
    return Predictor(
        name=name, key=CanonicalKey.PROTEIN, orientation=orientation,
        score_column=score_column,
        column_map={"protein_id": protein_id_column, "protein_sub": protein_sub_column},
        decision_threshold=decision_threshold, notes="Generic protein-language-model score.")


BUILTIN_PREDICTORS = {
    "AlphaMissense": alphamissense, "EVE": eve, "REVEL": revel,
    "CADD": cadd, "SIFT": sift, "PolyPhen-2": polyphen2,
}
