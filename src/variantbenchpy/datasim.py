"""Reproducible synthetic benchmark generator.

Real ClinVar + predictor downloads are large and gated; for testing, tutorials and
*method validation* a controllable simulator is invaluable. This one reproduces the
three structures VariantBenchPy exists to measure, so its own methods can be checked
against a known ground truth:

  - gene clustering        : variants grouped in genes of uneven size, each gene with
                             its own pathogenic rate and its own per-predictor
                             difficulty (=> within-gene correlated errors);
  - differential coverage  : each predictor scores a different, tunable fraction;
  - informative missingness: missingness can be made to depend on pathogenicity, so
                             the missingness-bias test has something real to detect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_AA = list("ACDEFGHIKLMNPQRSTVWY")

_KIND_COLUMNS = {
    "alphamissense": ("uniprot_id", "protein_variant", "am_pathogenicity", False),
    "eve":           ("uniprot_id", "mutations", "EVE_scores", False),
    "polyphen2":     ("uniprot_id", "protein_variant", "pph2_prob", False),
    "sift":          ("uniprot_id", "protein_variant", "SIFT_score", True),
    "plm":           ("uniprot_id", "protein_variant", "llr", True),
}


@dataclass
class PredictorSpec:
    """How one simulated predictor behaves.

    strength      : mean signal separating pathogenic from benign (higher => better).
    gene_sd       : SD of per-gene strength => performance heterogeneity across genes.
    missing       : baseline fraction of variants left unscored.
    missing_bias  : >0 makes pathogenic variants MORE likely to be missing
                    (informative missingness); 0 = missing-completely-at-random.
    """

    name: str
    kind: str
    strength: float = 2.0
    gene_sd: float = 1.0
    missing: float = 0.1
    missing_bias: float = 0.0
    group_effect: dict | None = None   # {group_id: extra strength} for a
    #                                    reproducible per-gene-group advantage


def simulate_benchmark(
    n_genes: int = 300,
    min_variants: int = 5,
    max_variants: int = 250,
    specs: list[PredictorSpec] | None = None,
    seed: int = 0,
    n_groups: int = 3,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return ``(variants_df, {predictor_name: raw_scores_df})``."""
    rng = np.random.default_rng(seed)
    if specs is None:
        specs = [
            PredictorSpec("AlphaMissense", "alphamissense", 2.4, 1.2, 0.08, 0.0),
            PredictorSpec("EVE", "eve", 2.0, 1.2, 0.22, 0.6),
            PredictorSpec("PolyPhen-2", "polyphen2", 1.7, 1.1, 0.12, 0.0),
            PredictorSpec("SIFT", "sift", 1.4, 1.1, 0.15, 0.0),
            PredictorSpec("ESM-1v(PLM)", "plm", 1.9, 1.3, 0.10, 0.0),
        ]

    genes, unis, subs, clnsig, labels, gidx = [], [], [], [], [], []
    for gi in range(n_genes):
        gene = f"GENE{gi:04d}"
        uni = f"P{gi:05d}"
        n_var = int(rng.integers(min_variants, max_variants + 1))
        path_rate = rng.beta(2, 2)
        seen = set()
        for _ in range(n_var):
            while True:
                pos = int(rng.integers(1, 1500))
                ref, alt = rng.choice(_AA, 2, replace=False)
                key = (ref, pos, alt)
                if key not in seen:
                    seen.add(key)
                    break
            lab = int(rng.random() < path_rate)
            genes.append(gene); unis.append(uni); subs.append(f"{ref}{pos}{alt}")
            clnsig.append("Pathogenic" if lab else "Benign"); labels.append(lab); gidx.append(gi)

    variants = pd.DataFrame({
        "gene": genes, "uniprot_id": unis, "protein_variant": subs,
        "clnsig": clnsig, "_label": labels, "_gene_idx": gidx,
    })
    y = variants["_label"].to_numpy()
    g_idx = variants["_gene_idx"].to_numpy()

    raw_tables: dict[str, pd.DataFrame] = {}
    for spec_i, spec in enumerate(specs):
        id_col, sub_col, score_col, lower_dmg = _KIND_COLUMNS[spec.kind]
        # deterministic per-predictor stream (NOT Python's per-process-salted hash())
        sr = np.random.default_rng(seed * 1000 + spec_i + 1)
        per_gene_strength = sr.normal(spec.strength, spec.gene_sd, size=n_genes).clip(0.1)
        if spec.group_effect:
            group_of_gene = np.arange(n_genes) % n_groups
            boost = np.array([spec.group_effect.get(int(group_of_gene[k]), 0.0)
                              for k in range(n_genes)])
            per_gene_strength = (per_gene_strength + boost).clip(0.1)
        strength = per_gene_strength[g_idx]
        signal = strength * y + sr.normal(0, 1, size=len(y))
        raw_score = -signal if lower_dmg else signal

        logit = np.log(spec.missing / (1 - spec.missing)) + spec.missing_bias * (y - y.mean())
        p_missing = 1 / (1 + np.exp(-logit))
        keep = sr.random(len(y)) > p_missing

        d = pd.DataFrame({
            id_col: variants["uniprot_id"].to_numpy()[keep],
            sub_col: variants["protein_variant"].to_numpy()[keep],
            score_col: raw_score[keep],
        })
        raw_tables[spec.name] = d.reset_index(drop=True)

    return variants[["gene", "uniprot_id", "protein_variant", "clnsig"]], raw_tables


# --- pre-defined scenarios for validating Step 9 operating characteristics ---
def scenario_specs(kind: str) -> list[PredictorSpec]:
    """Return predictor specs for a named validation scenario.

    The audit should report HIGH concordance for 'stable'/'dominant' and LOW
    concordance for 'gene_dependent'/'close'; 'informative_missingness' should show
    coverage sensitivity. If the audit calls everything fragile it is uninformative.
    """
    if kind == "stable":
        return [PredictorSpec("P1", "alphamissense", 3.0, 0.3, 0.05, 0.0),
                PredictorSpec("P2", "eve", 2.4, 0.3, 0.05, 0.0),
                PredictorSpec("P3", "polyphen2", 1.8, 0.3, 0.05, 0.0),
                PredictorSpec("P4", "sift", 1.2, 0.3, 0.05, 0.0),
                PredictorSpec("P5", "plm", 0.7, 0.3, 0.05, 0.0)]
    if kind == "dominant":
        return [PredictorSpec("P1", "alphamissense", 3.4, 0.4, 0.05, 0.0),
                PredictorSpec("P2", "eve", 1.6, 1.2, 0.05, 0.0),
                PredictorSpec("P3", "polyphen2", 1.5, 1.2, 0.05, 0.0),
                PredictorSpec("P4", "sift", 1.4, 1.2, 0.05, 0.0),
                PredictorSpec("P5", "plm", 1.3, 1.2, 0.05, 0.0)]
    if kind == "gene_dependent":
        return [PredictorSpec("P1", "alphamissense", 2.0, 2.2, 0.05, 0.0),
                PredictorSpec("P2", "eve", 2.0, 2.2, 0.05, 0.0),
                PredictorSpec("P3", "polyphen2", 2.0, 2.2, 0.05, 0.0),
                PredictorSpec("P4", "sift", 1.9, 2.2, 0.05, 0.0),
                PredictorSpec("P5", "plm", 1.9, 2.2, 0.05, 0.0)]
    if kind == "informative_missingness":
        return [PredictorSpec("P1", "alphamissense", 2.2, 0.6, 0.05, 0.0),
                PredictorSpec("P2", "eve", 2.1, 0.6, 0.45, 1.6),
                PredictorSpec("P3", "polyphen2", 2.0, 0.6, 0.05, 0.0),
                PredictorSpec("P4", "sift", 1.9, 0.6, 0.05, 0.0),
                PredictorSpec("P5", "plm", 2.0, 0.6, 0.05, 0.0)]
    if kind == "close":
        return [PredictorSpec("P1", "alphamissense", 2.0, 0.8, 0.05, 0.0),
                PredictorSpec("P2", "eve", 1.98, 0.8, 0.05, 0.0),
                PredictorSpec("P3", "polyphen2", 1.97, 0.8, 0.05, 0.0),
                PredictorSpec("P4", "sift", 1.96, 0.8, 0.05, 0.0),
                PredictorSpec("P5", "plm", 1.99, 0.8, 0.05, 0.0)]
    if kind == "null":            # 5 exchangeable predictors -> no true winner
        return [PredictorSpec("P1", "alphamissense", 1.9, 0.8, 0.08, 0.0),
                PredictorSpec("P2", "eve", 1.9, 0.8, 0.08, 0.0),
                PredictorSpec("P3", "polyphen2", 1.9, 0.8, 0.08, 0.0),
                PredictorSpec("P4", "sift", 1.9, 0.8, 0.08, 0.0),
                PredictorSpec("P5", "plm", 1.9, 0.8, 0.08, 0.0)]
    raise ValueError(f"unknown scenario {kind!r}")


def structural_specs(effect: float, n_groups: int = 3) -> list[PredictorSpec]:
    """P1 is the overall winner; P2 has a REPRODUCIBLE advantage in gene-group 1 of
    magnitude ``effect``. At effect 0 the ranking is genuinely stable (Type-I control);
    as effect grows, restricting to group 1 reproducibly flips the winner to P2."""
    return [
        PredictorSpec("P1", "alphamissense", 2.10, 0.30, 0.06, 0.0),
        PredictorSpec("P2", "eve", 1.90, 0.30, 0.06, 0.0, group_effect={1: effect}),
        PredictorSpec("P3", "polyphen2", 1.90, 0.30, 0.06, 0.0),
        PredictorSpec("P4", "sift", 1.85, 0.30, 0.06, 0.0),
        PredictorSpec("P5", "plm", 1.80, 0.30, 0.06, 0.0),
    ]
