"""VariantSet — the central data object the whole library revolves around.

A VariantSet holds, in one place: the harmonized variants (coordinates, gene,
transcript, protein change, clinical label, allele frequency, consequence), an oriented
predictor score matrix, and provenance (source, build, release, processing history). It
validates itself, filters cleanly into new VariantSets, summarises itself, and hands the
same object to every analysis module so nothing has to be re-wired per method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from variantbenchpy.harmonize.harmonizer import VariantHarmonizer
from variantbenchpy.harmonize.schema import (
    CanonicalKey, COL_ALT, COL_BUILD, COL_CHROM, COL_GENE, COL_LABEL,
    COL_POS, COL_PROTEIN_ID, COL_AA_POS, COL_REF, COL_SOURCE_ROW, COL_VARIANT_ID,
)
from variantbenchpy.io_formats import read_table, read_vcf
from variantbenchpy.predictors.base import Predictor, ScoreMatrix, ScoreOrientation

COL_ALLELE_FREQ = "allele_frequency"
COL_CONSEQUENCE = "consequence"
COL_REVIEW = "review_status"
COL_TRANSCRIPT = "transcript"
_EXTRA_FIELDS = (COL_ALLELE_FREQ, COL_CONSEQUENCE, COL_REVIEW, COL_TRANSCRIPT)

_PREDICTOR_HINTS = {
    "am_pathogenicity": ("AlphaMissense", False, 0.564),
    "alphamissense": ("AlphaMissense", False, 0.564),
    "am_score": ("AlphaMissense", False, 0.564),
    "eve_scores": ("EVE", False, 0.5), "eve": ("EVE", False, 0.5),
    "eve_score": ("EVE", False, 0.5),
    "revel": ("REVEL", False, 0.5), "revel_score": ("REVEL", False, 0.5),
    "cadd_phred": ("CADD", False, 20.0), "phred": ("CADD", False, 20.0),
    "cadd": ("CADD", False, 20.0),
    "sift_score": ("SIFT", True, 0.05), "sift": ("SIFT", True, 0.05),
    "pph2_prob": ("PolyPhen-2", False, 0.5), "polyphen": ("PolyPhen-2", False, 0.5),
    "polyphen2": ("PolyPhen-2", False, 0.5), "pph2": ("PolyPhen-2", False, 0.5),
}


@dataclass
class ValidationIssue:
    level: str
    check: str
    count: int
    detail: str = ""


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(i.level == "error" and i.count > 0 for i in self.issues):
            return "FAIL"
        if any(i.level == "warning" and i.count > 0 for i in self.issues):
            return "PASS WITH WARNINGS"
        return "PASS"

    def __str__(self) -> str:
        lines = ["Data quality"]
        for i in self.issues:
            mark = {"error": "\u2717", "warning": "\u26a0", "info": " "}[i.level]
            lines.append(f"  {mark} {i.check:<28} {i.count}"
                         + (f"   ({i.detail})" if i.detail else ""))
        lines.append(f"Overall data status: {self.status}")
        return "\n".join(lines)


class VariantSet:
    """One object holding variants, predictor scores, and provenance."""

    def __init__(self, variants: pd.DataFrame, score_matrix: ScoreMatrix | None,
                 metadata: dict, history: list[str], harmonization_report=None):
        self.variants = variants
        self.score_matrix = score_matrix
        self.metadata = metadata
        self.history = history
        self.harmonization_report = harmonization_report

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, column_map: dict[str, str],
                       key: CanonicalKey = CanonicalKey.PROTEIN,
                       extras: dict[str, str] | None = None,
                       predictor_columns="auto",
                       metadata: dict | None = None) -> "VariantSet":
        df = df.reset_index(drop=True)
        harmonizer = VariantHarmonizer(key=key)
        variants, report = harmonizer.harmonize(df, column_map)

        extras = extras or {}
        for canon, src in extras.items():
            if src in df.columns:
                variants[canon] = variants[COL_SOURCE_ROW].map(df[src])
        for c in _EXTRA_FIELDS:
            if c not in variants.columns:
                variants[c] = pd.NA
        if COL_ALLELE_FREQ in variants.columns:
            variants[COL_ALLELE_FREQ] = pd.to_numeric(variants[COL_ALLELE_FREQ], errors="coerce")

        used_cols = set(column_map.values()) | set(extras.values())
        detected = cls._detect_predictors(df, predictor_columns, used_cols)
        sm = cls._build_score_matrix(variants, df, detected, key) if detected else None

        meta = {"source": None, "genome_build": None, "release_date": None,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        meta.update(metadata or {})
        builds = variants[COL_BUILD].dropna().unique() if COL_BUILD in variants else []
        if meta["genome_build"] is None and len(builds) == 1:
            meta["genome_build"] = str(builds[0])

        hist = [f"loaded {len(df)} rows -> harmonized {report.n_output} variants "
                f"({report.n_duplicates_removed} dupes removed)"]
        if detected:
            hist.append(f"detected {len(detected)} predictor score columns: "
                        + ", ".join(sorted({v[0] for v in detected.values()})))
        return cls(variants, sm, meta, hist, report)

    @classmethod
    def from_csv(cls, path: str, column_map, **kw) -> "VariantSet":
        vs = cls.from_dataframe(read_table(path, sep=","), column_map, **kw)
        vs.metadata["source"] = path
        vs.history.insert(0, f"read CSV {path}")
        return vs

    @classmethod
    def from_tsv(cls, path: str, column_map, **kw) -> "VariantSet":
        vs = cls.from_dataframe(read_table(path, sep="\t"), column_map, **kw)
        vs.metadata["source"] = path
        vs.history.insert(0, f"read TSV {path}")
        return vs

    @classmethod
    def from_vcf(cls, path: str, info_fields: dict[str, str] | None = None,
                 column_map: dict[str, str] | None = None,
                 key: CanonicalKey = CanonicalKey.GENOMIC, **kw) -> "VariantSet":
        info_fields = info_fields or {"clinical_significance": "CLNSIG",
                                      "gene": "GENEINFO", "allele_frequency": "AF"}
        df = read_vcf(path, info_fields=info_fields)
        cmap = column_map or {"chrom": "chrom", "pos": "pos", "ref": "ref", "alt": "alt",
                              "gene": "gene", "clinical_significance": "clinical_significance"}
        extras = {COL_ALLELE_FREQ: "allele_frequency"} if "allele_frequency" in df.columns else None
        vs = cls.from_dataframe(df, cmap, key=key, extras=extras, **kw)
        vs.metadata["source"] = path
        vs.history.insert(0, f"read VCF {path}")
        return vs

    @staticmethod
    def _detect_predictors(df, predictor_columns, used_cols):
        detected: dict[str, tuple[str, bool, float | None]] = {}
        if predictor_columns == "auto":
            for col in df.columns:
                if col in used_cols:
                    continue
                hint = _PREDICTOR_HINTS.get(str(col).strip().lower())
                if hint:
                    detected[col] = hint
        elif isinstance(predictor_columns, dict):
            for col, spec in predictor_columns.items():
                detected[col] = spec if isinstance(spec, tuple) else (spec, False, None)
        elif isinstance(predictor_columns, (list, tuple)):
            for col in predictor_columns:
                detected[col] = _PREDICTOR_HINTS.get(str(col).strip().lower(),
                                                     (str(col), False, None))
        return detected

    @staticmethod
    def _build_score_matrix(variants, df, detected, key) -> ScoreMatrix:
        ids = variants[COL_VARIANT_ID]
        scores = pd.DataFrame(index=pd.Index(ids, name=COL_VARIANT_ID))
        predictors: dict[str, Predictor] = {}
        for src_col, (name, lower, thr) in detected.items():
            raw = pd.to_numeric(df[src_col], errors="coerce")
            aligned = variants[COL_SOURCE_ROW].map(raw).to_numpy()
            oriented = -aligned if lower else aligned
            scores[name] = oriented
            predictors[name] = Predictor(
                name=name, key=key,
                orientation=ScoreOrientation.LOWER_IS_DAMAGING if lower
                else ScoreOrientation.HIGHER_IS_DAMAGING,
                score_column=src_col, column_map={}, decision_threshold=thr)
        scores = scores.astype("float64")
        return ScoreMatrix(table=variants.reset_index(drop=True), scores=scores, predictors=predictors)

    def validate(self) -> ValidationReport:
        v = self.variants
        issues = []
        dup = self.harmonization_report.n_duplicates_removed if self.harmonization_report else 0
        issues.append(ValidationIssue("warning", "Duplicate variants", int(dup)))
        if COL_POS in v:
            pos = pd.to_numeric(v[COL_POS], errors="coerce")
            issues.append(ValidationIssue("error", "Invalid coordinates", int((pos <= 0).sum())))
        if COL_REF in v and COL_ALT in v:
            both = v[COL_REF].notna() & v[COL_ALT].notna()
            issues.append(ValidationIssue("error", "REF == ALT",
                                          int((both & (v[COL_REF] == v[COL_ALT])).sum())))
        if COL_AA_POS in v:
            ap = pd.to_numeric(v[COL_AA_POS], errors="coerce")
            issues.append(ValidationIssue("error", "Invalid protein positions", int((ap <= 0).sum())))
        issues.append(ValidationIssue("warning", "Missing gene",
                                      int(v[COL_GENE].isna().sum()) if COL_GENE in v else 0))
        if COL_PROTEIN_ID in v:
            issues.append(ValidationIssue("warning", "Missing protein mapping",
                                          int(v[COL_PROTEIN_ID].isna().sum())))
        issues.append(ValidationIssue("error", "Duplicate variant IDs",
                                      int(v[COL_VARIANT_ID].duplicated().sum())))
        if COL_BUILD in v:
            nb = v[COL_BUILD].dropna().nunique()
            issues.append(ValidationIssue("error", "Inconsistent genome build",
                                          0 if nb <= 1 else nb,
                                          detail="; ".join(map(str, v[COL_BUILD].dropna().unique()))
                                          if nb > 1 else ""))
        return ValidationReport(issues)

    def _clone(self, mask: pd.Series, note: str) -> "VariantSet":
        new_vars = self.variants[mask.values].reset_index(drop=True)
        sm = None
        if self.score_matrix is not None:
            keep_ids = new_vars[COL_VARIANT_ID]
            new_scores = self.score_matrix.scores.loc[
                self.score_matrix.scores.index.isin(keep_ids)]
            sm = ScoreMatrix(table=new_vars, scores=new_scores,
                             predictors=self.score_matrix.predictors)
        return VariantSet(new_vars, sm, dict(self.metadata),
                          self.history + [f"filter: {note} -> {int(mask.sum())} variants"],
                          self.harmonization_report)

    def filter_by_gene(self, genes) -> "VariantSet":
        genes = {genes} if isinstance(genes, str) else set(genes)
        return self._clone(self.variants[COL_GENE].isin(genes), f"gene in {sorted(genes)}")

    def filter_by_class(self, keep="labeled") -> "VariantSet":
        lab = self.variants[COL_LABEL]
        if keep == "pathogenic":
            m = lab == 1
        elif keep == "benign":
            m = lab == 0
        else:
            m = lab.notna()
        return self._clone(m, f"class={keep}")

    def filter_by_allele_frequency(self, max_af=None, min_af=None) -> "VariantSet":
        af = pd.to_numeric(self.variants.get(COL_ALLELE_FREQ), errors="coerce")
        m = pd.Series(True, index=self.variants.index)
        if max_af is not None:
            m &= af <= max_af
        if min_af is not None:
            m &= af >= min_af
        return self._clone(m, f"AF in [{min_af},{max_af}]")

    def filter_by_consequence(self, consequences) -> "VariantSet":
        cons = {consequences} if isinstance(consequences, str) else set(consequences)
        return self._clone(self.variants.get(COL_CONSEQUENCE,
                           pd.Series(index=self.variants.index)).isin(cons),
                           f"consequence in {sorted(cons)}")

    def summary(self) -> str:
        v = self.variants
        lab = v[COL_LABEL]
        n_path = int((lab == 1).sum()); n_ben = int((lab == 0).sum())
        n_other = int(lab.isna().sum())
        preds = self.score_matrix.predictor_names if self.score_matrix else []
        lines = ["VARIANT DATA SUMMARY", "=" * 20,
                 f"Variants: {len(v):,}",
                 f"Genes: {v[COL_GENE].nunique(dropna=True):,}",
                 f"Genome build: {self.metadata.get('genome_build') or 'unknown'}",
                 "Clinical labels",
                 f"  Pathogenic: {n_path:,}",
                 f"  Benign: {n_ben:,}",
                 f"  Other: {n_other:,}",
                 f"Predictors detected: {len(preds)}"]
        lines += [f"  {p}" for p in preds]
        lines.append(str(self.validate()))
        return "\n".join(lines)

    def benchmark_ready(self) -> str:
        preds = self.score_matrix.predictor_names if self.score_matrix else []
        lines = ["BENCHMARK READY", "=" * 15,
                 f"Variants available: {len(self.variants):,}",
                 f"Genes available: {self.variants[COL_GENE].nunique(dropna=True):,}",
                 f"Predictors available: {len(preds)}",
                 "Ready for:"]
        checks = ["coverage analysis", "missingness audit", "gene-clustered bootstrap",
                  "complete-case analysis", "ranking analysis"]
        ok = self.score_matrix is not None and len(preds) >= 2
        for c in checks:
            lines.append(f"  {'\u2713' if ok else '\u2717'} {c}")
        return "\n".join(lines)

    def _require_scores(self) -> ScoreMatrix:
        if self.score_matrix is None or len(self.score_matrix.predictor_names) == 0:
            raise ValueError("No predictor scores attached to this VariantSet.")
        return self.score_matrix

    def coverage(self):
        from variantbenchpy.coverage import analyze_coverage
        return analyze_coverage(self._require_scores())

    def evaluate(self, **kw):
        from variantbenchpy.metrics import evaluate_all
        return evaluate_all(self._require_scores(), **kw)

    def missingness_audit(self, **kw):
        from variantbenchpy.bias import missingness_audit, missingness_verdict
        audit = missingness_audit(self._require_scores(), **kw)
        return audit, missingness_verdict(audit)

    def complete_case(self, **kw):
        from variantbenchpy.bias import complete_case_summary
        return complete_case_summary(self._require_scores(), **kw)

    def ranking_fragility(self, gene_groups=None):
        from variantbenchpy.evaluation import benchmark_configurations, ranking_fragility
        _, ranks = benchmark_configurations(self._require_scores(), gene_groups=gene_groups)
        return ranking_fragility(ranks)

    def probability_best(self, **kw):
        from variantbenchpy.evaluation import probability_best
        return probability_best(self._require_scores(), **kw)

    def pairwise_delta(self, **kw):
        from variantbenchpy.evaluation import pairwise_delta_auroc
        return pairwise_delta_auroc(self._require_scores(), **kw)

    def fragility_report(self, gene_groups=None, subset_masks=None, n_boot=1000, seed=0):
        from variantbenchpy.evaluation import ranking_fragility_report
        return ranking_fragility_report(self._require_scores(), gene_groups=gene_groups,
                                        subset_masks=subset_masks, n_boot=n_boot, seed=seed)

    def reliability_audit(self, gene_groups=None, evidence_masks=None, metric="auroc",
                          n_boot_w=200, n_boot_prob=1000, seed=0):
        from variantbenchpy.evaluation import rank_reliability_audit
        return rank_reliability_audit(self._require_scores(), gene_groups=gene_groups,
                                      evidence_masks=evidence_masks, metric=metric,
                                      n_boot_w=n_boot_w, n_boot_prob=n_boot_prob, seed=seed)

    def __repr__(self) -> str:
        np_ = len(self.score_matrix.predictor_names) if self.score_matrix else 0
        return (f"VariantSet(variants={len(self.variants):,}, "
                f"genes={self.variants[COL_GENE].nunique(dropna=True)}, predictors={np_})")
