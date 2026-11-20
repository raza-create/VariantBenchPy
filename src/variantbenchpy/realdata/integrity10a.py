"""STEP 10A — real ClinVar data integrity audit (germline-only, pre-benchmark)."""
from __future__ import annotations
import gzip, re
import pandas as pd

_STARS = {
    "practice_guideline": 4, "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "no_assertion_criteria_provided": 0, "no_assertion_provided": 0,
    "no_classification_provided": 0, "no_classifications_from_unflagged_records": 0,
}
_PATH = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
_BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
_CONFLICT = {"Conflicting_classifications_of_pathogenicity",
             "Conflicting_interpretations_of_pathogenicity"}
_VUS = {"Uncertain_significance", "Uncertain_risk_allele", "not_provided"}


def _load_mane(path):
    import gzip
    genes = set()
    try:
        with gzip.open(path, "rt") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            gi = hdr.index("symbol") if "symbol" in hdr else None
            mi = hdr.index("MANE_status") if "MANE_status" in hdr else None
            for line in fh:
                c = line.rstrip("\n").split("\t")
                if mi is None or "MANE Select" in c[mi]:
                    if gi is not None and c[gi]:
                        genes.add(c[gi])
    except FileNotFoundError:
        return None
    return genes


def _info(field, key):
    m = re.search(rf"(?:^|;){key}=([^;]*)", field)
    return m.group(1) if m else ""


def audit(vcf_gz, release="unknown", mane_path="data/MANE.summary.txt.gz"):
    mane = _load_mane(mane_path)
    rows = []
    bucket = {"path": 0, "benign": 0, "conflict": 0, "vus": 0, "other": 0, "missing": 0}
    somatic_or_onco = 0
    with gzip.open(vcf_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            chrom, pos, _id, ref, alt, _q, _fl, info = f[:8]
            if "missense_variant" not in _info(info, "MC"):
                continue
            # germline-only: skip somatic/oncogenicity assertions
            if _info(info, "ONcClassification") or _info(info, "ONCDN") or _info(info, "SCIDN"):
                somatic_or_onco += 1
            clnsig = _info(info, "CLNSIG")
            if clnsig in _PATH:
                label, bkey = 1, "path"
            elif clnsig in _BENIGN:
                label, bkey = 0, "benign"
            elif clnsig in _CONFLICT:
                bucket["conflict"] += 1; continue
            elif clnsig in _VUS:
                bucket["vus"] += 1; continue
            elif clnsig == "":
                bucket["missing"] += 1; continue
            else:
                bucket["other"] += 1; continue
            bucket[bkey] += 1
            gene = _info(info, "GENEINFO")
            genes = [g.split(":")[0] for g in gene.split("|") if g]
            onc = bool(_info(info, "ONCDN")); sci = bool(_info(info, "SCIDN"))
            if mane:
                primary = next((g for g in genes if g in mane), genes[0] if genes else "")
                mane_resolved = bool(genes) and primary in mane
            else:
                primary = genes[0] if genes else ""
                mane_resolved = False
            rows.append({"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
                         "gene": primary, "all_genes": "|".join(genes),
                         "mane_resolved": mane_resolved,
                         "onc": onc, "sci": sci, "somatic_flag": onc or sci,
                         "n_genes": len(genes),
                         "label": label, "alleleid": _info(info, "ALLELEID"),
                         "variationid": _id,
                         "stars": _STARS.get(_info(info, "CLNREVSTAT"), 0),
                         "key": f"{chrom}-{pos}-{ref}-{alt}"})
    df = pd.DataFrame(rows)
    uniq = df.drop_duplicates("key")
    per_gene = df.groupby("gene")["label"]
    n_var = per_gene.count()
    both = per_gene.nunique() == 2

    L = ["STEP 10A — REAL DATA INTEGRITY", "=" * 30, "",
         f"ClinVar release: {release}   Genome build: GRCh38", "",
         f"Labeled P/LP or B/LB missense rows:   {len(df):,}",
         f"Unique genomic variants (chr:pos:ref:alt): {uniq.shape[0]:,}",
         f"Duplicate genomic keys:               {len(df) - uniq.shape[0]:,}",
         f"Unique ClinVar AlleleIDs:             {df['alleleid'].nunique():,}",
         f"Unique ClinVar VariationIDs:          {df['variationid'].nunique():,}",
         f"Variants mapping to >1 gene:          {(df['n_genes'] > 1).sum():,}",

         "Excluded (not in benchmark):",
         f"  VUS / not_provided:        {bucket['vus']:,}",
         f"  Conflicting P/B:           {bucket['conflict']:,}",
         f"  Other (risk/drug/assoc):   {bucket['other']:,}",
         f"  Missing classification:    {bucket['missing']:,}", "",
         f"Class balance:  Pathogenic {bucket['path']:,} ({bucket['path']/len(df):.1%})  |  "
         f"Benign {bucket['benign']:,} ({bucket['benign']/len(df):.1%})", "",
         f"Genes (any labeled variant):     {df['gene'].nunique():,}",
         f"  >=5 variants:   {(n_var >= 5).sum():,}",
         f"  >=10 variants:  {(n_var >= 10).sum():,}",
         f"  >=20 variants:  {(n_var >= 20).sum():,}",
         f"  both classes present: {both.sum():,}",
         f"  both classes AND >=10 variants: {(both & (n_var >= 10)).sum():,}", "",
         "Review status (stars):"]
    for s, n in uniq["stars"].value_counts().sort_index().items():
        L.append(f"  {s}-star: {n:,}")
    ge = both & (n_var >= 10)
    s1 = uniq[uniq["stars"] >= 1]; s2 = uniq[uniq["stars"] >= 2]
    multi = uniq[uniq["n_genes"] > 1]
    onc_n = int(uniq["onc"].sum()); sci_n = int(uniq["sci"].sum())
    both_os = int((uniq["onc"] & uniq["sci"]).sum()); union_os = int(uniq["somatic_flag"].sum())
    mane_n = int(uniq["mane_resolved"].sum())
    L += ["", "ANALYSIS POPULATIONS (prespecified)",
          f"  Variant-level:               {uniq.shape[0]:,} variants / {df['gene'].nunique():,} primary genes",
          f"  Gene-aware clustered:        {uniq.shape[0]:,} variants / {uniq['gene'].nunique():,} primary-gene clusters",
          f"  Within-gene AUROC (>=10, both classes): {ge.sum():,} genes",
          f"  High-evidence >=1-star:      {s1.shape[0]:,} variants",
          f"  High-evidence >=2-star:      {s2.shape[0]:,} variants",
          f"  Multi-gene sensitivity (exclude {multi.shape[0]:,}): {uniq.shape[0]-multi.shape[0]:,} variants",
          f"  Somatic/onco sensitivity (exclude {union_os:,}): {uniq.shape[0]-union_os:,} variants",
          f"  GENEINFO-fallback sensitivity (exclude {uniq.shape[0]-mane_n:,}): {mane_n:,} variants",
          "",
          "Somatic / oncogenicity annotations (single definition: ONCDN and/or SCIDN):",
          f"  ONC (ONCDN) variants:  {onc_n:,}",
          f"  SCI (SCIDN) variants:  {sci_n:,}",
          f"  Both ONC and SCI:      {both_os:,}",
          f"  Unique ONC or SCI:     {union_os:,}",
          "",
          "PRIMARY-GENE ASSIGNMENT (frozen rule; MANE-informed, gene-name level)",
          f"  MANE-informed assignment: {mane_n:,} ({mane_n/uniq.shape[0]:.1%})",
          f"  GENEINFO fallback:        {uniq.shape[0]-mane_n:,} ({(uniq.shape[0]-mane_n)/uniq.shape[0]:.1%})",
          f"  Unresolved primary:       {int((uniq['gene']=='').sum()):,}",
          "  NOTE: ClinVar VCF exposes no MANE transcript field; assignment is by MANE",
          "        Select GENE NAME, not variant-level MANE transcript consequence.",
          "  Rule: (1) GENEINFO gene that is a MANE Select gene; (2) else first GENEINFO",
          "        gene; (3) all_genes kept in metadata; (4) multi-gene and (5) GENEINFO-",
          "        fallback variants each excluded in separate sensitivity analyses.",
          "",
          "", "NOTE: labels are ClinVar germline aggregate classification only.",
          "NEXT: freeze transcript rule (MANE/canonical) before dbNSFP; keep EVE separate from popEVE."]
    df.attrs["report"] = "\n".join(L)
    return df, uniq, "\n".join(L)


if __name__ == "__main__":
    import sys
    _, _, rep = audit(sys.argv[1], release=sys.argv[2] if len(sys.argv) > 2 else "unknown")
    print(rep)
