"""Step 10 — real ClinVar GRCh38 missense ingestion (integrity-first)."""
from __future__ import annotations
import gzip, re
import pandas as pd

_STARS = {
    "practice_guideline": 4, "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_conflicting_interpretations": 1,
    "no_assertion_criteria_provided": 0, "no_assertion_provided": 0,
    "no_classification_provided": 0, "no_classifications_from_unflagged_records": 0,
}
_PATH = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
_BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}


def _info(field, key):
    m = re.search(rf"(?:^|;){key}=([^;]*)", field)
    return m.group(1) if m else ""


def parse_clinvar_missense(vcf_gz, max_records=None):
    rows, counts = [], {"total": 0, "missense": 0, "labeled": 0}
    with gzip.open(vcf_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            counts["total"] += 1
            if max_records and counts["total"] > max_records:
                break
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            chrom, pos, _id, ref, alt, _q, _fl, info = f[:8]
            if "missense_variant" not in _info(info, "MC"):
                continue
            counts["missense"] += 1
            clnsig = _info(info, "CLNSIG")
            if clnsig in _PATH:
                label = 1
            elif clnsig in _BENIGN:
                label = 0
            else:
                continue
            counts["labeled"] += 1
            _gi = _info(info, "GENEINFO")
            _ng = len([g for g in _gi.split("|") if g])
            rows.append({"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
                         "gene": _gi.split(":")[0].split("|")[0], "n_genes": _ng,
                         "clnsig": clnsig, "label": label,
                         "review_status": _info(info, "CLNREVSTAT"),
                         "stars": _STARS.get(_info(info, "CLNREVSTAT"), 0),
                         "variant_id": f"{chrom}-{pos}-{ref}-{alt}"})
    df = pd.DataFrame(rows).drop_duplicates(["chrom", "pos", "ref", "alt"]).reset_index(drop=True)
    df.attrs["counts"] = counts
    return df
