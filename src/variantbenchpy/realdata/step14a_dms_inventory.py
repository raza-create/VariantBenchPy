"""STEP 14A — external DMS inventory (ProteinGym v1.3). No predictor scoring."""
from __future__ import annotations
import glob, os, re
import numpy as np
import pandas as pd

CODON = {}
_B = "TCAG"
_AA = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
for i, b1 in enumerate(_B):
    for j, b2 in enumerate(_B):
        for k, b3 in enumerate(_B):
            CODON[b1 + b2 + b3] = _AA[i * 16 + j * 4 + k]
AA2COD = {}
for c, a in CODON.items():
    AA2COD.setdefault(a, []).append(c)

INCLUSION = [
    "Human proteins only (taxon == Human)",
    "ProteinGym substitution benchmark v1.3",
    "single amino-acid substitutions only (no multi-mutants in primary analysis)",
    "numeric experimental DMS_score required",
    "identifiable UniProt protein + reference target sequence",
    "no indels; no imputed experimental labels",
    "orientation: DMS_deleterious = -DMS_score (ProteinGym higher = fitter)",
    "primary metric: Spearman vs experimental effect (not AUROC)",
]


def _n_snv_realizations(aa_ref, aa_alt):
    """Minimum nucleotide changes between any ref-codon and any alt-codon,
    and how many distinct single-nucleotide routes exist."""
    best, routes = 3, 0
    for cr in AA2COD.get(aa_ref, []):
        for ca in AA2COD.get(aa_alt, []):
            d = sum(1 for x, y in zip(cr, ca) if x != y)
            if d < best:
                best, routes = d, 0
            if d == best == 1:
                routes += 1
    return best, routes


def classify(aa_ref, aa_alt, seq_ok):
    if not seq_ok:
        return "REFERENCE_MISMATCH"
    if aa_ref not in AA2COD or aa_alt not in AA2COD:
        return "NO_MAPPING"
    d, routes = _n_snv_realizations(aa_ref, aa_alt)
    if d > 1:
        return "CODON_MULTI_BASE_REQUIRED"
    return "CODON_UNIQUE_SINGLE_BASE_CHANGE" if routes == 1 else "CODON_MULTIPLE_SINGLE_BASE_CHANGES"


def run(base="data/proteingym_v1.3"):
    ref = pd.read_csv(os.path.join(base, "DMS_substitutions.csv"))
    human = ref[ref["taxon"].astype(str).str.lower() == "human"].copy()
    files = {os.path.basename(f): f for f in
             glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)}

    rows, per_assay = [], []
    for _, r in human.iterrows():
        fn = r["DMS_filename"]
        path = files.get(fn)
        if path is None:
            per_assay.append({**r[["DMS_id", "UniProt_ID"]].to_dict(),
                              "status": "FILE_MISSING", "n_single": 0, "n_numeric": 0})
            continue
        d = pd.read_csv(path)
        mcol = "mutant" if "mutant" in d.columns else d.columns[0]
        scol = "DMS_score" if "DMS_score" in d.columns else None
        single = d[~d[mcol].astype(str).str.contains(":")]
        seq = str(r["target_seq"])
        cats = []
        for mut in single[mcol].astype(str):
            m = re.fullmatch(r"([A-Z])(\d+)([A-Z*])", mut)
            if not m:
                cats.append("NO_MAPPING"); continue
            a, pos, b = m.group(1), int(m.group(2)), m.group(3)
            seq_ok = 1 <= pos <= len(seq) and seq[pos - 1] == a
            cats.append(classify(a, b, seq_ok))
        vc = pd.Series(cats).value_counts()
        n_num = int(pd.to_numeric(single[scol], errors="coerce").notna().sum()) if scol else 0
        per_assay.append({
            "DMS_id": r["DMS_id"], "UniProt_ID": r["UniProt_ID"],
            "molecule": r.get("molecule_name"), "pub": f"{r.get('first_author')} {r.get('year')}",
            "assay": r.get("selection_assay"), "selection": r.get("coarse_selection_type"),
            "n_total": int(r.get("DMS_total_number_mutants", len(d))),
            "n_single": int(len(single)), "n_numeric": n_num, "seq_len": int(r.get("seq_len", 0)),
            "unambiguous": int(vc.get("CODON_UNIQUE_SINGLE_BASE_CHANGE", 0)),
            "multi_realization": int(vc.get("CODON_MULTIPLE_SINGLE_BASE_CHANGES", 0)),
            "multi_nt": int(vc.get("CODON_MULTI_BASE_REQUIRED", 0)),
            "no_mapping": int(vc.get("NO_MAPPING", 0)),
            "ref_mismatch": int(vc.get("REFERENCE_MISMATCH", 0)),
            "status": "OK"})
        rows.extend(cats)

    A = pd.DataFrame(per_assay)
    A.to_parquet("data/dms_inventory_14a.parquet")
    tot = pd.Series(rows).value_counts()
    n_single = int(A["n_single"].sum())
    cand = A[(A["status"] == "OK") & (A["unambiguous"] > 0)]

    L = ["STEP 14A — EXTERNAL DMS INVENTORY", "=" * 33, "",
         "ProteinGym v1.3",
         f"Total assays:            {len(ref)}",
         f"Human assays:            {len(human)}",
         f"Unique human proteins:   {human['UniProt_ID'].nunique()}",
         f"Assay files read:        {int((A['status']=='OK').sum())}"
         f"  (missing: {int((A['status']!='OK').sum())})", "",
         f"Single-AA variants:      {n_single:,}",
         f"With numeric DMS_score:  {int(A['n_numeric'].sum()):,}", "",
         "Codon reachability (amino-acid level; NOT yet genomic mapping):"]
    for k in ["CODON_UNIQUE_SINGLE_BASE_CHANGE", "CODON_MULTIPLE_SINGLE_BASE_CHANGES",
              "CODON_MULTI_BASE_REQUIRED", "NO_MAPPING", "REFERENCE_MISMATCH"]:
        n = int(tot.get(k, 0))
        L.append(f"  {k:<34}{n:>9,}  ({n/max(n_single,1):.1%})")
    L += ["", f"Candidate assays remaining:   {len(cand)}",
          f"Candidate unique proteins:    {cand['UniProt_ID'].nunique()}",
          f"Codon-unique single-base variants: {int(cand['unambiguous'].sum()):,}",
          "  (codon reachability only - transcript/GRCh38/REF/ALT not yet established)",
          "", "MaveDB mapping: not yet queried (Step 14B)",
          "", "FROZEN INCLUSION RULES:"]
    L += [f"  - {r}" for r in INCLUSION]
    L += ["", "Per-assay table -> data/dms_inventory_14a.parquet",
          "", "Top 10 candidate assays by unambiguous SNV count:",
          cand.nlargest(10, "unambiguous")[
              ["DMS_id", "UniProt_ID", "selection", "n_single", "unambiguous",
               "multi_realization", "multi_nt"]].to_string(index=False)]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
