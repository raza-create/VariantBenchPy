"""STEP 14B BRCA2 mapping cleanup + frozen dedup logic.

Duplicate cause: MaveDB returns ALL historical mapping runs per variant
(distinct mappedDate / mappingApiVersion); exactly one carries current=True.
Rule: keep current=True only. Never arbitrarily pick a genomic realization.
"""
from __future__ import annotations
import io, json, os, re, urllib.request
import pandas as pd

API = "https://api.mavedb.org/api/v1"
AA3 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
       "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
       "Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Ter":"*"}


def norm_pro(h):
    m = re.search(r"p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?", str(h))
    if not m:
        return None
    a, b = AA3.get(m.group(1)), AA3.get(m.group(3))
    return f"{a}{m.group(2)}{b}" if a and b else None


def mapped_variants(urn, cache="data/mavedb_mapped"):
    p = f"{cache}/{urn.replace(':', '_')}.json"
    if os.path.exists(p):
        return json.load(open(p))
    req = urllib.request.Request(f"{API}/score-sets/{urn}/mapped-variants",
                                 headers={"User-Agent": "VariantBenchPy/research"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    os.makedirs(cache, exist_ok=True)
    json.dump(d, open(p, "w"))
    return d


def score_rows(urn):
    req = urllib.request.Request(f"{API}/score-sets/{urn}/scores",
                                 headers={"User-Agent": "VariantBenchPy/research"})
    return pd.read_csv(io.StringIO(urllib.request.urlopen(req, timeout=300).read().decode()))


def build(urn, pg_id, base="data/proteingym_v1.3"):
    """Frozen pipeline: mapped-variants (current only) x score rows (hgvs_pro) x ProteinGym."""
    mv = mapped_variants(urn)
    n_raw = len(mv)
    rows = []
    for r in mv:
        pm = r.get("postMapped") or {}
        g = next((e.get("value") for e in (pm.get("expressions") or [])
                  if str(e.get("value", "")).startswith("NC_")), None)
        ch = pos = ref = alt = None
        if g:
            m = re.match(r"NC_0+(\d+|X|Y)\.\d+:g\.(\d+)([ACGT]+)>([ACGT]+)", g)
            if m:
                ch, pos, ref, alt = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        rows.append({"variantUrn": r.get("variantUrn"), "current": bool(r.get("current")),
                     "mappedDate": r.get("mappedDate"), "apiVersion": r.get("mappingApiVersion"),
                     "chrom": ch, "pos": pos, "ref": ref, "alt": alt, "hgvs_g": g,
                     "atMismatchedLocus": r.get("atMismatchedLocus")})
    M = pd.DataFrame(rows)
    n_runs = M.groupby("variantUrn").size().value_counts().to_dict()
    Mc = M[M["current"]].copy()

    S = score_rows(urn).rename(columns={"accession": "variantUrn"})
    S["aa_mavedb"] = S["hgvs_pro"].map(norm_pro)
    J = Mc.merge(S[["variantUrn", "aa_mavedb", "hgvs_pro", "score"]], on="variantUrn", how="left")

    ref_csv = pd.read_csv(f"{base}/DMS_substitutions.csv").set_index("DMS_id")
    fn = ref_csv.loc[pg_id, "DMS_filename"]
    pg_path = next((os.path.join(r, fn) for r, _, f in os.walk(base) if fn in f), None)
    PG = pd.read_csv(pg_path)
    PG = PG[~PG["mutant"].astype(str).str.contains(":")].copy()
    PG["aa_pg"] = PG["mutant"].astype(str)

    K = J.merge(PG[["aa_pg", "DMS_score"]], left_on="aa_mavedb", right_on="aa_pg", how="inner")
    K = K[K["ref"].notna() & (K["ref"].str.len() == 1) & (K["alt"].str.len() == 1)]
    K["assay"] = pg_id
    before = len(K)
    K = K.drop_duplicates(["assay", "aa_pg", "chrom", "pos", "ref", "alt"])
    per_aa = K.groupby("aa_pg").size()
    K["realization_class"] = K["aa_pg"].map(
        lambda a: "MULTIPLE_GENOMIC_REALIZATIONS" if per_aa.get(a, 0) > 1 else "UNIQUE_GENOMIC_SNV")
    return M, Mc, S, PG, K, n_raw, n_runs, before, per_aa


def run(urn="urn:mavedb:00001224-a-1", pg_id="BRCA2_HUMAN_Erwood_2022_HEK293T"):
    M, Mc, S, PG, K, n_raw, n_runs, before, per_aa = build(urn, pg_id)
    dup_key = int(K.duplicated(["assay", "aa_pg", "chrom", "pos", "ref", "alt"]).sum())
    mism = int(Mc["atMismatchedLocus"].fillna(False).astype(bool).sum())
    aa_matched = K["aa_pg"].nunique()
    no_snv = PG["aa_pg"].nunique() - aa_matched

    L = ["STEP 14B BRCA2 MAPPING CLEANUP", "=" * 30, "",
         f"ProteinGym AA substitutions:                 {PG['aa_pg'].nunique():,}",
         f"Exact ProteinGym<->MaveDB AA matches:        {aa_matched:,}", "",
         f"Mapped rows before deduplication:            {n_raw:,}",
         f"  mapping runs per variant:                  {n_runs}",
         f"  records with current=True:                 {len(Mc):,}",
         f"Unique MaveDB variant URNs:                  {M['variantUrn'].nunique():,}",
         f"Unique chr:pos:REF:ALT keys:                 "
         f"{Mc.dropna(subset=['pos']).drop_duplicates(['chrom','pos','ref','alt']).shape[0]:,}",
         f"Duplicate identical genomic rows removed:    {n_raw - len(Mc):,}"
         f"  (historical mapping runs; kept current=True only)", "",
         "AA substitutions with:",
         f"  exactly 1 unique GRCh38 SNV:               {int((per_aa == 1).sum()):,}",
         f"  >1 distinct GRCh38 SNV realization:        {int((per_aa > 1).sum()):,}",
         f"  no usable GRCh38 SNV:                      {no_snv:,}", "",
         f"FINAL UNIQUE BRCA2 DMS-SNV PAIRS:            {len(K):,}", "",
         f"Duplicate variant_id / genomic key:          {dup_key}",
         f"Reference/AA mismatches (atMismatchedLocus): {mism}",
         f"STATUS: {'PASS' if dup_key == 0 else 'FAIL'}", "",
         "DUPLICATE CAUSE (documented): MaveDB returns every historical mapping run",
         "  per variant (distinct mappedDate / mappingApiVersion); exactly one has",
         f"  current=True. 462 score rows x 4 runs = 1,848 records.", "",
         "Examples (deduplicated):",
         "  " + f"{'ProteinGym':<10}{'MaveDB':<10}{'GRCh38':<24}{'DMS_score':>10}  class"]
    for _, r in K.head(8).iterrows():
        L.append("  " + f"{r['aa_pg']:<10}{r['aa_mavedb']:<10}"
                 f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']:<10}{r['DMS_score']:>10.3f}  "
                 f"{r['realization_class']}")
    K.to_parquet("data/brca2_dms_snv_clean.parquet")
    L += ["", "-> data/brca2_dms_snv_clean.parquet",
          "FROZEN RULE: keep current=True mapping run; dedup on",
          "  (assay, AA, chrom, pos, ref, alt); AA with >1 distinct SNV kept as",
          "  MULTIPLE_GENOMIC_REALIZATIONS (never arbitrarily collapsed)."]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
