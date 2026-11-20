"""STEP 14B join diagnostic — BRCA2 (urn:mavedb:00001224-a-1).

Chain: mapped-variant (genomic, keyed by variantUrn)
       -> score row accession (hgvs_pro)  -> normalized AA  -> ProteinGym mutant.
"""
from __future__ import annotations
import io, json, os, re, urllib.request
import pandas as pd

API = "https://api.mavedb.org/api/v1"
URN = "urn:mavedb:00001224-a-1"
PG_ID = "BRCA2_HUMAN_Erwood_2022_HEK293T"
AA3 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
       "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
       "Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Ter":"*"}


def norm_pro(h):
    """p.Ala123Val -> A123V"""
    m = re.search(r"p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?", str(h))
    if not m:
        return None
    a, b = AA3.get(m.group(1)), AA3.get(m.group(3))
    return f"{a}{m.group(2)}{b}" if a and b else None


def _mapped():
    p = f"data/mavedb_mapped/{URN.replace(':', '_')}.json"
    if os.path.exists(p):
        return json.load(open(p))
    req = urllib.request.Request(f"{API}/score-sets/{URN}/mapped-variants",
                                 headers={"User-Agent": "VariantBenchPy/research"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    os.makedirs("data/mavedb_mapped", exist_ok=True)
    json.dump(d, open(p, "w"))
    return d


def _scores():
    req = urllib.request.Request(f"{API}/score-sets/{URN}/scores",
                                 headers={"User-Agent": "VariantBenchPy/research"})
    return pd.read_csv(io.StringIO(urllib.request.urlopen(req, timeout=300).read().decode()))


def run(base="data/proteingym_v1.3"):
    mv = _mapped()
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
        rows.append({"variantUrn": r.get("variantUrn"), "chrom": ch, "pos": pos,
                     "ref": ref, "alt": alt, "hgvs_g": g})
    M = pd.DataFrame(rows)

    S = _scores().rename(columns={"accession": "variantUrn"})
    S["aa_mavedb"] = S["hgvs_pro"].map(norm_pro)

    J = M.merge(S[["variantUrn", "aa_mavedb", "hgvs_pro", "score"]], on="variantUrn", how="left")

    fn = pd.read_csv(f"{base}/DMS_substitutions.csv").set_index("DMS_id").loc[PG_ID, "DMS_filename"]
    pg_path = next((os.path.join(r, fn) for r, _, f in os.walk(base) if fn in f), None)
    PG = pd.read_csv(pg_path)
    PG = PG[~PG["mutant"].astype(str).str.contains(":")]
    PG["aa_pg"] = PG["mutant"].astype(str)

    K = J.merge(PG[["aa_pg", "DMS_score"]], left_on="aa_mavedb", right_on="aa_pg", how="left")
    snv = K["ref"].notna() & (K["ref"].str.len() == 1) & (K["alt"].str.len() == 1)
    both = K["aa_pg"].notna() & snv

    L = ["STEP 14B JOIN DIAGNOSTIC — BRCA2", "=" * 32, "",
         f"score set: {URN}   ProteinGym assay: {PG_ID}", "",
         f"MaveDB mapped variants:              {len(M):,}",
         f"  with variant URN:                  {int(M['variantUrn'].notna().sum()):,}",
         f"  with GRCh38 genomic expression:    {int(M['hgvs_g'].notna().sum()):,}",
         f"MaveDB score rows:                   {len(S):,}",
         f"matched to MaveDB score rows:        {int(J['aa_mavedb'].notna().sum()):,}", "",
         f"MaveDB protein substitutions normalized: {int(S['aa_mavedb'].notna().sum()):,}",
         f"ProteinGym substitutions:                {len(PG):,}", "",
         f"Exact AA matches:                    {int(K['aa_pg'].notna().sum()):,}",
         f"Exact AA match + exact GRCh38 SNV:   {int(both.sum()):,}", "",
         "Examples:",
         "  " + f"{'ProteinGym':<12}{'MaveDB':<12}{'GRCh38 chr:pos:ref:alt':<28}{'DMS_score':>10}"]
    for _, r in K[both].head(10).iterrows():
        L.append("  " + f"{r['aa_pg']:<12}{r['aa_mavedb']:<12}"
                 f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']:<12}{r['DMS_score']:>10.3f}")
    K.to_parquet("data/brca2_join_diagnostic.parquet")
    L += ["", "-> data/brca2_join_diagnostic.parquet",
          "", "KEY FINDING: mapped-variants carry genomic coordinates ONLY;",
          "  the amino-acid change comes from the score rows (hgvs_pro) joined by",
          "  accession == variantUrn. The earlier 0-join was caused by looking for",
          "  protein expressions inside the mapped-variant records."]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
