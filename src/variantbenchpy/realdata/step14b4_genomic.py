"""STEP 14B-part2 — retrieve MaveDB mapped variants for the 11 frozen score sets.

MaveDB postMapped VRS is protein-level for protein-target score sets and
genomic (NC_*) only for DNA-target score sets whose mapping completed.
Classification is based on the actual postMapped expression, not assumption.
"""
from __future__ import annotations
import json, os, re, time, urllib.request
import pandas as pd

API = "https://api.mavedb.org/api/v1"
CACHE = "data/mavedb_mapped"
CATS = ["EXACT_GRCH38_SNV", "MULTI_NUCLEOTIDE_VARIANT", "INDEL_OR_OTHER",
        "MAPPING_UNAVAILABLE", "REFERENCE_MISMATCH", "AMBIGUOUS_DMS_JOIN"]
AA3 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
       "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
       "Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Ter":"*"}


def _fetch(urn, timeout=600):
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{urn.replace(':', '_')}.json"
    if os.path.exists(path):
        return json.load(open(path)), "cached"
    req = urllib.request.Request(f"{API}/score-sets/{urn}/mapped-variants",
                                 headers={"User-Agent": "VariantBenchPy/research"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        json.dump(d, open(path, "w"))
        return d, "ok"
    except Exception as e:
        return None, f"FAIL:{type(e).__name__}"


def _parse(rec):
    """Return (chrom, pos, ref, alt, category, hgvs) from a mapped-variant record."""
    pm = rec.get("postMapped")
    if not pm:
        return (None,) * 4 + ("MAPPING_UNAVAILABLE", None)
    exprs = [e.get("value", "") for e in (pm.get("expressions") or [])]
    hg = next((e for e in exprs if e.startswith("NC_")), None)
    if hg is None:
        # protein-level mapping only (protein-target score set)
        return (None,) * 4 + ("MAPPING_UNAVAILABLE", exprs[0] if exprs else None)
    m = re.match(r"(NC_0+(\d+|X|Y)\.\d+):g\.(\d+)([ACGT]+)>([ACGT]+)", hg)
    if not m:
        return (None,) * 4 + ("INDEL_OR_OTHER", hg)
    chrom, pos, ref, alt = m.group(2), int(m.group(3)), m.group(4), m.group(5)
    if len(ref) != 1 or len(alt) != 1:
        return chrom, pos, ref, alt, "MULTI_NUCLEOTIDE_VARIANT", hg
    return chrom, pos, ref, alt, "EXACT_GRCH38_SNV", hg


def _pro_to_short(hgvs_pro):
    m = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", str(hgvs_pro))
    if not m:
        return None
    a, pos, b = AA3.get(m.group(1)), m.group(2), AA3.get(m.group(3))
    return f"{a}{pos}{b}" if a and b else None


def run(confirmed="data/mavedb_confirmed_14b.parquet", base="data/proteingym_v1.3"):
    C = pd.read_parquet(confirmed)
    pgref = pd.read_csv(f"{base}/DMS_substitutions.csv").set_index("DMS_id")
    rows, retrieved, failed = [], 0, []
    for _, c in C.iterrows():
        urn, dms_id = c["mavedb_score_set_urn"], c["proteinGym_DMS_id"]
        data, status = _fetch(urn)
        if data is None:
            failed.append((urn, status)); print(f"  {urn} {status}", flush=True); continue
        retrieved += 1
        # ProteinGym scores for AA join
        fn = pgref.loc[dms_id, "DMS_filename"]
        pg_path = None
        for root, _, files in os.walk(base):
            if fn in files:
                pg_path = os.path.join(root, fn); break
        pg = pd.read_csv(pg_path) if pg_path else pd.DataFrame(columns=["mutant", "DMS_score"])
        pg = pg[~pg["mutant"].astype(str).str.contains(":")]
        pgmap = dict(zip(pg["mutant"].astype(str), pg["DMS_score"]))
        counts = dict.fromkeys(CATS, 0)
        for rec in data:
            ch, pos, ref, alt, cat, hg = _parse(rec)
            pro = None
            pmx = rec.get("postMapped") or {}
            for e in (pmx.get("expressions") or []):
                if e.get("syntax") == "hgvs.p" or ":p." in str(e.get("value", "")):
                    pro = _pro_to_short(e.get("value")); break
            if pro is None:
                pre = rec.get("preMapped") or {}
                for e in (pre.get("expressions") or []):
                    pro = _pro_to_short(e.get("value"))
                    if pro:
                        break
            score = pgmap.get(pro) if pro else None
            if pro and score is None and cat == "EXACT_GRCH38_SNV":
                cat = "AMBIGUOUS_DMS_JOIN"
            counts[cat] = counts.get(cat, 0) + 1
            rows.append({"urn": urn, "mavedb_variant_id": rec.get("variantUrn") or rec.get("id"),
                         "proteinGym_DMS_id": dms_id, "aa_mutation": pro, "DMS_score": score,
                         "chrom": ch, "pos": pos, "ref": ref, "alt": alt,
                         "category": cat, "hgvs_g": hg})
        print(f"  {urn} {status} n={len(data):,} exact_snv={counts.get('EXACT_GRCH38_SNV',0):,}",
              flush=True)
        time.sleep(0.5)

    M = pd.DataFrame(rows)
    M.to_parquet("data/dms_genomic_mapping_14b.parquet")
    vc = M["category"].value_counts() if len(M) else pd.Series(dtype=int)
    usable = M[(M["category"] == "EXACT_GRCH38_SNV") & M["DMS_score"].notna()]

    L = ["STEP 14B-PART2 — GENOMIC MAPPING QC", "=" * 35, "",
         f"Score sets expected:                 {len(C)}",
         f"Score sets retrieved:                {retrieved}",
         f"Score sets failed:                   {len(failed)}"]
    for u, s in failed:
        L.append(f"    {u}  {s}")
    L += ["", f"MaveDB variants retrieved:           {len(M):,}",
          f"ProteinGym <-> MaveDB exact AA joins: {int(M['DMS_score'].notna().sum()):,}",
          "", "Mapping:"]
    for k in CATS:
        L.append(f"  {k:<34}{int(vc.get(k,0)):>9,}")
    L += ["", "FINAL:",
          f"  usable exact GRCh38 SNVs:          {len(usable):,}",
          f"  assays represented:                {usable['proteinGym_DMS_id'].nunique()}",
          f"  unique proteins:                   "
          f"{C.set_index('proteinGym_DMS_id').loc[usable['proteinGym_DMS_id'].unique(), 'proteinGym_protein'].nunique() if len(usable) else 0}",
          "", "NOTE: MaveDB postMapped is protein-level for protein-target score sets;",
          "  those yield no GRCh38 coordinates and are counted MAPPING_UNAVAILABLE.",
          "", "Per-assay usable exact SNVs:",
          (usable.groupby("proteinGym_DMS_id").size().sort_values(ascending=False).to_string()
           if len(usable) else "  none")]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
