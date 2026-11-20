"""STEP 14B — apply frozen BRCA2 mapping logic to all 11 confirmed score sets."""
from __future__ import annotations
import io, json, os, re, subprocess, time
import pandas as pd
from variantbenchpy.realdata.step14b6_cleanup import norm_pro

API = "https://api.mavedb.org/api/v1"
CACHE = "data/mavedb_mapped"


def _curl(url, out=None, retries=6):
    os.makedirs(CACHE, exist_ok=True)
    cmd = ["curl", "-sS", "--retry", str(retries), "--retry-delay", "3",
           "--max-time", "1800", "-H", "User-Agent: VariantBenchPy/research", url]
    if out:
        cmd += ["-o", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return None if r.returncode else out
    r = subprocess.run(cmd, capture_output=True, text=True)
    return None if r.returncode else r.stdout


def mapped_variants(urn):
    p = f"{CACHE}/{urn.replace(':', '_')}.json"
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            os.remove(p)
    if _curl(f"{API}/score-sets/{urn}/mapped-variants", out=p) is None:
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def score_rows(urn):
    os.makedirs(CACHE, exist_ok=True)
    out = f"{CACHE}/{urn.replace(':', '_')}_scores.csv"
    if not os.path.exists(out) or os.path.getsize(out) < 100:
        if _curl(f"{API}/score-sets/{urn}/scores", out=out) is None:
            return None
    for kw in ({}, {"engine": "python", "on_bad_lines": "skip"}):
        try:
            df = pd.read_csv(out, comment=None, **kw)
            if "accession" in df.columns:
                return df
        except Exception:
            continue
    return None


def process(urn, pg_id, base="data/proteingym_v1.3"):
    mv = mapped_variants(urn)
    if mv is None:
        return None, {"status": "RETRIEVAL_FAILED"}
    rows = []
    for r in mv:
        if not r.get("current"):
            continue
        pm = r.get("postMapped") or {}
        exprs = [e.get("value", "") for e in (pm.get("expressions") or [])]
        g = next((e for e in exprs if e.startswith("NC_")), None)
        ch = pos = ref = alt = None
        if g:
            m = re.match(r"NC_0+(\d+|X|Y)\.\d+:g\.(\d+)([ACGT]+)>([ACGT]+)", g)
            if m:
                ch, pos, ref, alt = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        rows.append({"variantUrn": r.get("variantUrn"), "chrom": ch, "pos": pos,
                     "ref": ref, "alt": alt, "hgvs_g": g,
                     "protein_only": g is None and bool(exprs),
                     "atMismatchedLocus": bool(r.get("atMismatchedLocus"))})
    Mc = pd.DataFrame(rows)
    S = score_rows(urn)
    if S is None or Mc.empty:
        return None, {"status": "NO_DATA"}
    S = S.rename(columns={"accession": "variantUrn"})
    S["aa_mavedb"] = S["hgvs_pro"].map(norm_pro)
    J = Mc.merge(S[["variantUrn", "aa_mavedb", "score"]], on="variantUrn", how="left")

    ref_csv = pd.read_csv(f"{base}/DMS_substitutions.csv").set_index("DMS_id")
    fn = ref_csv.loc[pg_id, "DMS_filename"]
    pg_path = next((os.path.join(r, fn) for r, _, f in os.walk(base) if fn in f), None)
    PG = pd.read_csv(pg_path)
    PG = PG[~PG["mutant"].astype(str).str.contains(":")].copy()
    PG["aa_pg"] = PG["mutant"].astype(str)

    K = J.merge(PG[["aa_pg", "DMS_score"]], left_on="aa_mavedb", right_on="aa_pg", how="inner")
    aa_matches = K["aa_pg"].nunique()
    snv = K[K["ref"].notna() & (K["ref"].astype(str).str.len() == 1)
            & (K["alt"].astype(str).str.len() == 1)].copy()
    snv["assay"] = pg_id
    snv = snv.drop_duplicates(["assay", "aa_pg", "chrom", "pos", "ref", "alt"])
    per_aa = snv.groupby("aa_pg").size()
    snv["realization_class"] = snv["aa_pg"].map(
        lambda a: "MULTIPLE_GENOMIC_REALIZATIONS" if per_aa.get(a, 0) > 1 else "UNIQUE_GENOMIC_SNV")
    stats = {"status": "OK", "score_rows": len(S), "current_mapped": len(Mc),
             "aa_matches": aa_matches, "unique_snv": len(snv),
             "multi_real": int((per_aa > 1).sum()),
             "protein_only": int(Mc["protein_only"].sum()),
             "mismatched": int(Mc["atMismatchedLocus"].sum())}
    return snv, stats


def run(confirmed="data/mavedb_confirmed_14b.parquet"):
    C = pd.read_parquet(confirmed)
    frames, srows = [], []
    for _, c in C.iterrows():
        urn, pg_id = c["mavedb_score_set_urn"], c["proteinGym_DMS_id"]
        try:
            K, st = process(urn, pg_id)
        except Exception as e:
            K, st = None, {"status": f"ERROR:{type(e).__name__}"}
        srows.append({"assay": pg_id, "urn": urn, **st})
        if K is not None and len(K):
            frames.append(K)
        print(f"  {pg_id[:42]:<42} {st.get('status')} "
              f"snv={st.get('unique_snv', 0)}", flush=True)
        time.sleep(0.5)
    ALL = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    T = pd.DataFrame(srows)
    if len(ALL):
        ALL.to_parquet("data/dms_snv_all_14b.parquet")

    L = ["STEP 14B — ALL CONFIRMED DMS GENOMIC MAPPING", "=" * 44, "",
         f"Confirmed score sets: {len(C)}", "",
         "  " + f"{'Assay':<32}{'ScoreRows':>10}{'CurrMapped':>12}{'AA_match':>10}"
         f"{'UniqueSNV':>11}{'ProtOnly':>10}"]
    for _, r in T.iterrows():
        L.append("  " + f"{r['assay'][:32]:<32}{r.get('score_rows',0) or 0:>10}"
                 f"{r.get('current_mapped',0) or 0:>12}{r.get('aa_matches',0) or 0:>10}"
                 f"{r.get('unique_snv',0) or 0:>11}{r.get('protein_only',0) or 0:>10}")
    ok = T[T["status"] == "OK"]
    L += ["", "TOTAL",
          f"  score sets successfully mapped:       {len(ok)}",
          f"  score sets failed retrieval:          {int((T['status']!='OK').sum())}",
          f"  exact ProteinGym<->MaveDB AA matches: {int(ok['aa_matches'].fillna(0).sum()):,}",
          f"  unique exact GRCh38 SNVs:             {len(ALL):,}",
          f"  multi-genomic realizations:           "
          f"{int((ALL['realization_class']=='MULTIPLE_GENOMIC_REALIZATIONS').sum()) if len(ALL) else 0:,}",
          f"  reference mismatches:                 {int(ok['mismatched'].fillna(0).sum()):,}",
          f"  protein-only mappings (no GRCh38):    {int(ok['protein_only'].fillna(0).sum()):,}",
          f"  assays represented:                   {ALL['assay'].nunique() if len(ALL) else 0}",
          f"  unique proteins:                      "
          f"{C.set_index('proteinGym_DMS_id').loc[ALL['assay'].unique(), 'proteinGym_protein'].nunique() if len(ALL) else 0}",
          "", "-> data/dms_snv_all_14b.parquet"]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
