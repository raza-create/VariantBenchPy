"""STEP 14B — ProteinGym <-> MaveDB crosswalk (assay matching layer)."""
from __future__ import annotations
import json, re, time, urllib.request
import pandas as pd

API = "https://api.mavedb.org/api/v1"


def _post(path, payload, timeout=60):
    req = urllib.request.Request(API + path, method="POST",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "VariantBenchPy/research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(path, timeout=60):
    req = urllib.request.Request(API + path,
                                 headers={"User-Agent": "VariantBenchPy/research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def search(term):
    try:
        return _post("/score-sets/search", {"text": term}).get("scoreSets", [])
    except Exception:
        return []


def crosswalk(base="data/proteingym_v1.3", pause=0.8):
    ref = pd.read_csv(f"{base}/DMS_substitutions.csv")
    human = ref[ref["taxon"].astype(str).str.lower() == "human"].copy()
    rows = []
    for _, r in human.iterrows():
        gene = str(r["UniProt_ID"]).split("_")[0]
        author = str(r.get("first_author", "")).split()[0] if r.get("first_author") else ""
        year = str(r.get("year", ""))
        hits = search(gene)
        time.sleep(pause)
        cands = []
        for h in hits:
            urn = h.get("urn")
            try:
                full = _get(f"/score-sets/{urn}")
            except Exception:
                continue
            time.sleep(pause)
            pubs = full.get("primaryPublicationIdentifiers") or []
            pmids = [p.get("identifier") for p in pubs]
            auths = [a.get("name", "") for p in pubs for a in (p.get("authors") or [])]
            first_auth = auths[0].split()[0] if auths else ""
            ytxt = " ".join(str(p.get("publicationYear", "")) for p in pubs)
            gene_ok = any(gene.upper() in str(t.get("name", "")).upper()
                          for t in full.get("targetGenes", []))
            author_ok = bool(author) and author.lower() == first_auth.lower()
            year_ok = bool(year) and year in ytxt
            cands.append({"urn": urn, "pmids": pmids, "n_variants": full.get("numVariants"),
                          "gene_ok": gene_ok, "author_ok": author_ok, "year_ok": year_ok,
                          "mappingState": full.get("mappingState")})
        exact = [c for c in cands if c["author_ok"] and c["gene_ok"]]
        prob = [c for c in cands if c["gene_ok"] and not c["author_ok"]]
        if len(exact) == 1:
            status, basis, pick = "EXACT_EXPERIMENT_MATCH", "gene+first_author(+year)", exact[0]
        elif len(exact) > 1:
            status, basis, pick = "MULTIPLE_MAVEDB_CANDIDATES", "multiple author+gene hits", exact[0]
        elif len(prob) >= 1:
            status, basis, pick = "PROBABLE_MATCH_REVIEW", "gene only", prob[0]
        else:
            status, basis, pick = "NO_MAVEDB_MATCH", "", None
        rows.append({"proteinGym_DMS_id": r["DMS_id"], "proteinGym_protein": r["UniProt_ID"],
                     "publication": f"{r.get('first_author')} {r.get('year')}",
                     "mavedb_score_set_urn": pick["urn"] if pick else None,
                     "mavedb_pmids": ",".join(pick["pmids"]) if pick else None,
                     "mavedb_n_variants": pick["n_variants"] if pick else None,
                     "mapping_state": pick["mappingState"] if pick else None,
                     "match_status": status, "match_basis": basis,
                     "n_candidates": len(cands)})
        print(f"  {r['DMS_id'][:45]:<45} {status}", flush=True)
    X = pd.DataFrame(rows)
    X.to_parquet("data/mavedb_crosswalk_14b.parquet")
    return X


def report(X):
    vc = X["match_status"].value_counts()
    ex = X[X["match_status"] == "EXACT_EXPERIMENT_MATCH"]
    L = ["STEP 14B — MAVEDB GENOMIC MAPPING (crosswalk layer)", "=" * 50, "",
         f"ProteinGym human assays:                  {len(X)}", "",
         "MaveDB crosswalk:",
         f"  exact experiment matches:               {int(vc.get('EXACT_EXPERIMENT_MATCH',0))}",
         f"  probable/manual-review matches:         {int(vc.get('PROBABLE_MATCH_REVIEW',0))}",
         f"  multiple candidates:                    {int(vc.get('MULTIPLE_MAVEDB_CANDIDATES',0))}",
         f"  no MaveDB match:                        {int(vc.get('NO_MAVEDB_MATCH',0))}", "",
         f"Exact matched unique proteins:            {ex['proteinGym_protein'].nunique()}",
         f"Exact matched MaveDB variants (declared): {int(pd.to_numeric(ex['mavedb_n_variants'], errors='coerce').sum()):,}",
         "",
         "mapping_state of exact matches:",
         "  " + str(ex["mapping_state"].value_counts().to_dict()),
         "",
         "NOTE: /score-sets/{urn}/mapped-variants returns HTTP 504 for large score sets;",
         "  post-mapped GRCh38 retrieval is deferred to 14B-part2 (bulk CSV route).",
         "", "Crosswalk table -> data/mavedb_crosswalk_14b.parquet"]
    return "\n".join(L)


if __name__ == "__main__":
    X = crosswalk()
    print()
    print(report(X))
