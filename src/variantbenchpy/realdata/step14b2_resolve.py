"""STEP 14B part 2 — resolve PROBABLE_MATCH_REVIEW crosswalk entries.

Evidence, strongest first:
  1. target protein sequence identity (MaveDB DNA translated -> == ProteinGym target_seq)
  2. UniProt external identifier on the MaveDB target
  3. publication PMID/title/author/year agreement
Statuses: EXACT_EXPERIMENT_MATCH / REJECTED_MISMATCH / MULTIPLE_CANDIDATES_UNRESOLVED
"""
from __future__ import annotations
import difflib, json, re, time, urllib.request
import pandas as pd

API = "https://api.mavedb.org/api/v1"
_B = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {b1 + b2 + b3: _AA[i * 16 + j * 4 + k]
         for i, b1 in enumerate(_B) for j, b2 in enumerate(_B) for k, b3 in enumerate(_B)}


def translate(dna):
    dna = dna.upper().replace("U", "T")
    return "".join(CODON.get(dna[i:i + 3], "X") for i in range(0, len(dna) - 2, 3)).rstrip("*")


def _get(path, timeout=60):
    req = urllib.request.Request(API + path, headers={"User-Agent": "VariantBenchPy/research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _post(path, payload, timeout=60):
    req = urllib.request.Request(API + path, method="POST", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "VariantBenchPy/research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def evaluate(full, pg_row):
    t = (full.get("targetGenes") or [{}])[0]
    ts = t.get("targetSequence") or {}
    seq = ts.get("sequence") or ""
    prot = translate(seq) if ts.get("sequenceType") == "dna" else seq
    pg_seq = str(pg_row["target_seq"])
    seq_exact = bool(prot) and prot == pg_seq
    seq_sub = bool(prot) and (prot in pg_seq or pg_seq in prot)
    ext = t.get("externalIdentifiers") or []
    uniprots = [e.get("identifier", {}).get("identifier") for e in ext
                if e.get("identifier", {}).get("dbName") == "UniProt"]
    pubs = full.get("primaryPublicationIdentifiers") or []
    pmids = [p.get("identifier") for p in pubs]
    titles = [p.get("title", "") for p in pubs]
    dois = [p.get("doi") for p in pubs if p.get("doi")]
    authors = [a.get("name", "") for p in pubs for a in (p.get("authors") or [])]
    fa = str(pg_row.get("first_author", "")).split()[0].lower() if pg_row.get("first_author") else ""
    author_ok = any(fa and a.lower().startswith(fa) for a in authors)
    year_ok = str(pg_row.get("year", "")) in " ".join(
        str(p.get("publicationYear", "")) for p in pubs)
    tsim = max([difflib.SequenceMatcher(None, _norm(pg_row.get("title", "")), _norm(x)).ratio()
                for x in titles] or [0.0])
    return {"seq_exact": seq_exact, "seq_sub": seq_sub, "uniprots": uniprots,
            "pmids": pmids, "dois": dois, "author_ok": author_ok, "year_ok": year_ok,
            "title_sim": round(tsim, 2), "prot_len": len(prot),
            "mapping_state": full.get("mappingState")}


def resolve(base="data/proteingym_v1.3",
            crosswalk="data/mavedb_crosswalk_14b.parquet", pause=0.7):
    X = pd.read_parquet(crosswalk)
    ref = pd.read_csv(f"{base}/DMS_substitutions.csv").set_index("DMS_id")
    todo = X[X["match_status"].isin(
        ["PROBABLE_MATCH_REVIEW", "MULTIPLE_MAVEDB_CANDIDATES"])]
    out = []
    for _, r in todo.iterrows():
        pg = ref.loc[r["proteinGym_DMS_id"]]
        gene = str(r["proteinGym_protein"]).split("_")[0]
        try:
            hits = _post("/score-sets/search", {"text": gene}).get("scoreSets", [])
        except Exception:
            hits = []
        time.sleep(pause)
        scored = []
        for h in hits:
            try:
                full = _get(f"/score-sets/{h['urn']}")
            except Exception:
                continue
            time.sleep(pause)
            ev = evaluate(full, pg)
            score = (4 * ev["seq_exact"] + 2 * ev["seq_sub"] + 2 * ev["author_ok"]
                     + ev["year_ok"] + (1 if ev["title_sim"] > 0.6 else 0))
            scored.append((score, h["urn"], full.get("numVariants"), ev))
        scored.sort(key=lambda z: -z[0])
        best = scored[0] if scored else None
        strong = [s for s in scored if s[3]["seq_exact"]]
        if len(strong) == 1:
            status, basis = "EXACT_EXPERIMENT_MATCH", "target protein sequence identity"
            pick = strong[0]
        elif len(strong) > 1:
            auth = [s for s in strong if s[3]["author_ok"]]
            if len(auth) == 1:
                status, basis, pick = "EXACT_EXPERIMENT_MATCH", "sequence identity + author", auth[0]
            else:
                status, basis, pick = "MULTIPLE_CANDIDATES_UNRESOLVED", \
                    f"{len(strong)} sequence-identical score sets", strong[0]
        elif best and best[3]["author_ok"] and best[3]["seq_sub"]:
            status, basis, pick = "EXACT_EXPERIMENT_MATCH", "sequence substring + author", best
        else:
            status, basis, pick = "REJECTED_MISMATCH", "no sequence identity", best
        ev = pick[3] if pick else {}
        out.append({"proteinGym_DMS_id": r["proteinGym_DMS_id"],
                    "proteinGym_protein": r["proteinGym_protein"],
                    "mavedb_score_set_urn": pick[1] if pick else None,
                    "mavedb_n_variants": pick[2] if pick else None,
                    "publication_pmid": ",".join(ev.get("pmids", []) or []),
                    "publication_doi": ",".join(ev.get("dois", []) or []),
                    "target_accession": ",".join(ev.get("uniprots", []) or []),
                    "assay_type": pg.get("coarse_selection_type"),
                    "seq_exact": ev.get("seq_exact"), "title_sim": ev.get("title_sim"),
                    "author_ok": ev.get("author_ok"), "mapping_state": ev.get("mapping_state"),
                    "match_basis": basis, "status": status, "n_candidates": len(scored)})
        print(f"  {r['proteinGym_DMS_id'][:45]:<45} {status}", flush=True)
    R = pd.DataFrame(out)

    conf_before = X[X["match_status"] == "EXACT_EXPERIMENT_MATCH"]
    conf_new = R[R["status"] == "EXACT_EXPERIMENT_MATCH"]
    final = pd.concat([
        conf_before.assign(status="EXACT_EXPERIMENT_MATCH",
                           match_basis="gene+first_author(+year)").rename(
            columns={"mavedb_pmids": "publication_pmid"}),
        conf_new], ignore_index=True)
    final.to_parquet("data/mavedb_confirmed_14b.parquet")
    R.to_parquet("data/mavedb_resolution_14b.parquet")

    L = ["STEP 14B — CROSSWALK RESOLUTION", "=" * 31, "", "Initial:",
         f"  exact matches:                  {int((X['match_status']=='EXACT_EXPERIMENT_MATCH').sum())}",
         f"  probable/manual review:         {int((X['match_status']=='PROBABLE_MATCH_REVIEW').sum())}",
         f"  multiple candidates:            {int((X['match_status']=='MULTIPLE_MAVEDB_CANDIDATES').sum())}",
         f"  no MaveDB match:                {int((X['match_status']=='NO_MAVEDB_MATCH').sum())}",
         "", "After metadata/sequence resolution:",
         f"  confirmed exact matches:        {len(conf_before) + len(conf_new)}",
         f"  rejected mismatches:            {int((R['status']=='REJECTED_MISMATCH').sum())}",
         f"  still ambiguous:                {int((R['status']=='MULTIPLE_CANDIDATES_UNRESOLVED').sum())}",
         f"  no MaveDB source:               {int((X['match_status']=='NO_MAVEDB_MATCH').sum())}",
         "",
         f"Confirmed unique proteins:        {final['proteinGym_protein'].nunique()}",
         f"Confirmed MaveDB score sets:      {final['mavedb_score_set_urn'].nunique()}",
         f"Confirmed declared variants:      "
         f"{int(pd.to_numeric(final['mavedb_n_variants'], errors='coerce').sum()):,}",
         "", "Resolution evidence: protein-sequence identity is decisive; author/year/title",
         "  used only as tie-breakers. Nothing auto-promoted without sequence agreement.",
         "", "-> data/mavedb_confirmed_14b.parquet, data/mavedb_resolution_14b.parquet", "",
         "CONFIRMED MATCHES:",
         final[["proteinGym_DMS_id", "proteinGym_protein", "mavedb_score_set_urn",
                "status"]].to_string(index=False)]
    return "\n".join(L)


if __name__ == "__main__":
    print(resolve())
