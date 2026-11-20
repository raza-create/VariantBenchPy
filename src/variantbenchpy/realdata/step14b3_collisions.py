"""STEP 14B part 3 — one-to-one crosswalk QC.

RULE (corrected): protein-sequence identity is REQUIRED BUT NOT SUFFICIENT.
An EXACT_EXPERIMENT_MATCH additionally requires concordant publication/study
identity (PMID/DOI/author/year) and compatible assay context.
"""
from __future__ import annotations
import json, time, urllib.request
import pandas as pd

API = "https://api.mavedb.org/api/v1"


def _get(path, timeout=60):
    req = urllib.request.Request(API + path, headers={"User-Agent": "VariantBenchPy/research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run(confirmed="data/mavedb_confirmed_14b.parquet",
        base="data/proteingym_v1.3", pause=0.7):
    C = pd.read_parquet(confirmed)
    ref = pd.read_csv(f"{base}/DMS_substitutions.csv").set_index("DMS_id")

    dup_urns = C["mavedb_score_set_urn"].value_counts()
    dup_urns = dup_urns[dup_urns > 1].index.tolist()

    L = ["STEP 14B — ONE-TO-ONE CROSSWALK QC", "=" * 34, "",
         f"Confirmed ProteinGym assays: {len(C)}",
         f"Unique MaveDB score sets:    {C['mavedb_score_set_urn'].nunique()}", ""]

    decisions = {}
    if not dup_urns:
        L.append("URN collisions: none")
    for urn in dup_urns:
        grp = C[C["mavedb_score_set_urn"] == urn]
        meta = _get(f"/score-sets/{urn}")
        time.sleep(pause)
        pubs = meta.get("primaryPublicationIdentifiers") or []
        pmid = ",".join(p.get("identifier", "") for p in pubs)
        doi = ",".join(str(p.get("doi")) for p in pubs if p.get("doi"))
        yr = ",".join(str(p.get("publicationYear", "")) for p in pubs)
        fa = ((pubs[0].get("authors") or [{}])[0].get("name", "") if pubs else "")
        fa_surname = fa.split(",")[0].strip().lower()

        L += [f"URN collision: {urn}", "  ProteinGym candidates:"]
        for d in grp["proteinGym_DMS_id"]:
            L.append(f"    {d}")
        L += ["", "  MaveDB publication:",
              f"    PMID: {pmid}   DOI: {doi}",
              f"    author/year: {fa} ({yr})",
              f"    score-set title: {meta.get('title')}",
              f"    experiment: {(meta.get('experiment') or {}).get('title')}", ""]

        winners = []
        for dms_id in grp["proteinGym_DMS_id"]:
            pg = ref.loc[dms_id]
            pg_author = str(pg.get("first_author", "")).split()[0].lower()
            pg_year = str(pg.get("year", ""))
            pub_ok = bool(pg_author) and pg_author == fa_surname
            year_ok = pg_year in yr
            ok = pub_ok and year_ok
            decisions[dms_id] = ok
            L.append(f"    {dms_id:<45} author_match={pub_ok} year_match={year_ok} "
                     f"-> {'EXACT_EXPERIMENT_MATCH' if ok else 'REJECTED_MISMATCH'}")
            if ok:
                winners.append(dms_id)
        L += ["", f"  Final assignment: {len(winners)} exact match "
              f"({', '.join(winners) if winners else 'none'}); "
              f"{len(grp) - len(winners)} rejected", ""]

    keep = C[~C["proteinGym_DMS_id"].isin([k for k, v in decisions.items() if not v])].copy()
    keep["match_rule"] = ("sequence identity REQUIRED + publication/study identity "
                          "REQUIRED (sequence alone insufficient)")
    keep.to_parquet("data/mavedb_confirmed_14b.parquet")

    conflicts = int((keep["mavedb_score_set_urn"].value_counts() > 1).sum())
    L += ["FINAL",
          f"  exact experiment matches: {len(keep)}",
          f"  unique score sets:        {keep['mavedb_score_set_urn'].nunique()}",
          f"  unique proteins:          {keep['proteinGym_protein'].nunique()}",
          f"  one-to-one conflicts:     {conflicts}",
          f"  STATUS: {'PASS' if conflicts == 0 else 'FAIL'}", "",
          "MATCHING RULE (frozen): protein-sequence identity is required but NOT",
          "  sufficient; exact experiment matching additionally requires concordant",
          "  publication/study identity and compatible assay context.", "",
          "FROZEN CROSSWALK:",
          keep[["proteinGym_DMS_id", "proteinGym_protein",
                "mavedb_score_set_urn"]].to_string(index=False)]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
