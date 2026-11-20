"""EVE (classical) integration with an explicit per-variant mapping cascade.

Key = (uniprot_entry 'GENE_HUMAN', aaref, position, aaalt). Accept a score only when
protein model exists AND ref-AA AND position AND alt-AA all agree. Every variant gets
a mapping state; gene-name-only rescue is NEVER done (isoform numbering differs)."""
from __future__ import annotations
import io, zipfile
import numpy as np
import pandas as pd

SCORE = "EVE_scores_ASM"


def load_and_join(eve_zip, slice_parquet):
    sl = pd.read_parquet(slice_parquet).drop_duplicates("variant_id").copy()
    sl["entry"] = sl["uniprot_entry"].astype(str)
    sl["pos"] = pd.to_numeric(sl["aapos"], errors="coerce")

    # proteins EVE actually models (from zip member names)
    with zipfile.ZipFile(eve_zip) as z:
        members = [m for m in z.namelist()
                   if m.startswith("variant_files/") and m.lower().endswith(".csv")]
        eve_proteins = {m.split("/")[-1].replace(".csv", "") for m in members}
        want = set(sl.loc[sl["entry"] != ".", "entry"]) & eve_proteins
        frames = []
        for m in members:
            entry = m.split("/")[-1].replace(".csv", "")
            if entry not in want:
                continue
            with z.open(m) as raw:
                df = pd.read_csv(io.TextIOWrapper(raw, "utf-8"),
                                 usecols=["wt_aa", "position", "mt_aa", SCORE])
            df["entry"] = entry
            frames.append(df)
    eve = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["wt_aa", "position", "mt_aa", SCORE, "entry"])
    eve = eve.rename(columns={"wt_aa": "aaref", "position": "pos",
                              "mt_aa": "aaalt", SCORE: "EVE"})
    eve["pos"] = pd.to_numeric(eve["pos"], errors="coerce")
    eve["EVE"] = pd.to_numeric(eve["EVE"], errors="coerce")
    dup = int(eve.duplicated(["entry", "aaref", "pos", "aaalt"]).sum())
    eve_u = eve.drop_duplicates(["entry", "aaref", "pos", "aaalt"])

    # protein-level presence + position presence for cascade states
    prot_has = set(eve_u["entry"])
    prot_pos = set(zip(eve_u["entry"], eve_u["pos"]))
    exact = eve_u.set_index(["entry", "aaref", "pos", "aaalt"])["EVE"]

    def state(r):
        if r["entry"] == "." or pd.isna(r["pos"]):
            return "no_protein_mapping", np.nan
        if r["entry"] not in prot_has:
            return "EVE_protein_not_covered", np.nan
        k = (r["entry"], r["aaref"], r["pos"], r["aaalt"])
        if k in exact.index:
            v = exact.loc[k]
            if pd.notna(v):
                return "EVE_exact_scored", float(v)
            return "EVE_exact_score_missing", float("nan")
        if (r["entry"], r["pos"]) in prot_pos:
            # position exists but this ref/alt AA combination does not
            return "EVE_AA_or_alt_mismatch", np.nan
        return "EVE_position_absent", np.nan

    res = sl.apply(state, axis=1, result_type="expand")
    sl["EVE_state"] = res[0]; sl["EVE"] = res[1]
    out = sl.set_index("variant_id")
    stats = {"eve_models_total": len(eve_proteins),
             "eve_models_intersecting_benchmark": len(want),
             "duplicate_eve_keys": dup,
             "state_counts": out["EVE_state"].value_counts().to_dict()}
    return out[["EVE", "EVE_state"]], stats


if __name__ == "__main__":
    eve, stats = load_and_join("data/eve/EVE_all_data.zip", "data/dbnsfp_slice.parquet")
    eve.to_parquet("data/eve_scores.parquet")
    import json
    print(json.dumps(stats, indent=2, default=str))
    s = eve["EVE"].dropna()
    print(f"EVE scored: {len(s):,}  range [{s.min():.3f}, {s.max():.3f}]  dup_keys={stats['duplicate_eve_keys']}")
