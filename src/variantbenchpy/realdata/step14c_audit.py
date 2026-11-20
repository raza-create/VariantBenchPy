"""STEP 14C integrity audit: BRCA2 exclusions + 14A/14C count reconciliation."""
from __future__ import annotations
import os, re
import pandas as pd
from variantbenchpy.realdata.step14c_mane_map import (
    load_cds_blocks, load_rna, map_assay, STATES)

PG_ID = "BRCA2_HUMAN_Erwood_2022_HEK293T"


def run(base="data/proteingym_v1.3"):
    ref = pd.read_csv(f"{base}/DMS_substitutions.csv")
    human = ref[ref["taxon"].astype(str).str.lower() == "human"].copy()
    genes = load_cds_blocks()
    rnas = load_rna([d["tx"] for d in genes.values()])

    # ---- (2) count reconciliation: mutants vs emitted rows ----
    n_mutants = 0
    rows_per_state = dict.fromkeys(STATES, 0)
    mut_per_state = dict.fromkeys(STATES, 0)
    for _, r in human.iterrows():
        fn = r["DMS_filename"]
        path = next((os.path.join(root, fn) for root, _, fs in os.walk(base) if fn in fs), None)
        if path is None:
            continue
        d = pd.read_csv(path)
        mcol = "mutant" if "mutant" in d.columns else d.columns[0]
        single = d[~d[mcol].astype(str).str.contains(":")]
        muts = single[mcol].astype(str).tolist()
        n_mutants += len(muts)
        gene = str(r["UniProt_ID"]).split("_")[0]
        target, err, mapped = map_assay(str(r["target_seq"]), gene, genes, rnas, muts)
        if err:
            rows_per_state[err] = rows_per_state.get(err, 0) + len(muts)
            mut_per_state[err] = mut_per_state.get(err, 0) + len(muts)
            continue
        seen = {}
        for mm in mapped:
            rows_per_state[mm["state"]] = rows_per_state.get(mm["state"], 0) + 1
            seen.setdefault(mm["mutant"], mm["state"])
        for _m, st in seen.items():
            mut_per_state[st] = mut_per_state.get(st, 0) + 1

    total_rows = sum(rows_per_state.values())
    total_muts = sum(mut_per_state.values())

    # ---- (1) BRCA2 exclusion reasons ----
    gold = pd.read_parquet("data/brca2_dms_snv_clean.parquet")
    r = human[human["DMS_id"] == PG_ID].iloc[0]
    _t, _e, mapped = map_assay(str(r["target_seq"]), "BRCA2", genes, rnas,
                               gold["aa_pg"].astype(str).tolist())
    st_by_mut, coords = {}, {}
    for mm in mapped:
        st_by_mut[mm["mutant"]] = mm["state"]
        if "pos" in mm:
            coords.setdefault(mm["mutant"], []).append(
                (mm["chrom"], mm["pos"], mm["ref"], mm["alt"]))
    G = gold.copy()
    G["mane_state"] = G["aa_pg"].map(st_by_mut)
    uniq = G[G["mane_state"] == "UNIQUE_MANE_GRCH38_SNV"].copy()
    def agree(row):
        c = coords.get(row["aa_pg"], [])
        return (str(row["chrom"]), int(row["pos"]), row["ref"], row["alt"]) in [
            (str(a), int(b), c2, d2) for a, b, c2, d2 in c]
    uniq["agrees"] = uniq.apply(agree, axis=1)
    excl = G[G["mane_state"] != "UNIQUE_MANE_GRCH38_SNV"]

    L = ["STEP 14C INTEGRITY AUDIT", "=" * 24, "",
         "ProteinGym substitution accounting:",
         f"  14A (single-AA mutants counted once):  329,664",
         f"  recount now (mutants):                 {n_mutants:,}",
         f"  14C emitted ROWS (multi-realization",
         f"    mutants emit one row per genomic SNV): {total_rows:,}",
         f"  14C mutants (deduplicated):            {total_muts:,}",
         f"  difference (rows - mutants):           {total_rows - total_muts:,}",
         f"  difference explained: "
         f"{'YES — MULTIPLE_MANE_GRCH38_SNVS emit >1 row per mutant' if total_rows - total_muts > 0 else 'NO'}",
         f"  final frozen total (mutants):          {total_muts:,}", "",
         "  per-state (mutants | rows):"]
    for s in STATES:
        L.append(f"    {s:<32}{mut_per_state.get(s,0):>10,} | {rows_per_state.get(s,0):>10,}")
    L += ["", "BRCA2 external validation:",
          f"  MaveDB SNVs:                  {len(G)}",
          f"  reconstructable by MANE:      {len(uniq)}",
          f"  exact agreement:              {int(uniq['agrees'].sum())}/{len(uniq)}",
          f"  excluded:                     {len(excl)}",
          "  exclusion reasons:"]
    for k, v in excl["mane_state"].value_counts(dropna=False).items():
        L.append(f"    {k}: {v}")
    if len(excl):
        L.append("  excluded mutants: " + ", ".join(excl["aa_pg"].astype(str).tolist()))
    ok = (len(uniq) > 0) and bool(uniq["agrees"].all())
    L += ["", f"STATUS: {'PASS' if ok else 'FAIL'}",
          "  (agreement is evaluated only on mutants MANE classifies as",
          "   UNIQUE_MANE_GRCH38_SNV; others are excluded by the frozen rule,",
          "   never arbitrarily resolved)"]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
