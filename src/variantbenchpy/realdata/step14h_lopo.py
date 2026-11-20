"""STEP 14H — leave-one-protein-out robustness (protein-weighted, no bootstrap)."""
from __future__ import annotations
import numpy as np
import pandas as pd

PREDS5 = ["AlphaMissense", "REVEL", "CADD", "SIFT", "PolyPhen-2"]


def run(spearman="data/dms_spearman_shared5_14e.parquet"):
    R = pd.read_parquet(spearman)
    W = R.pivot_table(index=["assay", "protein"], columns="predictor",
                      values="spearman_rho").dropna(subset=PREDS5)
    per_prot = W.groupby(level="protein")[PREDS5].median()
    proteins = list(per_prot.index)

    ref = {p: float(per_prot[p].median()) for p in PREDS5}
    ref_rank = sorted(PREDS5, key=lambda p: -ref[p])
    ref_diff = ref["AlphaMissense"] - ref["REVEL"]

    rows = []
    for prot in proteins:
        sub = per_prot.drop(index=prot)
        med = {p: float(sub[p].median()) for p in PREDS5}
        rank = sorted(PREDS5, key=lambda p: -med[p])
        rows.append({"removed": prot, "winner": rank[0],
                     "ranking": " > ".join(rank),
                     "same_ranking": rank == ref_rank,
                     "am": med["AlphaMissense"], "revel": med["REVEL"],
                     "am_minus_revel": med["AlphaMissense"] - med["REVEL"]})
    L_ = pd.DataFrame(rows)

    wc = L_["winner"].value_counts()
    out = ["STEP 14H — DMS LEAVE-ONE-PROTEIN-OUT ROBUSTNESS", "=" * 48, "",
           f"Proteins: {len(proteins)}",
           f"LOPO configurations: {len(L_)}", "",
           "Reference ranking:",
           "  " + " > ".join(ref_rank), "",
           "Winner across LOPO configurations:"]
    for p in PREDS5:
        out.append(f"  {p:<16}{int(wc.get(p, 0)):>4}")
    out += ["",
            f"AlphaMissense retained #1:",
            f"  {int((L_['winner'] == 'AlphaMissense').sum())} / {len(L_)}", "",
            "Ranking completely unchanged:",
            f"  {int(L_['same_ranking'].sum())} / {len(L_)}", "",
            "AlphaMissense - REVEL difference:",
            f"  reference:            {ref_diff:+.3f}",
            f"  minimum across LOPO:  {L_['am_minus_revel'].min():+.3f} "
            f"(remove {L_.loc[L_['am_minus_revel'].idxmin(), 'removed']})",
            f"  maximum across LOPO:  {L_['am_minus_revel'].max():+.3f} "
            f"(remove {L_.loc[L_['am_minus_revel'].idxmax(), 'removed']})",
            f"  configurations <= 0:  {int((L_['am_minus_revel'] <= 0).sum())}", "",
            "Most influential protein removals (smallest AM-REVEL first):",
            "  " + f"{'removed protein':<20}{'AM rho':>9}{'REVEL rho':>11}{'AM-REVEL':>10}  winner"]
    for _, r in L_.nsmallest(6, "am_minus_revel").iterrows():
        out.append("  " + f"{r['removed']:<20}{r['am']:>9.3f}{r['revel']:>11.3f}"
                   f"{r['am_minus_revel']:>10.3f}  {r['winner']}")

    diff_rank = L_[~L_["same_ranking"]]
    out += ["", f"Configurations with a changed ranking: {len(diff_rank)}"]
    for _, r in diff_rank.iterrows():
        out.append(f"  remove {r['removed']:<18}{r['ranking']}")

    L_.to_parquet("data/dms_lopo_14h.parquet")
    out += ["", "-> data/dms_lopo_14h.parquet",
            "(point estimates only; protein-weighted estimand; no bootstrap; EVE excluded)"]
    return "\n".join(out)


if __name__ == "__main__":
    print(run())
