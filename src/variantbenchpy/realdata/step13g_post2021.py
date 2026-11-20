"""STEP 13G — post-2021 temporal sensitivity (paired gene-clustered)."""
from __future__ import annotations
from math import erfc, sqrt
import numpy as np
import pandas as pd
from variantbenchpy.realdata.step13e_temporal_perf import (
    PREDS5, PRED6, WITHIN_MIN, _pooled, _macro, _holm, _paired_boot)

CUT = "2021-10-27"
PAIRS = [("REVEL", "AlphaMissense"), ("AlphaMissense", "CADD")]


def _layer(df, preds, gcol, n_boot, seed, title):
    sh = df[preds].notna().all(axis=1)
    sub = df[sh].copy()
    pm = sub["first_in_clinvar"] > pd.Timestamp(CUT)
    L = ["", title,
         f"  Full: {len(sub):,} / {sub[gcol].nunique():,} genes | "
         f"Post2021: {int(pm.sum()):,} / {sub.loc[pm, gcol].nunique():,} genes",
         "  " + f"{'Predictor':<15}{'Full':>8}{'Post2021':>10}{'ΔAUROC [95% CI]':>28}"]
    pf, pp_, dfu, dpo = _paired_boot(sub, preds, pm, gcol, "pooled", n_boot, seed)
    for p in preds:
        B = min(len(dfu[p]), len(dpo[p]))
        diff = dpo[p][:B] - dfu[p][:B]
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        L.append("  " + f"{p:<15}{pf[p]:>8.3f}{pp_[p]:>10.3f}"
                 f"   {pp_[p]-pf[p]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    mf, mp, _, _ = _paired_boot(sub, preds, pm, gcol, "macro", 1, seed)
    rk = lambda d: " > ".join(sorted(d, key=d.get, reverse=True))
    L += [f"  Pooled ranking Full:      {rk(pf)}",
          f"  Pooled ranking Post2021:  {rk(pp_)}",
          f"  Macro ranking  Full:      {rk(mf)}",
          f"  Macro ranking  Post2021:  {rk(mp)}",
          "  " + f"{'Pair':<28}{'ΔΔAUROC':>9}{'95% CI':>24}{'Holm p':>10}"]
    rows, pv = [], []
    for a, b in [(a, b) for a, b in PAIRS if a in preds and b in preds]:
        B = min(len(dfu[a]), len(dfu[b]), len(dpo[a]), len(dpo[b]))
        dd = (dpo[a][:B] - dpo[b][:B]) - (dfu[a][:B] - dfu[b][:B])
        pt = (pp_[a] - pp_[b]) - (pf[a] - pf[b])
        lo, hi = np.nanpercentile(dd, [2.5, 97.5])
        pv.append(erfc(abs(pt / (np.nanstd(dd) + 1e-12)) / sqrt(2)))
        rows.append((f"{a} vs {b}", pt, lo, hi))
    for (nm, pt, lo, hi), pa in zip(rows, _holm(np.array(pv))):
        ps = "<0.001" if pa < 0.001 else f"{pa:.3f}"
        L.append("  " + f"{nm:<28}{pt:>+9.4f}   [{lo:+.4f}, {hi:+.4f}] {ps:>8}")
    return "\n".join(L)


def run(path="data/benchmark_with_temporal.parquet", n_boot=400, seed=11):
    df = pd.read_parquet(path)
    df["first_in_clinvar"] = pd.to_datetime(df["germline_date_created"], errors="coerce")
    g = "primary_gene_10A"
    return "\n".join([
        "POST-2021 TEMPORAL SENSITIVITY", "=" * 30,
        f"landmark: germline First-in-ClinVar > {CUT} (EVE publication proxy)",
        "ΔAUROC / ΔΔAUROC CI: paired gene-clustered bootstrap (nested samples)",
        _layer(df, PREDS5, g, n_boot, seed, "5-PREDICTOR"),
        _layer(df, PRED6, g, n_boot, seed + 5, "6-PREDICTOR (incl. EVE)"),
    ])


if __name__ == "__main__":
    print(run())
