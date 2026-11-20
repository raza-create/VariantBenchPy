"""STEP 16 runner — build Fig. 1-5, run all QC checks, save only if every check passes."""
from __future__ import annotations
import os, sys, traceback
import pandas as pd
import matplotlib.pyplot as plt
from variantbenchpy.realdata.fignat_common import (
    QC, STRICT, OUT, SRC, load_bench, parse_audit, parse_step11, save)
from variantbenchpy.realdata.fignat_f1 import fig1
from variantbenchpy.realdata.fignat_f2 import fig2
from variantbenchpy.realdata.fignat_f3 import fig3
from variantbenchpy.realdata.fignat_f4 import fig4
from variantbenchpy.realdata.fignat_f5 import fig5


def main():
    os.makedirs(SRC, exist_ok=True)
    Q = QC()
    B = load_bench()
    A = parse_audit()
    G = pd.read_parquet("data/step12_reliability_grid_5pred.parquet")
    S = parse_step11()
    builders = [("Fig1", lambda: fig1(B, Q)), ("Fig2", lambda: fig2(A, G, Q)),
                ("Fig3", lambda: fig3(B, S, Q)), ("Fig4", lambda: fig4(B, Q)),
                ("Fig5", lambda: fig5(B, Q))]
    figs = {}
    for name, fn in builders:
        try:
            figs[name] = fn()
        except Exception as e:
            Q.check(f"{name} build", False, f"{type(e).__name__}: {e}")
            print(traceback.format_exc(limit=3))

    print("STEP 16 — NATURE FIGURE QC")
    print("=" * 26)
    for name, ok, detail in Q.rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<44}{detail}")
    failed = [r for r in Q.rows if not r[1]]
    print(f"\nchecks: {len(Q.rows)}   passed: {len(Q.rows) - len(failed)}   failed: {len(failed)}")

    print("\nNEW VALUES (not previously reported; transcribe from here only):")
    for name, v in Q.new:
        print(f"  {name}:")
        if isinstance(v, dict):
            for k, vv in v.items():
                print(f"      {k}: {vv}")
        else:
            print(f"      {v}")

    if failed and STRICT:
        for f in figs.values():
            plt.close(f)
        print("\nSTATUS: FAIL — no figures written. Paste this output.")
        sys.exit(1)
    for name, f in figs.items():
        save(f, name)
    print(f"\nSTATUS: {'PASS' if not failed else 'LAYOUT TEST (STRICT=0)'}")
    print(f"saved: {', '.join(os.path.join(OUT, n + '.pdf') for n in figs)} (+ .png)")
    print(f"source data: {SRC}/")


if __name__ == "__main__":
    main()
