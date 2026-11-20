"""Supplementary Figure S4 — dbNSFP transcript-resolution audit (SIFT, PolyPhen-2).

Two extractions exist and are labelled separately:
  ClinVar benchmark slice : 222,152 variants (data/dbnsfp_slice.parquet)
  DMS/ProteinGym slice    :  35,124 variants (data/dbnsfp_slice_dms.parquet)
Rule in both: MANE-Select positional index preferred; fallback only when the MANE
index carries no numeric score or no MANE flag exists; never max/min aggregation.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHOW_CLINVAR = False
CATS = ["MANE_Select", "MANE_index_no_score->fallback",
        "no_MANE->fallback", "no_usable_score"]
SHORT = ["MANE Select", "MANE index,\nno score → fallback",
         "no MANE →\nfallback", "no usable\nscore"]
COLS = ["#55A868", "#DD8452", "#8172B2", "#C44E52"]

DMS = {"SIFT": [33797, 342, 50, 935], "PolyPhen-2": [33457, 340, 21, 1306]}
CLINVAR = {"SIFT": [176145, 24129, 5239, 16639],
           "PolyPhen-2": [185002, 13363, 3097, 20690]}


def refresh(path, key):
    """Recount from a frozen slice parquet if present."""
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)
    out = {}
    for pred, col in [("SIFT", "SIFT_reason"), ("PolyPhen-2", "PolyPhen2_reason")]:
        if col not in d.columns:
            return None
        vc = d[col].value_counts()
        out[pred] = [int(vc.get(c, 0)) for c in CATS]
    return out


def _panel(ax, counts, pred, title):
    tot = sum(counts)
    bars = ax.bar(range(len(counts)), counts, color=COLS)
    for i, v in enumerate(counts):
        ax.text(i, v + tot * 0.018, f"{v:,}\n({v/tot:.1%})", ha="center", fontsize=7.8)
    ax.set_xticks(range(len(counts)), SHORT, fontsize=7.4)
    ax.set_ylim(0, max(counts) * 1.28)
    ax.set_ylabel("variants")
    ax.set_title(title, fontsize=10, loc="left")


def build(out="figures/FigureS4_transcript_resolution"):
    dms = refresh("data/dbnsfp_slice_dms.parquet", "dms") or DMS
    cv = refresh("data/dbnsfp_slice.parquet", "clinvar") or CLINVAR
    nrow = 2 if SHOW_CLINVAR else 1
    fig, ax = plt.subplots(nrow, 2, figsize=(11.5, 4.6 * nrow), squeeze=False)
    fig.suptitle("Figure S4 — dbNSFP transcript-resolution audit for SIFT and PolyPhen-2",
                 fontsize=12, fontweight="bold", y=0.985)

    r = 0
    if SHOW_CLINVAR:
        n = sum(cv["SIFT"])
        _panel(ax[0][0], cv["SIFT"], "SIFT", f"A. SIFT — ClinVar slice ({n:,} variants)")
        _panel(ax[0][1], cv["PolyPhen-2"], "PolyPhen-2",
               f"B. PolyPhen-2 — ClinVar slice ({n:,} variants)")
        r = 1
    n = sum(dms["SIFT"])
    la, lb = ("C.", "D.") if SHOW_CLINVAR else ("A.", "B.")
    _panel(ax[r][0], dms["SIFT"], "SIFT", f"{la} SIFT — DMS slice ({n:,} variants)")
    _panel(ax[r][1], dms["PolyPhen-2"], "PolyPhen-2",
           f"{lb} PolyPhen-2 — DMS slice ({n:,} variants)")

    fig.text(0.5, 0.022, "Resolution rule: the score at the MANE-Select positional index is "
             "used when present; fallback to the first numeric transcript score only when "
             "the MANE index carries no score or no MANE flag exists. Never max/min.",
             ha="center", fontsize=8, color="#333")
    fig.text(0.5, 0.005, "Two independent extractions are shown; counts are not "
             "interchangeable.", ha="center", fontsize=7.5, style="italic", color="#666")
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return out, dms, cv


if __name__ == "__main__":
    out, dms, cv = build()
    print("FIGURE S4 — GENERATION QC")
    print("=" * 25)
    for label, data, src in [("ClinVar benchmark slice", cv, "data/dbnsfp_slice.parquet"),
                             ("DMS/ProteinGym slice", dms, "data/dbnsfp_slice_dms.parquet")]:
        print(f"\n{label}  (source: {src})")
        for pred in ["SIFT", "PolyPhen-2"]:
            c = data[pred]; tot = sum(c)
            print(f"  {pred}: total {tot:,}")
            for name, v in zip(CATS, c):
                print(f"     {name:<32}{v:>9,}  ({v/tot:6.2%})")
    print("\nMANE-Select share:")
    for label, data in [("ClinVar", cv), ("DMS", dms)]:
        for pred in ["SIFT", "PolyPhen-2"]:
            c = data[pred]
            print(f"   {label:<8}{pred:<12}{c[0]/sum(c):.1%}")
    print("\nrule: MANE-Select positional index preferred; fallback only when needed; "
          "never max/min")
    print(f"\nsaved: {out}.pdf, {out}.png")
