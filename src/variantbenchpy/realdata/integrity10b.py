"""STEP 10B — real predictor integration & score-integrity validation (no AUROC)."""
from __future__ import annotations
import pandas as pd

# frozen panel + provenance (fill versions as files are added)
PANEL = {
    "AlphaMissense": {"version": "hg38 (2023, dm_alphamissense)", "direction": "higher=pathogenic", "raw_transform": None},
    "REVEL":         {"version": "v1.3",                          "direction": "higher=pathogenic", "raw_transform": None},
    "CADD":          {"version": "dbNSFP5.3.1 (PHRED)",           "direction": "higher=pathogenic", "raw_transform": None},
    "SIFT":          {"version": "dbNSFP5.3.1",                   "direction": "lower=pathogenic",  "raw_transform": "1 - SIFT"},
    "PolyPhen-2":    {"version": "dbNSFP5.3.1 HDIV",              "direction": "higher=pathogenic", "raw_transform": None},
    "EVE":           {"version": "EVE (Bayesian VAE, separate)",  "direction": "higher=pathogenic", "raw_transform": None},
}


def validate(cv: pd.DataFrame, scores: pd.DataFrame, panel=PANEL):
    """cv indexed by variant_id with columns incl label, gene, stars.
    scores indexed by variant_id, one column per predictor (benchmark-oriented,
    higher=pathogenic). Returns the STEP 10B report string."""
    n = len(cv)
    preds = [p for p in panel if p in scores.columns]
    L = ["STEP 10B — REAL PREDICTOR INTEGRATION", "=" * 37, "",
         f"ClinVar benchmark variants: {n:,}", "",
         f"{'Predictor':<14}{'Version':<26}{'Coverage':>9}{'P/LP':>8}{'B/LB':>8}"]
    is_path = cv["label"] == 1
    for p in preds:
        s = scores[p]
        cov = s.notna().mean()
        pcov = s[is_path].notna().mean()
        bcov = s[~is_path].notna().mean()
        L.append(f"{p:<14}{panel[p]['version']:<26}{cov:>8.1%}{pcov:>8.1%}{bcov:>8.1%}")

    # shared subset
    shared = scores[preds].notna().all(axis=1)
    L += ["", f"Shared {len(preds)}-predictor subset: {int(shared.sum()):,} ({shared.mean():.1%})"]

    # score validity
    L += ["", "SCORE VALIDITY"]
    for p in preds:
        s = pd.to_numeric(scores[p], errors="coerce")
        bad = int((~s.dropna().apply(lambda x: -1e6 < x < 1e6)).sum())
        const = s.dropna().nunique() <= 1
        L.append(f"  {p:<14} min={s.min():.3g} max={s.max():.3g} "
                 f"nan={int(s.isna().sum()):,} invalid={bad} "
                 f"{'CONSTANT!' if const else ''}")

    # empirical orientation check (no threshold optimization): mean(path) > mean(benign)?
    L += ["", "SCORE DIRECTION (empirical: mean pathogenic vs mean benign)"]
    for p in preds:
        s = pd.to_numeric(scores[p], errors="coerce")
        mp, mb = s[is_path].mean(), s[~is_path].mean()
        ok = "PASS" if mp > mb else "CHECK"
        L.append(f"  {p:<14} path_mean={mp:.3f} benign_mean={mb:.3f}  {panel[p]['direction']:<18}{ok}")

    # coverage by gene
    L += ["", "COVERAGE BY GENE"]
    for p in preds:
        g = scores[p].notna().groupby(cv["gene"]).mean()
        L.append(f"  {p:<14} median_gene_cov={g.median():.1%}  genes<50%={int((g<0.5).sum()):,}")

    # coverage by evidence level
    L += ["", "COVERAGE BY EVIDENCE (star)"]
    for star in [0, 1, 2]:
        m = cv["stars"] >= star if star else cv["stars"] >= 0
        row = "  ".join(f"{p}={scores.loc[m, p].notna().mean():.0%}" for p in preds)
        lbl = "all" if star == 0 else f">={star}star"
        L.append(f"  {lbl:<9} {row}")

    warn = any("CONSTANT" in x or "CHECK" in x for x in L)
    L += ["", f"STATUS: {'PASS WITH WARNINGS' if warn else 'PASS'}"]
    return "\n".join(L)
