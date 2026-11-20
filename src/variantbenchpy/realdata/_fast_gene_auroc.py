"""Vectorized per-gene macro AUROC via Mann-Whitney rank-sum identity (no apply/loops)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def gene_macro_auroc(y, s, g, min_n=1, min_per_class=1):
    m = ~np.isnan(np.asarray(s, dtype="float64"))
    y = np.asarray(y)[m].astype("int8")
    s = np.asarray(s, dtype="float64")[m]
    g = np.asarray(g)[m]
    if y.size == 0:
        return np.nan
    df = pd.DataFrame({"y": y, "s": s, "g": g})
    df["r"] = df.groupby("g")["s"].rank(method="average")
    df["rpos"] = np.where(df["y"] == 1, df["r"], 0.0)
    agg = df.groupby("g").agg(
        npos=("y", "sum"),
        n=("y", "size"),
        srp=("rpos", "sum"),
    )
    agg["nneg"] = agg["n"] - agg["npos"]
    valid = (agg["npos"] >= min_per_class) & (agg["nneg"] >= min_per_class) & (agg["n"] >= min_n)
    a = agg[valid]
    auc = (a["srp"] - a["npos"] * (a["npos"] + 1) / 2.0) / (a["npos"] * a["nneg"])
    return float(auc.mean()) if len(auc) else np.nan
