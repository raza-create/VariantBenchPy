"""Join real predictor scores onto a ClinVar variant table by GRCh38 coordinates.

Keys everything on (chrom,pos,ref,alt) with 'chr'-prefixed contigs. Streams the big
score files once, keeping only rows whose key is in the ClinVar set (memory-safe).
"""
from __future__ import annotations
import gzip, io, zipfile
import pandas as pd


def _keyset(clinvar):
    return set(clinvar["variant_id"])  # "chr-pos-ref-alt"


def score_alphamissense(clinvar, path, chunk=2_000_000):
    keys = _keyset(clinvar)
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            chrom = c[0][3:] if c[0].startswith("chr") else c[0]
            k = f"{chrom}-{c[1]}-{c[2]}-{c[3]}"
            if k in keys:
                out[k] = float(c[8])
    return pd.Series(out, name="AlphaMissense")


def score_revel(clinvar, zip_path):
    keys = _keyset(clinvar)
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        # pick the largest member (the score table), whatever its name/extension
        name = max(names, key=lambda n: z.getinfo(n).file_size)
        with z.open(name) as raw:
            for i, line in enumerate(io.TextIOWrapper(raw, "utf-8")):
                if i == 0:
                    continue
                c = line.rstrip("\n").split(",")
                # chr,hg19_pos,grch38_pos,ref,alt,aaref,aaalt,REVEL,Ensembl_transcriptid
                pos38, rev = c[2], c[7]
                if pos38 == "." or rev == "":
                    continue
                chrom = c[0][3:] if c[0].startswith("chr") else c[0]
                k = f"{chrom}-{pos38}-{c[3]}-{c[4]}"
                if k in keys:
                    out[k] = max(out.get(k, -1.0), float(rev))
    return pd.Series(out, name="REVEL")


def score_dbnsfp(clinvar, dbnsfp_gz, cols):
    """Generic dbNSFP loader for CADD/SIFT/PolyPhen/EVE etc.
    cols: dict {out_name: (dbnsfp_colname, transform)} where transform maps raw->float."""
    keys = _keyset(clinvar)
    out = {name: {} for name in cols}
    with gzip.open(dbnsfp_gz, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ix = {h: i for i, h in enumerate(header)}
        ci = {"chr": ix.get("#chr", ix.get("hg38_chr")), "pos": ix.get("pos(1-based)", ix.get("hg38_pos(1-based)")),
              "ref": ix["ref"], "alt": ix["alt"]}
        want = {name: (ix.get(col), fn) for name, (col, fn) in cols.items() if ix.get(col) is not None}
        for line in fh:
            c = line.rstrip("\n").split("\t")
            _ch = c[ci['chr']]; _ch = _ch[3:] if _ch.startswith("chr") else _ch
            k = f"{_ch}-{c[ci['pos']]}-{c[ci['ref']]}-{c[ci['alt']]}"
            if k not in keys:
                continue
            for name, (j, fn) in want.items():
                v = c[j]
                if v not in (".", ""):
                    try:
                        out[name][k] = fn(v)
                    except ValueError:
                        pass
    return {name: pd.Series(d, name=name) for name, d in out.items()}
