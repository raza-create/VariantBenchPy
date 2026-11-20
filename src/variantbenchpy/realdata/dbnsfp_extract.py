"""Extract only the benchmark ClinVar variants from a frozen dbNSFP 5.3.1a archive."""
from __future__ import annotations
import gzip, hashlib, io, json, zipfile, datetime
import pandas as pd

WANT = {
    "chr": ["#chr", "hg38_chr", "chr"],
    "pos": ["pos(1-based)", "hg38_pos(1-based)", "pos"],
    "ref": ["ref"], "alt": ["alt"],
    "tx":  ["Ensembl_transcriptid"],
    "gene": ["genename"],
    "CADD": ["CADD_phred"],
    "SIFT_raw": ["SIFT_score"],
    "PolyPhen2_raw": ["Polyphen2_HDIV_score"],
    "mane": ["MANE_transcript", "MANE_Select", "Ensembl_MANE_Select"],
}


def _idx(header, names):
    for n in names:
        if n in header:
            return header.index(n)
    return None


def _resolve_tx(raw_scores, tx_list, mane_tx):
    vals = raw_scores.split(";")
    txs = tx_list.split(";") if tx_list else []
    def num(v):
        try:
            return float(v)
        except ValueError:
            return None
    if mane_tx and txs:
        for v, t in zip(vals, txs):
            if t and mane_tx.split(".")[0] == t.split(".")[0] and num(v) is not None:
                return num(v), t, "MANE"
    seq = list(zip(vals, txs)) if txs else [(vals[0] if vals else "", "")]
    for v, t in seq:
        if num(v) is not None:
            return num(v), t, "first-listed"
    return float("nan"), "", "unresolved"


def extract(zip_path, clinvar_keys, out_parquet, manifest_path):
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if ("variant" in m.lower() and ".chr" in m)]
        if not members:
            members = [m for m in z.namelist() if ".chr" in m and (m.endswith(".gz") or "variant" in m.lower())]
        rows, cols = [], None
        for mem in sorted(members):
            with z.open(mem) as raw:
                fh = gzip.open(raw, "rt") if mem.endswith(".gz") else io.TextIOWrapper(raw, "utf-8")
                header = fh.readline().rstrip("\n").split("\t")
                if cols is None:
                    cols = {k: _idx(header, v) for k, v in WANT.items()}
                    miss = [k for k in ("chr", "pos", "ref", "alt") if cols[k] is None]
                    if miss:
                        raise SystemExit(f"dbNSFP header missing {miss}; first cols={header[:10]}")
                for line in fh:
                    c = line.rstrip("\n").split("\t")
                    ch = c[cols["chr"]].replace("chr", "")
                    key = f"{ch}-{c[cols['pos']]}-{c[cols['ref']]}-{c[cols['alt']]}"
                    if key not in clinvar_keys:
                        continue
                    mane_tx = c[cols["mane"]] if cols["mane"] is not None else ""
                    tx = c[cols["tx"]] if cols["tx"] is not None else ""
                    sr_raw = c[cols["SIFT_raw"]] if cols["SIFT_raw"] is not None else "."
                    pp_raw = c[cols["PolyPhen2_raw"]] if cols["PolyPhen2_raw"] is not None else "."
                    sv, st, srr = _resolve_tx(sr_raw, tx, mane_tx)
                    pv, pt, prr = _resolve_tx(pp_raw, tx, mane_tx)
                    rows.append({"variant_id": key,
                                 "CADD": c[cols["CADD"]] if cols["CADD"] is not None else ".",
                                 "SIFT_raw": sr_raw, "PolyPhen2_raw": pp_raw,
                                 "SIFT_resolved": sv, "SIFT_tx": st, "SIFT_reason": srr,
                                 "PolyPhen2_resolved": pv, "PolyPhen2_tx": pt, "PolyPhen2_reason": prr})
    df = pd.DataFrame(rows).drop_duplicates("variant_id")
    df["CADD"] = pd.to_numeric(df["CADD"], errors="coerce")
    df["PolyPhen-2"] = pd.to_numeric(df["PolyPhen2_resolved"], errors="coerce")
    df["SIFT"] = 1.0 - pd.to_numeric(df["SIFT_resolved"], errors="coerce")
    df.to_parquet(out_parquet)
    manifest = {
        "dbnsfp_archive": str(zip_path),
        "extraction_date": datetime.date.today().isoformat(),
        "requested_columns": WANT,
        "match_rule": "exact chr-pos(1based)-ref-alt on GRCh38 (chr stripped)",
        "transcript_rule": "MANE transcript score if present else first-listed; never max/min",
        "orientation": {"CADD": "raw", "PolyPhen-2": "raw", "SIFT": "1 - SIFT_score"},
        "out_parquet_sha256": hashlib.sha256(open(out_parquet, "rb").read()).hexdigest(),
        "n_extracted": int(len(df))}
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"extracted {len(df):,} variants -> {out_parquet}")
    print(f"SIFT unresolved: {(df['SIFT_reason']=='unresolved').sum():,} | "
          f"PolyPhen unresolved: {(df['PolyPhen2_reason']=='unresolved').sum():,}")
    return df


if __name__ == "__main__":
    import sys
    keys = set(l.strip() for l in open(sys.argv[2]))
    extract(sys.argv[1], keys, "data/dbnsfp_slice.parquet", "data/dbnsfp_slice.manifest.json")
