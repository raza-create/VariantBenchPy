"""Tabix-slice dbNSFP 5.3.1a (GRCh38 BGZF) to the benchmark ClinVar variants."""
from __future__ import annotations
import hashlib, json, subprocess, datetime
import pandas as pd

C = {"chr": 0, "pos": 1, "ref": 2, "alt": 3, "aaref": 4, "aaalt": 5,
     "aapos": 13, "gene": 14, "tx": 16, "uniprot": 18, "uentry": 19,
     "mane": 28, "sift": 48, "pp2": 54, "cadd": 154}


def _num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _resolve(raw, tx_field, mane_field):
    """dbNSFP transcript-aligned fields: the k-th ';'-value corresponds to the k-th
    Ensembl_transcriptid; MANE field holds 'Select' at the MANE transcript's index.
    Rule: SIFT/PolyPhen at the 'Select' index -> else first numeric (fallback)."""
    vals = raw.split(";")
    txs = tx_field.split(";") if tx_field else []
    manes = mane_field.split(";") if mane_field else []
    # 1) MANE Select positional index
    for i, m in enumerate(manes):
        if m == "Select" and i < len(vals) and _num(vals[i]) is not None:
            return _num(vals[i]), (txs[i] if i < len(txs) else ""), "MANE_Select"
    # 2) MANE present but its index has no numeric score -> ambiguous
    has_mane = any(m == "Select" for m in manes)
    numeric = [(i, _num(v)) for i, v in enumerate(vals) if _num(v) is not None]
    if not numeric:
        return float("nan"), "", "no_usable_score"
    if has_mane:
        i, v = numeric[0]
        return v, (txs[i] if i < len(txs) else ""), "MANE_index_no_score->fallback"
    # 3) no MANE flag at all -> canonical/first-numeric fallback
    i, v = numeric[0]
    return v, (txs[i] if i < len(txs) else ""), "no_MANE->fallback"


def _regions(keys):
    seen = set()
    for k in keys:
        ch, pos, _r, _a = k.split("-", 3)
        seen.add((ch, int(pos)))
    return sorted(seen, key=lambda x: (x[0], x[1]))


def extract(bgzf, clinvar_keys, out_parquet, manifest_path, batch=4000):
    keyset = set(clinvar_keys)
    regions = _regions(clinvar_keys)
    rows = []
    for i in range(0, len(regions), batch):
        chunk = regions[i:i + batch]
        args = ["tabix", bgzf] + [f"{ch}:{p}-{p}" for ch, p in chunk]
        out = subprocess.run(args, capture_output=True, text=True).stdout
        for line in out.splitlines():
            c = line.split("\t")
            if len(c) <= C["cadd"]:
                continue
            key = f"{c[C['chr']]}-{c[C['pos']]}-{c[C['ref']]}-{c[C['alt']]}"
            if key not in keyset:
                continue
            sift_raw, pp_raw = c[C["sift"]], c[C["pp2"]]
            sv, st, sr = _resolve(sift_raw, c[C["tx"]], c[C["mane"]])
            pv, pt, pr = _resolve(pp_raw, c[C["tx"]], c[C["mane"]])
            rows.append({"variant_id": key, "dbnsfp_gene": c[C["gene"]],
                         "uniprot": c[C["uniprot"]].split(";")[0],
                         "uniprot_entry": c[C["uentry"]].split(";")[0],
                         "aaref": c[C["aaref"]], "aaalt": c[C["aaalt"]],
                         "aapos": c[C["aapos"]].split(";")[0],
                         "CADD_phred_raw": c[C["cadd"]],
                         "SIFT_raw": sift_raw, "PolyPhen2_HDIV_raw": pp_raw,
                         "SIFT_resolved": sv, "SIFT_tx": st, "SIFT_reason": sr,
                         "PolyPhen2_resolved": pv, "PolyPhen2_tx": pt, "PolyPhen2_reason": pr})
    df = pd.DataFrame(rows).drop_duplicates("variant_id")
    df["CADD"] = pd.to_numeric(df["CADD_phred_raw"], errors="coerce")
    df["PolyPhen-2"] = pd.to_numeric(df["PolyPhen2_resolved"], errors="coerce")
    df["SIFT"] = 1.0 - pd.to_numeric(df["SIFT_resolved"], errors="coerce")
    df.to_parquet(out_parquet)
    manifest = {
        "dbnsfp_source": bgzf, "branch": "5.3.1a academic", "genome": "GRCh38",
        "extraction_date": datetime.date.today().isoformat(),
        "columns_1based": {"chr":1,"pos":2,"ref":3,"alt":4,"genename":15,
                           "Ensembl_transcriptid":17,"MANE":29,"SIFT_score":49,
                           "Polyphen2_HDIV_score":55,"CADD_phred":155},
        "match_rule": "exact chr-pos-ref-alt (bare contigs, GRCh38)",
        "transcript_rule": "MANE transcript score else first numeric; never max/min",
        "orientation": {"CADD":"raw PHRED","PolyPhen-2":"raw HDIV","SIFT":"1 - SIFT_score"},
        "out_parquet_sha256": hashlib.sha256(open(out_parquet,"rb").read()).hexdigest(),
        "n_extracted": int(len(df))}
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"extracted {len(df):,} / {len(keyset):,} benchmark variants")
    print("TRANSCRIPT RESOLUTION (SIFT | PolyPhen-2):")
    cats = ["MANE_Select", "MANE_index_no_score->fallback", "no_MANE->fallback", "no_usable_score"]
    for cat in cats:
        s_n = int((df["SIFT_reason"] == cat).sum())
        p_n = int((df["PolyPhen2_reason"] == cat).sum())
        print(f"  {cat:<32} {s_n:>8,} | {p_n:>8,}")
    return df


if __name__ == "__main__":
    import sys
    keys = [l.strip() for l in open(sys.argv[2]) if l.strip()]
    extract(sys.argv[1], keys, "data/dbnsfp_slice.parquet", "data/dbnsfp_slice.manifest.json")
