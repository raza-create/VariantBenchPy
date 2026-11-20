"""STEP 13B — stream ClinVar VCV XML for temporal provenance of benchmark variants."""
from __future__ import annotations
import gzip
import xml.etree.ElementTree as ET
import pandas as pd


def extract(xml_gz, ids_tsv, out_parquet, progress=500000):
    m = pd.read_csv(ids_tsv, sep="\t", header=None, names=["variation_id", "variant_id"],
                    dtype=str)
    want = set(m["variation_id"])
    rows, n_seen = [], 0
    with gzip.open(xml_gz, "rb") as fh:
        ctx = ET.iterparse(fh, events=("end",))
        for _, elem in ctx:
            if not elem.tag.endswith("VariationArchive"):
                continue
            n_seen += 1
            vid = elem.get("VariationID")
            if vid in want:
                g = elem.find(".//Classifications/GermlineClassification")
                rows.append({
                    "variation_id": vid,
                    "vcv_accession": elem.get("Accession"),
                    "vcv_version": elem.get("Version"),
                    "vcv_date_created": elem.get("DateCreated"),
                    "vcv_date_last_updated": elem.get("DateLastUpdated"),
                    "germline_date_created": g.get("DateCreated") if g is not None else None,
                    "germline_most_recent_submission": g.get("MostRecentSubmission") if g is not None else None,
                    "germline_date_last_evaluated": g.get("DateLastEvaluated") if g is not None else None,
                })
            elem.clear()
            if n_seen % progress == 0:
                print(f"  scanned {n_seen:,} archives, matched {len(rows):,}", flush=True)
    df = pd.DataFrame(rows)
    df = df.merge(m, on="variation_id", how="left")
    df.to_parquet(out_parquet)
    print(f"scanned {n_seen:,} archives; matched {len(df):,}")
    return df


if __name__ == "__main__":
    extract("data/ClinVarVCVRelease_2026-08.xml.gz", "data/benchmark_variationids.tsv",
            "data/clinvar_temporal_provenance.parquet")
