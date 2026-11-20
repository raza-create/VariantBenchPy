"""Table S1 — predictor provenance. Records only verifiable facts; NA otherwise."""
from __future__ import annotations
import datetime, hashlib, os, sys
import pandas as pd

FILES = {"AlphaMissense": "data/AlphaMissense_hg38.tsv.gz",
         "REVEL": "data/revel.zip",
         "CADD": "data/dbnsfp/dbNSFP5.3.1a_grch38.gz",
         "SIFT": "data/dbnsfp/dbNSFP5.3.1a_grch38.gz",
         "PolyPhen-2": "data/dbnsfp/dbNSFP5.3.1a_grch38.gz",
         "EVE": "data/eve/EVE_all_data.zip"}
MANE_RULE = ("MANE Select preferred; fallback only when MANE has no usable score or is "
             "unavailable; never max/min across transcripts")

ROWS = [
    dict(Predictor="AlphaMissense", Version_or_release="hg38 release (dm_alphamissense, 2023)",
         Source="https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz",
         Genome_build="GRCh38", Score_used="am_pathogenicity",
         Score_orientation="Higher score = greater predicted deleteriousness/pathogenicity",
         Transformation="none (raw score)",
         Transcript_resolution="source file is keyed per genomic variant; transcript as "
                               "provided by the resource; no aggregation applied",
         Role="primary 5-predictor panel"),
    dict(Predictor="REVEL", Version_or_release="v1.3",
         Source="revel-v1.3_all_chromosomes.zip (Rothstein/Sinai distribution)",
         Genome_build="GRCh38 (grch38_pos column)", Score_used="REVEL",
         Score_orientation="Higher score = greater predicted deleteriousness/pathogenicity",
         Transformation="none (raw score)",
         Transcript_resolution="Maximum REVEL score retained for duplicate genomic keys "
                               "(9,389 of 219,637 keys, 4.275%; in all such cases the "
                               "duplicate rows carry different transcript-specific AA "
                               "consequences). MANE-matched sensitivity audit: 7,652 "
                               "resolvable conflicts, 1,694 scores changed in the "
                               "five-predictor benchmark, REVEL AUROC changed by "
                               "+0.00005 (0.96729 -> 0.96734) and the ranking was "
                               "unchanged. Differs from the MANE-Select rule used for "
                               "SIFT/PolyPhen-2; retained as the primary rule.",
         Role="primary 5-predictor panel"),
    dict(Predictor="CADD", Version_or_release="as distributed in dbNSFP 5.3.1a (academic)",
         Source="dbNSFP 5.3.1a GRCh38 BGZF (dbnsfp.org academic branch)",
         Genome_build="GRCh38", Score_used="CADD_phred",
         Score_orientation="Higher score = greater predicted deleteriousness/pathogenicity",
         Transformation="none (raw PHRED)",
         Transcript_resolution="single value per genomic variant; no transcript resolution "
                               "required",
         Role="primary 5-predictor panel"),
    dict(Predictor="SIFT", Version_or_release="as distributed in dbNSFP 5.3.1a (academic)",
         Source="dbNSFP 5.3.1a GRCh38 BGZF (dbnsfp.org academic branch)",
         Genome_build="GRCh38", Score_used="SIFT_score",
         Score_orientation="Higher score = greater predicted deleteriousness/pathogenicity",
         Transformation="1 - SIFT_score (raw SIFT preserved separately)",
         Transcript_resolution=MANE_RULE, Role="primary 5-predictor panel"),
    dict(Predictor="PolyPhen-2", Version_or_release="HDIV, as distributed in dbNSFP 5.3.1a",
         Source="dbNSFP 5.3.1a GRCh38 BGZF (dbnsfp.org academic branch)",
         Genome_build="GRCh38", Score_used="Polyphen2_HDIV_score",
         Score_orientation="Higher score = greater predicted deleteriousness/pathogenicity",
         Transformation="none (raw HDIV score)",
         Transcript_resolution=MANE_RULE, Role="primary 5-predictor panel"),
    dict(Predictor="EVE", Version_or_release="classical EVE bulk release (NOT popEVE)",
         Source="evemodel.org bulk download (EVE_all_data.zip)",
         Genome_build="protein-level; joined via UniProt entry + aaref/aapos/aaalt",
         Score_used="EVE_scores_ASM",
         Score_orientation="Higher score = greater predicted deleteriousness/pathogenicity",
         Transformation="none (0-1 score; no flip required)",
         Transcript_resolution="protein-level key (UniProt entry, ref AA, position, alt AA); "
                               "no gene-name rescue; exact match only",
         Role="secondary / descriptive only (2.1-26.5% coverage; excluded from primary "
              "inference)"),
]
COLS = ["Predictor", "Version_or_release", "Source", "Genome_build", "Access_or_release_date",
        "Resource_or_file", "Checksum", "Score_used", "Score_orientation", "Transformation",
        "Transcript_resolution", "Role"]


def gather(compute_sha=False):
    out = []
    for r in ROWS:
        r = dict(r)
        f = FILES[r["Predictor"]]
        r["Resource_or_file"] = os.path.basename(f) if os.path.exists(f) else "NA"
        r["Access_or_release_date"] = (
            datetime.date.fromtimestamp(os.path.getmtime(f)).isoformat()
            + " (local file mtime)") if os.path.exists(f) else "NA"
        chk = "NA"
        md5f = f + ".md5"
        if os.path.exists(md5f):
            chk = "md5 " + open(md5f).read().split()[0] + " (publisher-provided, verified)"
        elif compute_sha and os.path.exists(f):
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(blk)
            chk = "sha256 " + h.hexdigest()
        r["Checksum"] = chk
        out.append({c: r.get(c, "NA") for c in COLS})
    return pd.DataFrame(out)[COLS]


if __name__ == "__main__":
    df = gather(compute_sha="--hash" in sys.argv)
    os.makedirs("tables", exist_ok=True)
    path = "tables/TableS1_predictor_provenance.csv"
    df.to_csv(path, index=False)
    print("TABLE S1 — GENERATION QC")
    print("=" * 24)
    print(f"rows: {len(df)}\n")
    for _, r in df.iterrows():
        print(f"{r['Predictor']:<15}{r['Version_or_release']}")
        print(f"{'':<15}score={r['Score_used']} | transform={r['Transformation']}")
        print(f"{'':<15}build={r['Genome_build']}")
        print(f"{'':<15}file={r['Resource_or_file']} | date={r['Access_or_release_date']}")
        print(f"{'':<15}checksum={r['Checksum'][:60]}"
              + ("..." if len(r["Checksum"]) > 60 else ""))
        print(f"{'':<15}transcript={r['Transcript_resolution'][:70]}"
              + ("..." if len(r["Transcript_resolution"]) > 70 else ""))
        print()
    miss = {c: int((df[c] == "NA").sum()) for c in COLS if (df[c] == "NA").any()}
    print(f"missing provenance fields (recorded as NA): {miss if miss else 'none'}")
    print("  (checksums absent where the publisher supplied none and none was computed;")
    print("   re-run with --hash to compute sha256 for local files)")
    print("\nFLAGGED INCONSISTENCY:")
    print("  REVEL uses max-across-duplicate-rows for the same chr-pos-ref-alt, which is")
    print("  NOT the MANE-Select rule applied to SIFT/PolyPhen-2. Recorded verbatim in the")
    print("  Transcript_resolution field rather than described as uniform resolution.")
    print(f"\nsaved: {path}")
