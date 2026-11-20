"""STEP 13A — frozen predictor chronology.

Proxy dates only. Temporal exposure != training-data leakage: a variant absent from
ClinVar may still have appeared in HGMD, gnomAD, UniProt or publications used by a
predictor. Predictors without a defensible cutoff are excluded from strict
post-cutoff comparisons and reported descriptively only.
"""
CHRONOLOGY = {
    "AlphaMissense": {"version": "hg38 official (dm_alphamissense)",
                      "cutoff": "2023-09-22", "basis": "publication_date_proxy (Science, e-pub)",
                      "strict_eligible": True},
    "REVEL":         {"version": "v1.3",
                      "cutoff": "2016-09-22", "basis": "publication_date_proxy (AJHG online); "
                                                       "true HGMD/ESP training cutoff undocumented",
                      "strict_eligible": True},
    "PolyPhen-2":    {"version": "dbNSFP5.3.1a HDIV",
                      "cutoff": "2010-04", "basis": "publication_month_proxy (Nat Methods)",
                      "strict_eligible": True},
    "EVE":           {"version": "official EVE bulk (classical)",
                      "cutoff": "2021-10-27", "basis": "publication_date_proxy (Nature); "
                                                       "resource updated 2021-11-08 (distinct concept)",
                      "strict_eligible": True},
    "CADD":          {"version": "dbNSFP5.3.1a CADD_phred",
                      "cutoff": "unknown", "basis": "score release in dbNSFP undated "
                                                    "(original paper 2014-02, not a provenance cutoff)",
                      "strict_eligible": False},
    "SIFT":          {"version": "dbNSFP5.3.1a SIFT_score",
                      "cutoff": "unknown", "basis": "homology-based, no supervised training set "
                                                    "(classical paper 2003-07)",
                      "strict_eligible": False},
}


def report():
    L = ["STEP 13A — PREDICTOR CHRONOLOGY (frozen)", "=" * 40, "",
         f"{'Predictor':<15}{'Cutoff':<14}{'Strict':<8}Basis"]
    for k, v in CHRONOLOGY.items():
        L.append(f"{k:<15}{v['cutoff']:<14}{'YES' if v['strict_eligible'] else 'NO':<8}{v['basis']}")
    L += ["",
          "Strict post-cutoff comparisons: " +
          ", ".join(k for k, v in CHRONOLOGY.items() if v["strict_eligible"]),
          "Descriptive only (no defensible cutoff): " +
          ", ".join(k for k, v in CHRONOLOGY.items() if not v["strict_eligible"]),
          "",
          "NOTE: dates are publication proxies, not verified training cutoffs.",
          "  Analysis is 'temporal exposure-resistant', NOT 'leakage-free'."]
    return "\n".join(L)


if __name__ == "__main__":
    print(report())


# --- STEP 13B FROZEN SEMANTICS (validated 2026-09-03) ---
# germline_first_in_clinvar := VariationArchive/Classifications/GermlineClassification/@DateCreated
# Validation: 20/20 exact matches against ClinVar web "Germline -> First in ClinVar"
# (+ reference case VCV000051586 = 2014-04-01). Mismatches: 0. STATUS: PASS.
TEMPORAL_FIELD = "germline_date_created"
TEMPORAL_FIELD_MEANING = "germline First-in-ClinVar (VCV public date for germline classification)"
TEMPORAL_VALIDATION = {"checked": 20, "matches": 20, "mismatches": 0, "status": "PASS",
                       "release": "ClinVarVCVRelease_2026-08.xml.gz (md5 verified)",
                       "benchmark_coverage": 0.99933}
