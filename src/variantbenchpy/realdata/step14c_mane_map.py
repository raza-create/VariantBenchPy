"""STEP 14C — MANE Select GRCh38 reconstruction for ProteinGym DMS substitutions.

CDS offset is found by frame search requiring a clean single-ORF translation that
equals the CDS block length; CDS index -> genomic coordinate by strand-aware exon
walk. REF/ALT are complemented on the minus strand. No arbitrary choice is ever
made among multiple genomic realizations.
"""
from __future__ import annotations
import gzip, os, re
import pandas as pd

MANE = "data/mane"
GFF = f"{MANE}/MANE.GRCh38.v1.5.refseq_genomic.gff.gz"
RNA = f"{MANE}/MANE.GRCh38.v1.5.refseq_rna.fna.gz"
_B = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {b1 + b2 + b3: _AA[i * 16 + j * 4 + k]
         for i, b1 in enumerate(_B) for j, b2 in enumerate(_B) for k, b3 in enumerate(_B)}
AA2COD = {}
for c, a in CODON.items():
    AA2COD.setdefault(a, []).append(c)
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
STATES = ["UNIQUE_MANE_GRCH38_SNV", "MULTIPLE_MANE_GRCH38_SNVS", "MULTI_NUCLEOTIDE_REQUIRED",
          "NO_MANE_SELECT", "MANE_SEQUENCE_MISMATCH", "AMBIGUOUS_TARGET_ALIGNMENT",
          "REFERENCE_AA_MISMATCH"]


def translate(s):
    return "".join(CODON.get(s[i:i + 3], "X") for i in range(0, len(s) - 2, 3))


def load_cds_blocks():
    """gene -> {tx, chrom, strand, blocks[(start,end)]}"""
    genes = {}
    with gzip.open(GFF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            g = re.search(r"gene=([^;]+)", f[8])
            t = re.search(r"Parent=rna-([^;]+)", f[8])
            if not g or not t:
                continue
            d = genes.setdefault(g.group(1), {"tx": t.group(1), "chrom": f[0],
                                              "strand": f[6], "blocks": []})
            d["blocks"].append((int(f[3]), int(f[4])))
    for d in genes.values():
        d["blocks"].sort()
    return genes


def load_rna(txids):
    want, seqs, cur, buf = set(txids), {}, None, []
    with gzip.open(RNA, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur in want:
                    seqs[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if cur in want:
            seqs[cur] = "".join(buf)
    return seqs


def find_cds(rna, cds_len):
    for off in range(0, max(1, len(rna) - cds_len + 1)):
        cds = rna[off:off + cds_len]
        if not cds.startswith("ATG"):
            continue
        p = translate(cds)
        if p.endswith("*") and "*" not in p[:-1]:
            return off, cds, p[:-1]
    return None, None, None


def cds_index_to_genomic(i, blocks, strand, cds_len):
    """0-based CDS index -> 1-based GRCh38 position."""
    if strand == "+":
        rem = i
        for a, b in blocks:
            L = b - a + 1
            if rem < L:
                return a + rem
            rem -= L
    else:
        rem = i
        for a, b in reversed(blocks):
            L = b - a + 1
            if rem < L:
                return b - rem
            rem -= L
    return None


def map_assay(pg_seq, gene, genes, rnas, muts):
    d = genes.get(gene)
    if d is None:
        return None, "NO_MANE_SELECT", []
    cds_len = sum(b - a + 1 for a, b in d["blocks"])
    rna = rnas.get(d["tx"])
    if rna is None:
        return None, "NO_MANE_SELECT", []
    off, cds, prot = find_cds(rna, cds_len)
    if cds is None:
        return None, "MANE_SEQUENCE_MISMATCH", []
    if pg_seq == prot:
        shift, target = 0, "exact_full_length"
    elif pg_seq in prot:
        if prot.count(pg_seq) > 1:
            return None, "AMBIGUOUS_TARGET_ALIGNMENT", []
        shift, target = prot.index(pg_seq), "exact_fragment"
    else:
        return None, "MANE_SEQUENCE_MISMATCH", []

    out = []
    for mut in muts:
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z*])", str(mut))
        if not m:
            out.append({"mutant": mut, "state": "MANE_SEQUENCE_MISMATCH"}); continue
        a, p, b = m.group(1), int(m.group(2)), m.group(3)
        ci = (p - 1 + shift) * 3
        if ci + 3 > len(cds):
            out.append({"mutant": mut, "state": "MANE_SEQUENCE_MISMATCH"}); continue
        codon = cds[ci:ci + 3]
        if CODON.get(codon) != a:
            out.append({"mutant": mut, "state": "REFERENCE_AA_MISMATCH"}); continue
        realizations = []
        for k in range(3):
            for nt in "ACGT":
                if nt == codon[k]:
                    continue
                alt_cod = codon[:k] + nt + codon[k + 1:]
                if CODON.get(alt_cod) != b:
                    continue
                gpos = cds_index_to_genomic(ci + k, d["blocks"], d["strand"], len(cds))
                ref_g = codon[k] if d["strand"] == "+" else COMP[codon[k]]
                alt_g = nt if d["strand"] == "+" else COMP[nt]
                realizations.append((d["chrom"].replace("chr", ""), gpos, ref_g, alt_g))
        if not realizations:
            out.append({"mutant": mut, "state": "MULTI_NUCLEOTIDE_REQUIRED"}); continue
        uniq = sorted(set(realizations))
        state = "UNIQUE_MANE_GRCH38_SNV" if len(uniq) == 1 else "MULTIPLE_MANE_GRCH38_SNVS"
        for ch, gp, rf, al in uniq:
            out.append({"mutant": mut, "state": state, "chrom": ch, "pos": gp,
                        "ref": rf, "alt": al, "tx": d["tx"], "strand": d["strand"]})
    return target, None, out


def run(base="data/proteingym_v1.3", validate_brca2=True):
    ref = pd.read_csv(f"{base}/DMS_substitutions.csv")
    human = ref[ref["taxon"].astype(str).str.lower() == "human"].copy()
    genes = load_cds_blocks()
    rnas = load_rna([d["tx"] for d in genes.values()])

    tgt_counts = {"exact_full_length": 0, "exact_fragment": 0, "NO_MANE_SELECT": 0,
                  "MANE_SEQUENCE_MISMATCH": 0, "AMBIGUOUS_TARGET_ALIGNMENT": 0}
    var_counts = dict.fromkeys(STATES, 0)
    rows = []
    for _, r in human.iterrows():
        gene = str(r["UniProt_ID"]).split("_")[0]
        fn = r["DMS_filename"]
        path = next((os.path.join(root, fn) for root, _, fs in os.walk(base) if fn in fs), None)
        if path is None:
            continue
        d = pd.read_csv(path)
        mcol = "mutant" if "mutant" in d.columns else d.columns[0]
        single = d[~d[mcol].astype(str).str.contains(":")]
        target, err, mapped = map_assay(str(r["target_seq"]), gene, genes, rnas,
                                        single[mcol].astype(str).tolist())
        if err:
            tgt_counts[err] = tgt_counts.get(err, 0) + 1
            var_counts[err] = var_counts.get(err, 0) + len(single)
            continue
        tgt_counts[target] += 1
        smap = dict(zip(single[mcol].astype(str), single.get("DMS_score")))
        for mm in mapped:
            var_counts[mm["state"]] = var_counts.get(mm["state"], 0) + 1
            if mm["state"] == "UNIQUE_MANE_GRCH38_SNV":
                rows.append({"assay": r["DMS_id"], "protein": r["UniProt_ID"], "gene": gene,
                             "mutant": mm["mutant"], "DMS_score": smap.get(mm["mutant"]),
                             "chrom": mm["chrom"], "pos": mm["pos"], "ref": mm["ref"],
                             "alt": mm["alt"], "tx": mm["tx"]})
        print(f"  {r['DMS_id'][:44]:<44} {target}", flush=True)

    U = pd.DataFrame(rows).drop_duplicates(["assay", "mutant", "chrom", "pos", "ref", "alt"])
    U.to_parquet("data/dms_mane_snv_14c.parquet")

    L = ["STEP 14C — MANE/GRCh38 DMS MAPPING", "=" * 34, "",
         f"ProteinGym human assays:        {len(human)}",
         f"Unique proteins:                {human['UniProt_ID'].nunique()}",
         f"Single-AA substitutions:        {sum(var_counts.values()):,}", "",
         "Target mapping:",
         f"  exact MANE full-length:       {tgt_counts['exact_full_length']}",
         f"  exact MANE fragment:          {tgt_counts['exact_fragment']}",
         f"  no MANE Select:               {tgt_counts['NO_MANE_SELECT']}",
         f"  sequence mismatch:            {tgt_counts['MANE_SEQUENCE_MISMATCH']}",
         f"  ambiguous alignment:          {tgt_counts['AMBIGUOUS_TARGET_ALIGNMENT']}", "",
         "Variant mapping:"]
    for s in STATES:
        L.append(f"  {s:<32}{var_counts.get(s,0):>10,}")
    L += ["", "PRIMARY EXTERNAL BENCHMARK (UNIQUE_MANE_GRCH38_SNV only)",
          f"  unique GRCh38 SNVs:           {len(U):,}",
          f"  assays represented:           {U['assay'].nunique()}",
          f"  unique proteins:              {U['protein'].nunique()}"]

    if validate_brca2 and os.path.exists("data/brca2_dms_snv_clean.parquet"):
        gold = pd.read_parquet("data/brca2_dms_snv_clean.parquet")
        mine = U[U["assay"] == "BRCA2_HUMAN_Erwood_2022_HEK293T"]
        j = gold.merge(mine, left_on="aa_pg", right_on="mutant", how="left",
                       suffixes=("_mavedb", "_mane"))
        agree = int(((j["chrom_mavedb"].astype(str) == j["chrom_mane"].astype(str)) &
                     (j["pos_mavedb"] == j["pos_mane"]) &
                     (j["ref_mavedb"] == j["ref_mane"]) &
                     (j["alt_mavedb"] == j["alt_mane"])).sum())
        L += ["", "BRCA2 independent validation:",
              f"  MaveDB exact SNVs:            {len(gold)}",
              f"  MANE reconstructed:           {int(j['pos_mane'].notna().sum())}",
              f"  exact chr:pos:REF:ALT agreement: {agree}/{len(gold)}",
              f"  STATUS: {'PASS' if agree == len(gold) else 'FAIL'}"]
    L.append("\n-> data/dms_mane_snv_14c.parquet")
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
