# VariantBenchPy

Code accompanying *VariantBenchPy: benchmark design as a measurable source of
uncertainty in variant-effect predictor evaluation* (submitted to *Genome Biology*).

VariantBenchPy evaluates variant-effect predictors while treating the benchmark
definition itself as a source of uncertainty. It provides gene-clustered
bootstrap inference, a prespecified grid of benchmark designs, auditing of the
selection induced by predictor coverage, selection-aware (cross-fitted) tests of
design-induced changes in relative performance, and an exact within-gene and
between-gene decomposition of AUROC.

## Installation

```bash
git clone https://github.com/raza-create/VariantBenchPy.git
cd VariantBenchPy
pip install -e .
```

Python 3.10 or later is required.

## Reproducing the main figures

From the repository root:

| Manuscript | Script          | Output                  |
|------------|-----------------|-------------------------|
| Figure 1   | `plot_fig2.py`  | `Fig/fig2.png`, `.pdf`  |
| Figure 2   | `plot_fig3.py`  | `Fig/fig3.png`, `.pdf`  |
| Figure 3   | `plot_fig4.py`  | `Fig/fig4.png`, `.pdf`  |
| Figure 4   | `plot_fig5.py`  | `Fig/fig5.png`, `.pdf`  |

Script and file numbers are one higher than the manuscript's figure numbers
because an earlier overview figure was removed. Each script redraws its figure
from the values in `figures/nature/source_data/`, checks the values reported in
the manuscript, and stops with an error if any disagree.

Supplementary figures S1-S6 and tables S1-S4 are in `supplementary/`, with
SHA-256 checksums in `supplementary/MANIFEST.txt` and `figures/FINAL_MANIFEST.txt`.

## Input data

Input data are not distributed here. The analysis uses ClinVar (GRCh38 VCF and
VCV XML, release 16 August 2026), AlphaMissense (official GRCh38 release),
REVEL v1.3, dbNSFP v5.3.1a (academic release), classical EVE, ProteinGym v1.3,
MaveDB and MANE v1.5; sources and versions are listed in Table S1. Scores
derived from dbNSFP are not redistributed, in keeping with its licence. The
real-data pipeline in `src/variantbenchpy/realdata/` reads these inputs from a
local `data/` directory.

## Repository layout

- `src/variantbenchpy/` - the package: harmonization, metrics, evaluation
  (reliability audit and selection-aware inference), coverage and missingness
  diagnostics, simulation, and real-data pipelines
- `plot_fig2.py` to `plot_fig5.py` - main-figure scripts
- `figures/nature/source_data/` - every value plotted in the main figures
- `figures/`, `supplementary/`, `tables/` - supplementary figures and tables

## Licence

MIT; see `LICENSE`.

## Citation

If you use VariantBenchPy, please cite the accompanying article (details will
be added on publication) and the archived release on Zenodo.
