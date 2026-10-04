# Source index — v1.1 / W9 (2026-10-04)

R3–R8 and R8-FIX content is retained from the previous release. The following additions/updates are copied byte-for-byte from the author's local sources. No experiment or model inference was run during release preparation.

| Source | Repository destination |
|---|---|
| `paper_w1/evidence/` | `tables/evidence/` (entire tree, including R10 and W6 post hoc analyses) |
| `paper_w1/CLAIM_EVIDENCE.csv` (198 rows), `DISCREPANCIES.md`, `REVISION_NOTES.md`, `BUILD_LOG.md`, `number_audit.json` | `tables/` |
| `paper_w1/scripts/*.py` | `code/paper_scripts/` |
| `paper_w1/figures/src/{make_figs.py,FIGURE_SOURCES.md,fig1_audit_overview.tex}` | `figures/source/` |
| `evidence/w6/W6_POSTHOC_TABLE.csv`, `evidence/r10/R10_MAIN_TABLE.csv` | `figures/data/{w6,r10}/` |
| `r10_execution/code/` and `modules/*/protocol/*.py` | `code/R10/` |
| `r10_execution/{prereg,ops}/`, `R10_SEAL.json`, `R10_DELIVERY.json`, `R10_CONCLUSION_ZH.txt` | `protocols/R10/` |

The R10 raw-result archive contains `raw/`, `modules/`, `tech/`, `diagnostics/`, `inputs/` and `ops/`; it excludes `transfer/`. R10 raw trajectories are not in GitHub. Paper manuscript source/PDF, checkpoints and datasets are excluded from GitHub. R9 is outside v1.1.

## Checksum interpretation

`checksums/<repository directory>/SHA256SUMS.txt` contains current exported-file hashes with repository-relative paths. Run a checksum check from the repository root. Original in-tree checksum lists and seals preserve their original source paths and contents; their coverage and exceptions are documented in `SOURCE_VERIFICATION_NOTES.md`. Dynamic Zenodo-package hash indexes are excluded from generated directory manifests to prevent archive/self-hash cycles.
