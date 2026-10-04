# Better Predictors, Not Better Plans

Code and archived evidence for **Better Predictors, Not Better Plans: Auditing Planning Bottlenecks in a JEPA World Model**, manuscript W7 (2026-10-04), software release **v1.1**. Repository: https://github.com/2GyounnnG/lewm-planning-audit. Reserved Zenodo DOI: **10.5281/zenodo.23134310**. Paper LaTeX/PDF, model weights, datasets and per-run trajectories are distributed separately or excluded from this repository.

## Contents

- `code/`: R3–R8, R8-FIX and R10 execution, statistics and sealing code; `code/paper_scripts/` contains the W6/W7 analysis and audit scripts.
- `tables/`: the current paper's complete evidence tree, the 198-row claim ledger, discrepancies, revision notes, build log and number audit.
- `figures/source/`: plotting scripts and the standalone TikZ overview; `figures/data/` includes the R10 and W6 plot inputs.
- `protocols/`: execution and preregistration records, seals and receipts, including R10.
- `manifests/`: source provenance and checksums for this distribution. Original sealed checksum files are retained verbatim; generated publication checksums live under `manifests/checksums/`.

R10 modules:
- **A — Native history option:** official Reacher predictor with the library's native `history_len=3` versus `history_len=1` (the optional refit extension was not run).
- **B — TwoRoom history:** real-history replanning versus paired same-machine single-frame controls on the fresh case set.
- **C — Cube history:** real-history replanning versus paired same-machine single-frame controls, retaining the documented nonexact symmetric reset limitation.
- **D — Frame/action controls:** real frames with null actions and current-frame copies with executed actions; these unpaired inputs rank interventions without isolating a unique mechanism.

R10 delivered all 9,000 planned new episode runs. The endpoints are success-rate differences after averaging model/stream values within each case, with 5,000 case-bootstrap replicates and percentile 95% intervals. Same-machine controls for B/C/D were registered before formal execution. W6/W7 post hoc analyses include predictor-by-history interaction, pooling across case sets, operating-point interactions and family-cluster sensitivity; they are labeled separately from preregistered results.

## Reproduction levels

1. **From archived tables:** use `tables/evidence/` and the scripts in `code/paper_scripts/`.
2. **From original result values:** use the Zenodo raw-result archives and the round-specific execution/statistics code. R10 per-run trajectories are in `raw_results_R10.tar.gz`, not GitHub.
3. **Checkpoint inference:** use the separately supplied refit checkpoints and the pinned upstream projects and task data.
4. **From-scratch training:** use the upstream projects and task data; training is outside this release task.

The paper scripts preserve the original paper layout: run them from a working directory containing `evidence/` (a copy or link to `tables/evidence/`). `make_figs.py` preserves its source layout expectations; see `figures/source/FIGURE_SOURCES.md`. Some manuscript-number checks additionally require the privately held manuscript source, which is excluded here.

## Upstream pins and hardware

- `le-wm`: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- `stable-worldmodel`: `abdced49809d5eae38e24b27dc7b635c502c4812` for the original audit and all modules except R10 A.
- **R10 module A only:** `stable-worldmodel` `63988116d34cde56aea1240d5e58eb158ac67dc0`, which supplies the native history option. Cross-version single-frame differences are descriptive.
- Original audit hardware: vast.ai, **8 × RTX 5090**.
- R10 hardware: vast.ai, **8 × NVIDIA GeForce RTX 4090**, as recorded in `protocols/R10/ops/R10_LAUNCH.json`.

## Licenses and archive

Audit code is MIT; data and tables are CC BY 4.0. Upstream components retain their applicable licenses. Zenodo DOI: `10.5281/zenodo.23134310`. The Zenodo upload remains a draft until the author publishes it. R9/PreJEPA results and weights are outside this release.
