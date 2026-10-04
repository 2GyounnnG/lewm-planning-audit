# Better Predictors, Not Better Plans

Code and sealed tables for **“Better Predictors, Not Better Plans: Auditing Planning Bottlenecks in a JEPA World Model.”** This repository is prepared as a private GitHub repository. Paper LaTeX source and PDFs, datasets, per-run trajectories, raw execution logs, checkpoints, and model weights are excluded. A Zenodo deposit is prepared separately; Reserved Zenodo DOI: `10.5281/zenodo.23134310`.

## Contents

- `code/`: R3--R8 and R8-FIX execution/statistics code and paper analysis helpers.
- `figures/`: figure-generation scripts and sealed tabular inputs used by the plots.
- `tables/`: sealed main/secondary tables, claim ledger, discrepancy record, and R8 corrections.
- `protocols/`: execution protocols, preregistration documents, sealed receipts, and SHA records.
- `manifests/`: small case/provenance manifests and package source indexes.

## Reproduction levels

1. **Sealed-table recomputation:** reproduce estimates from the CSV/JSON tables under `tables/`.
2. **Raw-result recomputation:** use raw-result archives in the prepared Zenodo package.
3. **Checkpoint inference:** use the separately prepared refit-checkpoint archive where redistribution is permitted.
4. **From-scratch training:** possible with the upstream projects and task data; outside this sealed audit package.

## Upstream pins and hardware

- `le-wm`: commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- `stable-worldmodel`: commit `abdced49809d5eae38e24b27dc7b635c502c4812`.
- Execution hardware: vast.ai, 8 × RTX 5090.

Code is released under the MIT License. The data and tables prepared for Zenodo are intended for CC BY 4.0. The reserved Zenodo DOI is `10.5281/zenodo.23134310`.
