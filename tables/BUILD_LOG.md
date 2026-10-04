# BUILD_LOG

## Scope and provenance

- Main source: `main.tex`; supplementary source: `supplementary.tex`.
- Engine: local XeLaTeX via `latexmk -xelatex -interaction=nonstopmode -halt-on-error`; class options `elsarticle` `preprint,review,10pt`.
- No training, inference, simulation, or closed-loop execution was added. R4--R8 result directories were read only; figures and diagnostics were regenerated from copied sealed tables/raw values.
- The local archive contains 63 copied files from the external `/Volumes/MyProj` source tree, 1,638,848,610 bytes (about 1.526 GiB), under `source_copy/`; byte and SHA checks passed. The working provenance is therefore local and the external disk may be disconnected.
- R8 V2 local verification matched 11,303/11,303 files. Valid M1 E1/E3, M2, M3 real/repeated-frame, M4, M5 and M6 rows are used. The pre-fix M1 E2 and M3 PRED_PAST rows are superseded; the R8-FIX handling is recorded in `DISCREPANCIES.md` and the supplementary correction note.

## Numerical traceability

- `scripts/audit_numbers.py` (SHA-256 `4d140d00cc4e915ebd723a48245e1bd017878eadad8bc1b0d6d3182490e8a8be`) extracts numeric tokens from the main source, removes citation keys/labels/graphics metadata, and compares the remaining tokens with `CLAIM_EVIDENCE.csv`.
- Final audit: 389 numeric tokens, 385 matched to claim rows (98.97%); four unmatched tokens are non-claim metadata/layout values: the affiliation postcode `9500`, `92093`, and the checklist column widths `0.23`, `0.70`. The full occurrence list is retained in `number_audit.json`.
- `CLAIM_EVIDENCE.csv` SHA-256: see the latest R8-FIX addendum below.
- The claim ledger includes source paths and SHA-256 values for the original, R4--R7, and valid R8 endpoints, figures, and recomputed known-case rows. Invalid R8 endpoints are present only as `INVALID` rows without numerical estimates.

## Figures and tables

- `figures/src/make_figs.py` reads archived CSV/JSON values directly; current script SHA-256: `05ddff9aba40573f938ec12c0c496fd7c050364351acb7578d9e215d7c0d7094`.
- Figure 1 is the supplied TikZ overview. Figures 2--5 were regenerated after R8 integration. Figure 3c includes Reacher and PushT velocity rows; Figure 5d uses the 100-case M4 extension and states that the original 20 cases are included.
- Source paths and SHA-256 values are listed in `figures/FIGURE_SOURCES.md` and `CLAIM_EVIDENCE.csv`. Generated matplotlib labels were checked at a minimum of 6.5 pt.

## Compilation and layout

- Local XeLaTeX/BibTeX build completed successfully after the final edits. `main.pdf` is 25 pages total, with the article body and front/back matter through the declarations occupying 19 pages before the bibliography; the bibliography occupies the remaining pages. `supplementary.pdf` is 3 pages.
- Final SHA-256: `main.pdf` `386a3bd2e5c9fd9dbe8679fe580c33032dde5d0e3549d58449143cc200389183`; `supplementary.pdf` `5883ae45ec2b7a08f4afafbb1e9ee7d5b63919d32deea6e2bd18b640ba786b49`.
- The local review-mode build has no undefined references or fatal errors. It reports one 5.5 pt overfull line in Related Work and several bibliography URL underfull boxes; figures and tables fit their boxes. The KBS body-length target is met before references (19 pages), while the full PDF including references is 24 pages.
- The built-in editor compiler was run on the open `/Users/richwang/Documents/ChatGPT/热/paper_w1/main.tex`. It cannot compile this multi-file project in its standalone context because `sections/01_introduction.tex` is not visible to that compiler. The editor remains open on `main.tex`; the local multi-file XeLaTeX build is the verified compilation.

## Deliverables

- `REVISION_NOTES.md` records the numbered revision history through R8 integration.
- `DISCREPANCIES.md` records the invalid E2/PRED_PAST implementation results and no numerical value from either is used.
- `highlights.txt` and `cover_letter.md` are prepared; no submission or publication action was taken.

## R8-FIX rebuild addendum (2026-10-04)

- R8-FIX F1 and F2 files were copied into `evidence/r8/fix/` before use. No training, inference, simulation, or closed-loop execution was added.
- F1 corrected E2 source: `R8_M1_E2_CORRECTED_TABLE.csv`, SHA-256 `b538a12db542669051a465a9d50fe0507a7f4bc8861296cdf5c13b22c5812a36`; reported endpoints are +12.8 [9.9, 15.9] and +10.0 [7.4, 12.8] percentage points. Descriptive original rows are −2.3 and −2.6 points in the supplement only.
- F2 strict validation source: `TECH_GATE_STRICT.json`, SHA-256 `66260d1edf91412437ddef7ea923b654321fd805dfcd21263c9a04b91f842295`, and `TECH_GATE_ZHAT_ONLY.json`, SHA-256 `1b55790051fc028e082be16cfaa6aefa069463eb025cfc3c1ed62451819de30b`. The strict two-past +0.01 gate failed; PRED_PAST is not evaluated as a confirmatory endpoint. The 1,200 formal rows are labelled POSTHOC_AFTER_TECH_FAIL / INVALID_TECH_GATE and appear only as exploratory supplement evidence.
- Figure 4c was regenerated from the corrected E2 CSV. Current plotting script SHA-256 is `4c0ce9fac7a48ea6f8d14d5d88300a850e7746dbc7d558453639b79ac5c78a98`; `fig4_history.pdf` SHA-256 is `2602e0e467f78d54dc96363a97b7afb05852fb64207872b853b9ec40f82a2672`; `FIGURE_SOURCES.md` (SHA-256 `523c29f7ad08464f39c4edde96b2f6cb5232d8c46af7a129165eab9aaa46cc8f`) includes the E2 source and digest.
- Local `latexmk -xelatex -interaction=nonstopmode -halt-on-error` succeeds for `main.tex` and `supplementary.tex` with `elsarticle` `preprint,review,10pt`. Current outputs are 26 main pages (references begin on page 21, so 20 pages through the manuscript and declarations) and 5 supplementary pages. Main PDF SHA-256: `22a6266e0d6b3c852fda13a74b0d631bfae4b40ef2f5818a1f2903b3bf6b8667`; supplementary PDF SHA-256: `0ff67bb70d5e7417d0ffab1a259e5d43b97a78dc7c805a8ddf2e78f8ae23b83c`.
- `scripts/audit_numbers.py` rerun after the correction: 438 numeric tokens, 434 matched to `CLAIM_EVIDENCE.csv` (99.09%); the four unmatched tokens are affiliation postcode `9500`, `92093` and checklist layout widths `0.23`, `0.70`. `number_audit.json` SHA-256 is `6349035d5e4b4931a18fdf57e351075f7a26f6cc8498581fcf71d660264c5d83`; `CLAIM_EVIDENCE.csv` SHA-256 is `d5e656868644a92bec4ce7b1c179a6139b217654afe19aed78d218724fdec682`.
- The built-in editor compiler was run on the open `main.tex`. It fails before reaching the manuscript because the standalone Tectonic context cannot resolve the project-relative `sections/01_introduction.tex`; the local multi-file XeLaTeX build is the verified compilation. The editor remains open on `main.tex`.
- `highlights.txt` was checked after the R8-FIX pass; the five lines are 83, 60, 76, 66, and 77 characters respectively, all within the 85-character limit and containing no PRED_PAST exploratory endpoint.
- The final abstract is 230 whitespace-delimited words and fits on the first page; it contains the doubled-budget near-ceiling sentence and no PRED_PAST result.

## W3 figure/table rebuild (subtask)

- Rebuilt `figures/fig2_same_plans.pdf`, `figures/fig4_history.pdf`, and `figures/fig5_pusht_ranking.pdf` from `figures/src/make_figs.py`; the script reads the copied sealed R5/R8 CSVs, including `M4_ALL_RAW_VALUES.csv` (SHA-256 `292bcada4a48807cfc896532dcd1a5c071fc5292a8a024ade503e89bca72a9fe` in the local source ledger).
- Figure 2b now uses descriptive labels, solid primary markers, hollow extension markers, and a fresh-case history reference band. Figure 4c is the budget-by-offset interaction plot with corrected budget contrasts. Figure 5d reports pooled success rates for latent ranking, simulator-latent rescoring, and simulator-task rescoring.
- Table 1 removes repeated protocol columns; Tables 2--3 use descriptive names and omit the source column. The known-case Reacher row is +3.7 [−0.8, +8.4].
- Supplementary material includes the code-name crosswalk. Supplementary XeLaTeX build passed (5 pages).

## W3 final build and audit (2026-10-04)

- W3 changes were applied in the existing multi-file source. No training, inference, simulation, or closed-loop execution was added.
- The local review-mode XeLaTeX/BibTeX build succeeds for `main.tex` and `supplementary.tex`. The main PDF is 27 pages in review mode; the manuscript, declarations, and checklist run through page 22 and references run from page 23 through page 27. The supplementary PDF is 5 pages. The requested 17-page body/23-page total target is therefore not met under the required review spacing; the files remain readable and the limitation is recorded rather than hidden.
- The built-in editor compiler was run on the open `main.tex` and fails before the manuscript because its standalone Tectonic context cannot resolve the project-relative `sections/01_introduction.tex`. The local multi-file XeLaTeX build is the verified compilation; the editor remains open on `main.tex`.
- The final numeric audit reports 422 numeric tokens, 416 matched to `CLAIM_EVIDENCE.csv` (98.58%), and six unmatched metadata/layout tokens: affiliation postcodes `9500` and `92093`, and checklist width fractions `0.19`, `0.23`, `0.38`, and `0.14`. No scientific estimate is unmatched.
- Final SHA-256: `main.pdf` `5d1a9bd8dfbeea83f2ba21bffb84f21331489aacccc31cfdc5ff0decff24f478`; `supplementary.pdf` `5d75912a0a998ac68eeda4f26575562d06bf7650324029f71a8a401f3259ee7a`; `CLAIM_EVIDENCE.csv` `fcae439fa5a15413bfc49d383759b556f562ee9956aa6ace0c3ed9c0a690d35c`; `number_audit.json` `7ac1b73ed0471ae369a6f388d0f915a4dd5b3d06e72be885f3692637dd9c4bee`.
- Final figure-script SHA-256 is `17812190ea0fcf6168cf44ce7ca0a4f03c2a86365e8cc7aed9bef64e55f59375`. Figure PDFs: Figure 1 `a2081ac208c6901e34d48cdd83e170df19523d6a316a091697e59bfcf112f7b6`, Figure 2 `3faa50be2535495be9fe186182814ac70d44d47cd32657e6bbde843247ae0911`, Figure 3 `712a43c6e1d2e3706444ec82de8e5524d02594bda4e6d7d16fcf414e03097189`, Figure 4 `5283e73dd982edb7fd20ee16cf54b9bc29fce88d8307e14a63273e1ebe29e419`, and Figure 5 `3c3e4d951ab9f8c7fd28aa45d9f2038dce889713a86b436047ed06f4836f8b33`.
- The supplementary source has one previously observed 1.6-point overfull line in the R8 correction paragraph; it is wrapped with `sloppypar` in the final build. No main-PDF overfull boxes remain; bibliography URL underfull boxes are harmless.

## W3 final ledger refresh (2026-10-04)

- Added the sealed M5 post hoc first-plan claim to `CLAIM_EVIDENCE.csv`: 87.9% versus 40.9% over 1,068 paired runs, with the final −0.5-point descriptive contrast and M5 JSON/CSV/seal digests recorded together.
- `scripts/audit_numbers.py` was rerun after this ledger update. The audit remains 422 numeric tokens, 416 matched (98.58%); the six unmatched tokens are only affiliation postcodes `9500` and `92093` and checklist layout fractions `0.19`, `0.23`, `0.38`, and `0.14`.
- Refreshed hashes: `CLAIM_EVIDENCE.csv` `93e17f9692152742c3118f2291820a78038cc878fd05d1f16fc9dcda8ee26737`; `number_audit.json` `7ac1b73ed0471ae369a6f388d0f915a4dd5b3d06e72be885f3692637dd9c4bee`.

## W3 post-refresh build (2026-10-04)

- Rebuilt both outputs after the final ledger and abstract synchronization. Local `latexmk -g -xelatex -interaction=nonstopmode -halt-on-error` succeeds for the multi-file main and supplementary sources in `elsarticle` review mode.
- Current page counts are 27 for `main.pdf` and 5 for `supplementary.pdf`; no `Overfull \\hbox`, undefined-reference, or fatal-error diagnostics remain. Underfull table cells and bibliography URL lines remain layout warnings.
- Current output hashes: `main.pdf` `9d9ec74f3bf6b1266cb1fde10a2372181ad2c83513507abee0fd0eb0d6d4e82d`; `supplementary.pdf` `4d562b7237c64ec27968318fc564afa1b15a40eda385b230fc66ebd2c22f31d1`.

## Built-in editor recheck (2026-10-04)

- Rechecked the open `main.tex` with the built-in standalone compiler after the final source/ledger pass. It again stops at `\\input{sections/01_introduction}` because the standalone document context does not include the sibling `sections/` directory. No source replacement or alternate editor tab was created; the local multi-file XeLaTeX build above is the verified build.

## W3 provenance status refresh (2026-10-04)

- The historical `C_NUM_R5_RECOMPUTED` row is retained but marked `SUPERSEDED_W3`; the manuscript's active known-case interval is the sealed +3.7 [−0.8,+8.4] row. This status change does not alter the numeric audit: 422 tokens, 416 scientific/claim values matched, six metadata/layout tokens unmatched.
- Current ledger SHA-256: `CLAIM_EVIDENCE.csv` `77857d78d6a1fc5aede3548492b727120f32d119492bdfdc30afb3849b5620f7`; current audit SHA-256: `number_audit.json` `7ac1b73ed0471ae369a6f388d0f915a4dd5b3d06e72be885f3692637dd9c4bee`.

## Supplement M5 provenance refresh (2026-10-04)

- Added the post hoc first-plan rates (87.9% real history from the first call versus 40.9% single frame; 1,068 runs; final −0.5 points) to the R8 module-details paragraph so the supplement mirrors the main text and the claim ledger.
- Supplementary XeLaTeX rebuild succeeds; the supplement is now 6 pages. Current `supplementary.pdf` SHA-256: `d3c1517ecc651d89d00c515e61353ea63470c41dd6e9472dc0d3293afbb44a8c`.

## W4 build (Claude, 2026-10-04)

- Built in a Linux workspace with `latexmk -xelatex -interaction=nonstopmode -halt-on-error` (TeX Live, elsarticle.cls and elsarticle-num.bst copied into the source folder). Figures 2-5 regenerated with `figures/src/make_figs.py` (matplotlib 3.10, Liberation Sans) and copied to `figures/`.
- Pages: `main.pdf` 25 (main text ends on page 19; declarations page 20; references pages 21-25); `supplementary.pdf` 6. v5 was 27 pages.
- Diagnostics: no undefined references or citations. One 2.2 pt overfull box "while \output is active" on page 1 is pre-existing (also in the v5 log) and not visible.
- Numeric audit: 412 tokens, 406 matched (98.54%); the six unmatched tokens are layout and address values (float fractions 0.9 and 0.07, postcodes 9500 and 92093, checklist column widths 0.21 and 0.15).
- CLAIM_EVIDENCE.csv: 146 rows (six W4 rows added: TwoRoom/Cube query reach, PushT positive range, M4 rates recomputed from raw values, Reacher joint names, VP2 and Lambert literature checks).
- Final W4 hashes (after the data-citation change): `main.pdf` `695ab59a6d75d060be04db234a9d1ad04d4ab5bd66439b19133614094455f66f`; `supplementary.pdf` `3f5d4a6ff463364000e23cf774f678888896a8e9d91e2394cb3f0035d7ab36b6`; `main.tex` `4dbd0e96a761ac421ae4670065bf580e734de72252f43953ec01be1744219bb4`; `refs.bib` `e9bae0637a8321bf45214623528fa72ce471ba2b172d7b02d5958542ce59e565`; `CLAIM_EVIDENCE.csv` `1af19c43a1edef25c7e9cb68dfd0eea75812d644b8f6c13ade4542033ff439e6`; `number_audit.json` `4503d22c47ba8cc640155a24d948ff4667d71f6127a7de3ff05b482f48cf7596`; `figures/src/make_figs.py` `b0aebaa051e5e1ae91ac306079aaedf2031c14aac1c62bff58b5f4594123020b`.
- Figure PDFs: Fig. 2 `f5b72c8db7b8951f348142bc97629589ad5d62a70b0234c0e9a5a80e9990ba7e`; Fig. 3 `4e8da77ca2526efd434b8a69aee34f240b467171f127ccbb848e877863828544`; Fig. 4 `f6d23749c09601c1651a75df4ef3deab76259daa86863b07bc0514db1433db38`; Fig. 5 `e8ecab7c6aacd21297eeda27874f8dc0bdb9b769951e4f2bd62db3dcb7ed30c8`. Fig. 1 unchanged.
- A flat copy of the source (sections and tables inlined, figures in the same folder) compiles to text identical to this build; it is the `source/` folder of the submission package.
- Final hashes after the reference polish (these supersede the line above for the files listed): `main.pdf` `5151340ebaa16ab1bd6f0e8c8895e75c5c22327e4d0a674d20a1eb4e32309d4f`; `refs.bib` `f382c318c0708110bb1846b22b9eb805498efaa9135dce3c10156ce4165298c2`; `main.bbl` `67e1be16e0d5593f5d2db4dfc4618f6b03e7fe1226580166893240fde352d78e`. Page count unchanged (25).
- After inserting the GitHub URL and DOI (W4-15): `main.pdf` `7636e62bf49b139c00ad500a42fd71b52c88d1e975ffc33a7136de5011b407b9`; `main.tex` `11c2acdbac623a3efc6f469a83f3f6c3fc0e299c73f48b38b263c788222bf059`; `refs.bib` `3577af27dda13a34c841535364d3875355f1de0d68bddbc59a715d3e244d5370`; `main.bbl` `d151dc332b34ba7248d01a7dde5aa9704dfbf5a8da1ac653e8342baed94ca153`; `CLAIM_EVIDENCE.csv` `59fa83cf4b13d00279973ed3e1d278546b9cfb06e2736d1cfb926ce8fd704d23` (147 rows); `number_audit.json` `fe78e46df53823d1268d81501156d782f2a407c4ebb42f31355c1cf053574d21` (415 tokens, 409 matched; the six unmatched are the same layout and address values). Page count 25.

## W5 build (repositioning, 2026-10-04)

- Pages: `main.pdf` 27 (abstract and keywords fit on page 1; main text ends on page 20; references 51 entries). Supplement unchanged.
- Abstract 204 words. Highlights 67–82 characters.
- Numeric audit after W5: 412 tokens, 406 matched; the six unmatched tokens are the same layout and address values.
- Hashes: `main.pdf` `d165a4bcd9b8afee897c357d673fd862f603bf8b6f79349ea82f2b84095ec5b4`; `main.tex` `8da96182fad46f834c43da0b3555a79b7df66b4be42f1999ad78f434a4af0427`; `refs.bib` `184426057030c873bfaeabd103b5e48269c9b3de6b6ec7196e25ceb286a76c1a`; `main.bbl` `244cebbed93e395d7dbb376a64b0418e4bce3de66c553402df1e28f4471c6698`; `CLAIM_EVIDENCE.csv` `7b389418f4c490eed1d199113f4800724e7007954e0a22c1d73d5e543515fe05` (159 rows).


## W6 build (2026-10-04)

- Pages: `main.pdf` 33 (abstract 213 words and keywords on page 1; main text ends on page 25; 55 references); `supplementary.pdf` 8. The flat `source/` build (`~/work/flat_w6`) gives identical text for both documents.
- Diagnostics: no undefined references or citations; the only overfull box is the pre-existing 2.2 pt box on page 1 of each document.
- Numeric audit: 596 tokens, 590 matched; the six unmatched are the same layout and address values (0.9, 0.07, 9500, 92093, 0.21, 0.15).
- CLAIM_EVIDENCE.csv: 184 rows (25 W6 rows: interaction, pooled and history-set contrasts, family sensitivity, budget ceilings, interface, software versions, fresh-set selection and sealing times, denominators, D_total/menus, literature checks).
- Hashes (final W6, after the operating-point interaction rows): `main.pdf` `c6976fbc6a112967e676fb722c2fe1d754abf314ac72dee4dceb27dc7ef35fab`; `supplementary.pdf` `168cf001acfb82019f3fa266b6e0c56ad719d3999ede39bce8556310551f4893`; `main.tex` `0a4dcd6d94becc3033a6639ffda25ad059d2fd450a078c96cbad3172dc873084`; `refs.bib` `c123723f32d61b321ff6228297f5869be9a06330ee5c96b2c4d1e56f7b0d124c`; `CLAIM_EVIDENCE.csv` `db1613d077586ff4d3a706b389d6af130e4ca1a13ff69a05385f38deaa92df87` (188 rows); `number_audit.json` `8c165d8793a7ebcc71872338085e24071949da5a1aca351e2ff6082214abf113` (642 tokens, 636 matched; the unmatched are layout and address values plus the `-6` of a `\cmidrule{5-6}` column range); `evidence/w6/W6_POSTHOC_TABLE.csv` `65f837398a8853c9f75518bdba5513dac661972ea1fa3e91c1207f77d2f007e4`; `evidence/w6/W6_INTERACTION_OPERATING_POINTS.csv` `8a6f7a1ffc9144536cfcea0ef0cc7be6640961121c0ce8ab518a709677b50494`; `evidence/w6/W6_FAMILY_SENSITIVITY.json` `e1232de4312818b9f15e94bdc8b0740eb0bbd1697f8ce554d69abf6da9b611ae`; `figures/fig1_audit_overview.pdf` `8003a3f4300ecb06639de9a43af48bbdf36d49cac58557243b8920765927e083`; `figures/fig2_same_plans.pdf` `63fc140dd28a2f2965cb4dd45d1c680006a73cbd70f096dc79b749c6b6550e6c`.


## W7 build (2026-10-04, R10 merged)

- Pages: `main.pdf` 35 (abstract 210 words and keywords on page 1; main text ends on page 28; 55 references); `supplementary.pdf` 9.
- Diagnostics: no undefined references or citations; the only overfull box is the pre-existing 2.2 pt box on page 1 of each document. The flat `source/` build gives identical text for both documents.
- Numeric audit: 707 tokens, 699 matched; unmatched are the same layout and address values and the `-6` of a `\\cmidrule` range.
- CLAIM_EVIDENCE.csv: 198 rows (10 R10 rows).
- Hashes: `main.pdf` `1915cb58d309138b1cf647db5ced27781f286ea5306a41472619ac39fa0d02f4`; `supplementary.pdf` `7d708d6cf0e6cba4e6bb893fae9215b9dcab6e03aff4840c45800df1cf45a078`; `main.tex` `81dd5878e7b55158bc03547fe80bcfd53ed04f671e4b1d1e2d71ceafc9f64c40`; `CLAIM_EVIDENCE.csv` `27a3576d3144b4dc0bc9f4de3e65065f6fa2309d783084846b50318066f04d39`; `number_audit.json` `d7636c1bf825a4313d3c4bf4964d2fc06fa2d2514534d9da20f7bf8b2669d69b`; `evidence/r10/R10_MAIN_TABLE.csv` `fc8c2a3dbfbce4a494333193fca2549bc6922ad7f61bd6c874cc5db69f3aae04`; `figures/fig1_audit_overview.pdf` `74a85cfe99cbf55c7f3db1345be7e85bf157869d81f2c2024df10c6182a2bf1c`; `figures/fig4_history.pdf` `7c2b60199623f9ed80948e048bf60ff97adc99f0fd0da7e46385f7f49b42cbc5`; `figures/src/make_figs.py` `74cffe2e28e07ed73740c9e03c1350b97b3e47c4a17cd6235320887089cb5b2e`.


## W8 build (2026-10-04)

- Pages: `main.pdf` 36 (abstract 212 words and keywords on page 1); `supplementary.pdf` 9. No `\resizebox`; the only overfull box is the pre-existing 2.2 pt box on page 1 of each document. Flat build identical in text.
- Numeric audit: 702 tokens, 693 matched; unmatched are layout values (incl. the 2.5 pt `\tabcolsep`), address values and the `-6` of a `\cmidrule` range.
- Hashes: `main.pdf` `c6422d89cfc945560dfea48fa8a4a7e4beb21fb5421e5ddb336359a19a03d452`; `supplementary.pdf` `beadfb988a78276b26d12cfe178b3a853673e16877f6274cb598c5f884b01486`; `main.tex` `b11f8541cf887de075c443ab144aad422f7d7c9ab0bb0beee094c66869b1b725`; `refs.bib` `9424a805d0ad590aadd3deb81b7617cf2abd5aa013f8cd18a5efd56a9b7c62b6`; `CLAIM_EVIDENCE.csv` `27a3576d3144b4dc0bc9f4de3e65065f6fa2309d783084846b50318066f04d39`; `tables/table2_main.tex` `be26e1447fa3ae7939ac0118b26283fa2c17032e811ef89b73955af1c0be3be3`; `tables/table3_replanning.tex` `9ec0376159c9063e4a63f494c4ae90b6d6d546c9ff6d8d488ca5f22b17f39301`.


## W9 build (2026-10-04)

- Pages: `main.pdf` 36 (abstract 210 words and keywords on page 1); `supplementary.pdf` 9; flat build identical in text; numeric audit unchanged (702 tokens, 693 matched).
- Hashes: `main.pdf` `69431ac9c68f742f8c01f59bf0db3650414d910d9d7a390ff29465c93a7c7b1b`; `supplementary.pdf` `beadfb988a78276b26d12cfe178b3a853673e16877f6274cb598c5f884b01486`; `main.tex` `3cf6d58a7fe9214201f1c0ce2fe41a4d5a420177e57666df6bf5e5bd28816f3f`; `refs.bib` `3c150adf8c4130686a6246b96bb4adc7c01aff37bd09303617f97dea8dbf2a05`.
- After the title change: `main.pdf` `7c05e4afc8960bb23aa8ba9299ab2d35d3c2b42a47961ec005ecac2e85589749`; `supplementary.pdf` `106f618cbeab4b3ba7889563d6b87ce54c9e9eaa011457da26edae3fcadcbf01`; `main.tex` `a7e9b8e6b9d011ee32df2ed06aa4e67bb7466343e57a03188f0944bbef269909`; `refs.bib` `daeb98bd92fd88bb7ad698f786e4e24ff54fde81ef0033299557014ad438e0b8`. Page count 36; abstract and keywords on page 1.
