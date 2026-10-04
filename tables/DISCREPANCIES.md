# DISCREPANCIES

No unresolved numerical discrepancy was used to select a favorable value. The package seed ledger was treated only as a locator; every numerical claim in `CLAIM_EVIDENCE.csv` was checked against an archived table or raw-value file.

Known scope differences retained rather than reconciled:

- The R5 merged tables carry source-table SHA fields from the earlier R4 delivery; the manuscript cites the local merged-table SHA and preserves the embedded source SHA fields.
- The R5 real-history replanning table reports the known-case analysis; the fresh-case endpoint is recomputed from the R6 raw/table pair and is reported separately. The harmonized known-case recomputation is rounded to one decimal point in the manuscript.
- Cube planner-randomness cells are missing after the technical gate. They are marked technically unevaluable and never treated as zero success.
- R7 stress and random-action tables are now copied locally and used directly. The stress test changes target distance and budget together, so their separate effects remain unidentified.
- R4 CPU recovery failures and R5 recovery failures are retained as limitations; no tolerance, precision, batch size, or backend was changed.
- Approximate bars read from the original LeWM Figure 6 were removed from Table 1; the compact protocol table makes no direct numerical comparison with that figure because the protocols differ.

Metadata discrepancies retained for audit:

- `R6_LOCAL_DELIVERY_CHECK.json` contains stale self-referential inventory metadata, while the scientific report-file hashes in the R6 seal match the local copies. This does not alter the R6 endpoint or its raw/table provenance.
- The R4 delivery index records an older digest for one X2 recovery log after a later append. The paper uses the sealed X2 summary table and does not select between the log digests.

## R8 corrections recorded for this manuscript

- Before R8-FIX, the delivered M1 E2 rows in `R8_MAIN_TABLE_V2.csv` were excluded because they computed the wrong contrast quantity; this entry is superseded by the F1 resolution below.
- Before R8-FIX, the delivered M3 `PRED_PAST` rows were excluded because the predicted-history injection did not alter the deployed history; this entry is superseded by the F2 resolution below.
- Valid R8 V2 rows (M1 E1/E3, M2 fresh cases and random-action reference, M3 real/repeated-frame contrasts, M4, M5, and M6) match the locally copied SHA-256-verified tables and are used as labelled.

## R8-FIX resolution

- F1 corrected the M1 E2 quantity. The corrected endpoint is the budget-50 real-history gain minus the budget-100 real-history gain, yielding +12.8 [9.9, 15.9] points at offset 25 and +10.0 [7.4, 12.8] at offset 50. The original rows are retained only as descriptive real-history success-rate differences (budget 50 minus budget 100: −2.3 and −2.6 points) in the supplement.
- The original PRED_PAST batch remains superseded because its second plan was identical to the real-history plan. The corrected batch passed the context-alignment diagnostic, but the strict pre-specified gate that perturbs only the two predicted past latent positions by 0.01 failed because the plan did not change. Its 1,200 rows are therefore labelled POSTHOC_AFTER_TECH_FAIL / INVALID_TECH_GATE and reported only as exploratory supplement evidence; no confirmatory PRED_PAST endpoint is selected.

## W3 source reconciliation

- The known-case Reacher three-stream single-frame refit contrast is taken from the R5 sealed replanning table as +3.7 [−0.8, +8.4] points. The earlier table cell used −1.0 as its lower bound; it is corrected in Table 2 and Figure 2b.
- The five-step refit reduction depends on the queried context. The recorded three-frame evaluation gives PushT −1.3%, whereas the R5 single-frame query check gives a 4.2% reduction (conditional interval −33.5% to +38.5%). Both values are retained with their source labels; the deployment-query value should be used when discussing the planner's single-frame interface.

## W3 corrections and source reconciliations

- Table 2's known-case Reacher A–C row previously used a lower bound of −1.0. The R5 sealed value is +3.7 [−0.8,+8.4]; the source value is used throughout the revised table, text, and Figure 2b.
- The M5 first-plan comparison was previously described as a gain appearing at the first replanning call. The sealed analysis instead compares first-plan-only success from the first call (87.9% versus 40.9% over 1,068 runs); the final endpoint is −0.5 points. The prose now labels this post hoc and descriptive.
- The R5 query-condition table contains two valid PushT h=5 reductions: −1.3% for the recorded three-frame diagnostic and +4.2% for the deployed single-frame query (conditional interval −33.5% to +38.5%). The manuscript uses the deployment-query value when discussing the planner interface and retains the diagnostic value in parenthetical context; neither is substituted for the other.
- The W3 table restructuring removes mechanism and protocol-extension rows from Table 3 because it is restricted to real-history minus single-frame contrasts. Long-horizon refit–official contrasts remain in Table 2, with their evidence labels.
- The historical `C_NUM_R5_RECOMPUTED` ledger row is retained for provenance but marked `SUPERSEDED_W3`; the manuscript uses the R5 sealed three-stream interval +3.7 [−0.8,+8.4] rather than the earlier recomputation bundle.

## W4 corrections (2026-10-04)

- Reacher joint naming: the text and supplement crosswalk said "elbow" while Fig. 3c said "wrist". The DeepMind Control Suite `reacher.xml` names the joints `shoulder` (unlimited) and `wrist` (limited to +/-160 degrees), consistent with the joint_angle_1 range of -2.879 to 2.798 rad. The manuscript now uses shoulder and wrist.
- Fig. 5d rates (91.7/100.0/99.7) and contrasts were typed into `make_figs.py`; they are now computed from `M4_ALL_RAW_VALUES.csv` and read from `M4_SECONDARY_TABLE.csv`. The values are unchanged.
- "The refit does improve prediction on states that the planner visits" had no planner-visited-state source in the ledger; it is replaced by the supported statement that the refit improves prediction under the deployed single-frame query, with numbers in Section 5.
- Table 4 used the three-frame h=5 diagnostic for PushT (1.3%); under the W3 reconciliation rule the deployment-query value (4.2%) is used when discussing the planner interface.
- Tables 2 and 3 previously mixed design labels (primary, fresh, stress, operating-point) with evidence labels; they now use the Section 3.3 evidence levels. The primary refit checks are named in the Table 2 caption.
