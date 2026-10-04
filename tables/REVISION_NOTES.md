# Numbered revision notes

1. Rewrote Section 3.1 to state that every training-window frame is a real observation, causal next-frame prediction is used at every position, position 0 sees one frame, and deployment starts from one current frame before autoregressive rollout. Section 5 now identifies the single-frame query as an in-distribution position-0 prefix and reports the 8--13% refit and ±2% context-matched changes.
2. Audited the Reacher probe truth. Section 5 reports joint_angle_0 (first hinge/shoulder) R² 0.17 and joint_angle_1 (second hinge) R² 1.00, and explains that joint_angle_0 is an unwrapped cumulative qpos coordinate extending beyond ±π.
3. Added the Elsevier generative-AI declaration, the Methods code/plot-assistance statement, the Figure 1 AI note, and a formal Data availability draft with repository/DOI placeholders.
4. Rebuilt the bibliography with 46 checked entries, exact Lambert et al. L4DC/PMLR 120 citation, and non-printing support comments. Removed approximate original Figure 6 bars from Table 1 and replaced the table with the compact protocol table.
5. Removed workflow labels and the uncompleted long-horizon subsection from the manuscript. The completed doubled-distance/doubled-budget stress test and descriptive random-action reference are reported in Sections 4, 6.2, 8, and Table 1; source traceability is in the supplementary appendix.
6. Replaced the abstract with the revised ≤250-word version without a limitation list.
7. Consolidated evidence-level wording in Section 3 and retained only scope-changing limitations in the Results sections; other limitations are in Section 8.
8. Standardized official predictor, refit predictor (mean of three seeds), context-matched refit, single-frame query, three-frame query, real-history replanning, planner streams A--C, one-decimal percentage points, intervals, and minus signs.
9. Added the known-case failure-rate changes (20.7→4.7% and 17.0→2.6%), entry-episode success (65.8→94.1%), and paired flips (196 failure→success; 18 success→failure) in Section 6 and the evidence ledger.
10. Reframed the introduction around the prediction-to-decision gap, converted contributions to four numbered items linked to Figure 1, added end-of-paragraph distinctions in related work, and corrected punctuation.
11. Replaced Figure 1 with the supplied TikZ overview and regenerated Figures 2--5 using direct archived-CSV reads. Figure source files and SHA-256 digests are registered in `FIGURE_SOURCES.md` and `CLAIM_EVIDENCE.csv`.
12. Moved figures/tables near first citation, used the compact seven-column Table 1, rounded Table 3 to one decimal, and kept the figure captions aligned with the supplied recommendations.
13. Compiled `elsarticle` with the `review` option at 10 pt; the main PDF is 20 pages including figures and tables. Generated matplotlib PDFs use embedded Helvetica CID TrueType and labels/ticks at least 6.5 pt.

14. Updated the frontmatter and declarations with Ruiqi Wang, the supplied ORCID 0009-0005-4942-5821, UC San Diego affiliation, corresponding-author email, CRediT statement, competing-interest statement, Funding statement, and the supplied Elsevier AI declaration wording. Applied the same author metadata to the supplementary source.

## Latest revision pass (R7 and cross-check corrections)

15. Corrected the task ordering and intervals for refit-minus-official success in the Introduction and Section 4; corrected the known-case official Reacher history result to 79.3% to 95.3%.
16. Replaced the abstract with the supplied R7 version, including the 6--74 query ratio, PushT stress result, and near-ceiling Reacher stress result; changed “ordinary success” to “closed-loop success” and “629 episodes” to “629 episode runs”.
17. Added the doubled-distance/doubled-budget stress subsection, random-action lower-bound column and intervals in Table 1, and directional fresh-case flip counts; updated the discussion and scope files.
18. Removed the self-supervised background paragraph, fixed the related-work discussion to distinguish Tian et al.'s VP$^2$ benchmark, and verified BWM as arXiv:2607.29302.
19. Restored PushT velocity rows in Figure 3c and added the PushT single-frame velocity qualification; regenerated Figures 2--5 from local archived CSV copies.
20. Added source comments for the four future R8 insertion points, the numeric audit script and its report, and updated `CLAIM_EVIDENCE.csv` with R7 tables, seals, and SHA-256 values.

## A revision pass (current request)

A1. Table 3 now contains only real-history minus single-frame rows. The two long-horizon predictor contrasts were moved to Panel B of Table 2; both captions identify their distinct estimands.

A2. The official arXiv record for 2607.29302 was checked: it is *BWM: A Low-Cost High-Fidelity World Simulator for Robot Learning*. It supports closed-loop policy evaluation, risk anticipation, and policy ranking, but does not establish a planner-bottleneck diagnosis. The BWM citation was therefore removed from the planner-bottleneck sentence; Terver et al. remains as the supporting citation. arXiv 2609.36305 was confirmed to be a different Bilinear World Models paper.

A3. Single-author wording was applied throughout the manuscript and Figure 1 source note: “the author” is used, and authorial plural first-person references were removed.

A4. Contribution 2 now names the preregistered long-horizon stress test. The conclusion now states that the history gain shrinks to about one point near the ceiling and that random-action floors reveal partly easy cases.

A5. Figure 1, Methods, and the Elsevier declaration now identify Claude (Anthropic; claude-opus-5-5) and OpenAI Codex (GPT-6; OpenAI) consistently.

A6. Data availability now states open Zenodo availability with the DOI retained as an author placeholder.

A7. Replaced `highlights.txt` with five checked highlights; current lengths are 80, 60, 76, 66, and 77 characters, all within the 85-character limit.

A8. Recompiled the `elsarticle` review manuscript, reran the numeric traceability script, and updated `BUILD_LOG.md`. This earlier note was superseded by the R8 integration pass below; the valid R8 V2 modules are now incorporated and the two invalid modules remain excluded.

## R8 integration and final manuscript pass (items 1--13)

1. Renamed the manuscript and Section 4 to “Better Predictors, Not Better Plans: Auditing Planning Bottlenecks in a JEPA World Model”.
2. Replaced the abstract and introduction with the query--horizon--ranking audit narrative; the contribution list now includes fresh cases and the preregistered long-horizon stress test.
3. Expanded Table 2 to the valid refit--official contrasts from the original, fresh, stress, budget, and CEM-900 conditions, including the known-case three-stream single/history contrasts.
4. Expanded Table 3 to the history comparisons from the known, fresh, stress, M1, M3, and M5 conditions. M1 E2 and M3 PRED_PAST values are deliberately absent because their delivered implementations were invalid.
5. Integrated valid R8 M2 fresh-case contrasts and fresh random-action floors, M3 real/repeated-frame contrasts, M4 100-case PushT oracle rescoring, M5 protocol-extension history results, and M6 CEM-900 descriptive sensitivity.
6. Added the U1/U2 E1 and E3 operating-point results. The invalid E2 quantity is documented in the supplement and `DISCREPANCIES.md` pending an R8-FIX V3 delivery.
7. Added the R8 correction note for the PRED_PAST injection failure; no PRED_PAST number is used in the main text, tables, or figures.
8. Rebuilt Figures 2--5 from local archived CSV tables. Figure 2b now shows the valid predictor contrasts; Figure 4 includes mechanism and operating-point panels; Figure 5d uses the 100-case M4 oracle extension, including the original 20 cases.
9. Updated Table 1 with original and fresh random-action lower bounds and the Cube interpretation warning.
10. Added implementation provenance for the single-frame official query and training history sizes, and added local R8 sources and SHA-256 digests to the evidence archive and claim ledger.
11. Added the R8 corrections and module details to the supplementary appendix and retained the required R4 alternative-explanation and unidentified-scope limitation record.
12. Updated the title, AI-assisted writing statements, single-author wording, Data availability placeholder, references (including the verified Bilinear World Models citation), highlights, and cover-letter draft.
13. Recompiled the `elsarticle` review manuscript and supplementary material, reran the numeric audit, and recorded any remaining validation limits in `BUILD_LOG.md`; no training, inference, simulation, or closed-loop run was added.

## Superseding final pass (latest consolidated request)

1. Reorganized the paper around query mismatch: Section 4 establishes the predictor-control discrepancy, Sections 5--6 locate context and horizon failures and test zero-training history repair, and Sections 7--8 separate scoring, search, randomness, and benchmark difficulty.
2. Replaced the abstract with the requested sub-250-word version and kept the title “Better Predictors, Not Better Plans: Auditing Planning Bottlenecks in a JEPA World Model”.
3. Added the five-paragraph introduction logic, the four contribution bullets, the direct Tian et al. VP$^2$ comparison, the verified Bilinear World Models citation (arXiv:2609.36305), and the removal of BWM arXiv:2607.29302 from the planner-bottleneck claim.
4. Table 2 now contains the complete valid refit--official contrast set with explicit evidence level and source columns. Table 3 reports the history interventions and controls with evidence level and source columns. M1 E2 and M3 PRED_PAST remain absent from both tables.
5. Integrated valid R8 V2 M1 E1/E3, M2, M3 real/repeated-frame, M4, M5, and M6 results, including fresh-case contrasts, random-action floors, 100-case oracle rescoring, protocol extension, and CEM-900 sensitivity. Invalid E2 and PRED_PAST values are recorded only as `INVALID` in the ledger and discrepancy note.
6. Updated the single-frame training-distribution explanation, unwrapped shoulder-angle explanation, PushT velocity control, budget/target operating points, first-plan descriptive comparison, stream sensitivity, random floors, and conclusion boundary statements.
7. Regenerated Figures 2--5 from archived tables, restored PushT velocity rows in Figure 3c, updated Figure 4 operating-point/mechanism panels, and updated Figure 5d to 100 cases including the original 20. Figure source paths and digests are current in `figures/FIGURE_SOURCES.md`.
8. Synchronized all three AI-use statements with Claude (Anthropic; claude-opus-5-5) and OpenAI Codex (GPT-6; OpenAI), changed plural author references to singular, and retained the Zenodo DOI placeholder.
9. Updated the five highlights, cover-letter draft, claim ledger, discrepancy record, and numeric audit. The final audit matches 387/391 extracted claim numbers; only affiliation/layout metadata remains unmatched.
10. Recompiled the current multi-file source with the `elsarticle` review option. The verified local output is 24 pages including bibliography (19 pages through the manuscript/declarations before references) and 3 supplementary pages. The built-in standalone editor compiler was also attempted and cannot resolve the project’s `sections/` inputs; this limitation is recorded in `BUILD_LOG.md`.

## Latest consolidated request: numbered compliance record

1. **Main line.** Defined query mismatch in the Introduction and organized the manuscript as phenomenon (Section 4), diagnosis/repair (Sections 5--6), and scoring/search generalization (Sections 7--8).
2. **Title.** Applied the requested title and changed all “same plans” wording to “better plans” or an equivalent neutral statement.
3. **Abstract.** Installed the supplied 233-word abstract; no invalid R8 E2 or PRED_PAST number is included.
4. **Introduction and contributions.** Added the LeWM TwoRoom probing/planning question, five-paragraph logic, fresh-case and long-horizon evidence, and four contributions linked to Figure 1 and the checklist.
5. **Related work.** Removed the self-supervised background paragraph, discussed Tian et al. VP$^2$ directly, checked BWM 2607.29302 and removed it from the bottleneck claim, and cited Pariente et al. 2609.36305 for the cross-planner comparison.
6. **Results.** Corrected task ordering, interval signs, query ratios, Reacher angle interpretation, stress/fresh/random-action values, R6 flips, M1 E1/E3, M2, M3 valid arms, M5, and M6; invalid E2/PRED_PAST values are excluded.
7. **Discussion.** Added the three-route interpretation, planner-stream and random-action floors, Cube caution, near-ceiling history boundary, five-item checklist, and concentrated limitations.
8. **Figures and tables.** Regenerated Figures 2--5 from archived files; Figure 3c includes PushT velocity, Figure 4 includes valid mechanism/operating-point rows, and Figure 5d covers 100 cases. Tables 1--3 use compact protocol/contrast layouts with evidence levels and sources.
9. **Language.** Standardized official predictor, refit predictor (mean of three seeds), context-matched refit, single-frame query, three-frame query, real-history replanning, planner streams A--C, percentage-point precision, and minus signs; singular author wording is used.
10. **Software facts.** Added the pinned LeWM and stable-worldmodel commit references and recorded the history-length fact check in the claim ledger.
11. **AI and data statements.** Synchronized Figure 1, Methods, and Elsevier declaration language with Claude (Anthropic; claude-opus-5-5) and OpenAI Codex (GPT-6; OpenAI); retained Zenodo DOI [DOI].
12. **Reproducibility files.** Updated `CLAIM_EVIDENCE.csv`, `DISCREPANCIES.md`, `BUILD_LOG.md`, `highlights.txt`, `cover_letter.md`, and the supplementary source-traceability appendix; no external release or submission was made.
13. **Build.** Local XeLaTeX/BibTeX with `elsarticle` review mode succeeds (25-page main PDF including references; 19 pages through declarations; 3-page supplement). The built-in standalone compiler cannot see the multi-file `sections/` inputs, as recorded in `BUILD_LOG.md`.

## R8-FIX correction pass (current delivery)

1. Integrated F1's corrected M1 E2 estimand. Section 6.4 and Figure 4c report budget-50 minus budget-100 real-history gains of +12.8 [9.9, 15.9] points at offset 25 and +10.0 [7.4, 12.8] at offset 50; both lower bounds are positive under the preregistered label. The original −2.3 and −2.6 point rows are retained only as descriptive real-history success-rate differences in the supplement. The abstract now states that the gain falls to about one point when doubled budget allows more replanning, and the corresponding contribution and discussion statements were updated.
2. Reclassified corrected PRED_PAST under F2. The strict preregistered two-past-position, +0.01 perturbation gate left the second plan unchanged, so the endpoint is not evaluated. The original injected batch remains numerically unused; the corrected 1,200-row batch is labelled POSTHOC_AFTER_TECH_FAIL / INVALID_TECH_GATE and reported only as exploratory supplementary evidence, with G1 alignment, 694/694 plan differences, gate details, exploratory estimates, and 176/22 directional flips documented.
3. Clarified that REPEAT3 uses copied current frames and zero actions, whereas REAL2, REAL3, and PRED_PAST pair past frames with actually executed action prefixes; the audit does not separate those contributions. The exact motion/action wording is synchronized across the abstract, Introduction, Section 6.3, checklist, discussion, and limitations.
4. Added local R8-FIX source copies, seal and validation digests, corrected claims, superseded discrepancy statuses, Figure 4c source registration, and regenerated figures. Local XeLaTeX/BibTeX succeeds in review mode; the updated numeric audit reports 438 tokens, 434 matched (four affiliation/layout metadata tokens unmatched). The built-in editor compiler was attempted and still cannot resolve the multi-file `sections/` inputs; the limitation is recorded in `BUILD_LOG.md`.

## W3 consolidated revision record

1. Corrected the known-case three-stream Reacher single-frame refit contrast to +3.7 [−0.8,+8.4] and synchronized Table 2, Figure 2b, the stream-specific analysis, and the claim ledger. The stream-specific values are −1.7 [−8.3,+5.7], +15.0 [+6.7,+24.0], and −2.3 [−10.0,+6.0] for streams A–C; stream B's official success is 68%.
2. Rewrote the descriptive first-plan analysis: real history from the first call succeeds on the first plan in 87.9% of 1,068 runs versus 40.9% for one frame, while final success differs by −0.5 points because real-history replanning recovers most single-frame failures.
3. Merged the history subsections into “When history matters”, reporting the four budget-by-offset history gains followed by the two corrected budget contrasts (+12.8 [9.9,15.9] and +10.0 [7.4,12.8]); the text attributes the contrast to replanning opportunity controlled by budget.
4. Corrected the Bilinear World Models moving-goal exclusion, repository URLs/titles, Figure 1 and Figure 2 captions, the singleton author wording, and all five static-context phrases. Added the four-column five-row checklist and the missing Section 8.3 lead paragraph.
5. Verified query-conditioned reductions from the sealed R5 table: PushT h=5 is 4.2% under the deployed single-frame query (1.3% under the recorded three-frame diagnostic); TwoRoom and Cube retain large h=5 single-frame reductions. The corresponding claim and discrepancy records distinguish both quantities.
6. Replaced the abstract with the 249-word corrected version, including the deployment-query PushT value, fresh-case history replication, budget boundary, and primary-comparison scope. Added the introduction motivation, Tian et al. positioning, contribution revisions, stream-randomness evidence, and TwoRoom/Cube interpretation.
7. Rebuilt Figures 2–5 from archived CSV/JSON values. Figure 2b uses filled primary and hollow extension markers with the fresh-history band; Figure 4c shows the budget-by-offset interaction with corrected contrasts; Figure 5d reports three pooled success rates and deltas for the 100-case simulator-rescoring extension. Tables 1–3 are compact, omit source columns, and use descriptive condition names.
8. Added the source/code crosswalk and R8 correction details to the Supplement, updated `CLAIM_EVIDENCE.csv` with all W3 estimates and current figure-script SHA, refreshed `DISCREPANCIES.md`, `highlights.txt`, and `BUILD_LOG.md`, and compiled both review-mode PDFs. No experiment, submission, publication, or release action was performed.

## W3 item-by-item compliance addendum

1. **Known-case Reacher interval.** Replaced the three-stream single-frame row with the R5 sealed +3.7 [−0.8,+8.4] value in Table 2, the stream analysis, Figure 2b, and the claim ledger.
2. **M5 first-plan interpretation.** Reworded the result as a post hoc descriptive first-plan-alone comparison (87.9% versus 40.9% over 1,068 runs), with the −0.5-point final contrast explained by later replanning.
3. **History section.** Merged the former history subsections into “When history matters”, placing the four history gains before the two corrected budget contrasts and attributing the contrast to replanning opportunity controlled by budget.
4. **Bilinear World Models and references.** Corrected the moving-goal exclusion wording; checked arXiv:2607.29302 as BWM and removed it from the planner-bottleneck claim; retained the supported Terver citation and corrected the stable-worldmodel/le-wm URLs and title casing.
5. **Figures and captions.** Updated Figure 1's evidence strip, Figure 2's interval wording, Figure 4c's budget–offset interaction, and the Figure 5d success-rate presentation; all plotted sources and script SHA are in the ledger.
6. **Discussion and conclusion.** Added the missing checklist lead, updated the fresh-case predictor wording, removed unsupported action-selection claims, and stated the budget/near-ceiling and random-action boundaries.
7. **Query and mechanism facts.** Standardized static-context parentheses, completed the evidence-grade paragraph, stated the TwoRoom one-frame equivalence, used the deployed PushT h=5 value, and retained the TwoRoom/Cube scoring/search limitation.
8. **Tables and traceability.** Table 1 is compact with random-action floors; Table 2 includes long-horizon predictor rows; Table 3 contains only real-history minus single-frame rows. The M5 first-plan claim and all W3 values are recorded in `CLAIM_EVIDENCE.csv` with source paths and SHA-256 digests.
9. **Build and delivery.** Main and supplementary sources compile locally in `elsarticle` review mode; the numeric audit was rerun. The built-in standalone compiler limitation, 27-page main output, 5-page supplement, and six non-scientific unmatched metadata/layout tokens are recorded in `BUILD_LOG.md`.

## W4 revision (Claude, applied directly to the sources, 2026-10-04)

W4-1. Abstract replaced (204 words). It now states the three outcomes of the diagnosis (Reacher: context; PushT: scored horizon; TwoRoom and Cube: the improvement reaches the query yet success is unchanged) and ends with the checklist sentence.
W4-2. Introduction: added the decision-aware/control-centric paragraph (Lambert et al.; Tian et al., VP2) that states what remains open; tightened the LeWM paragraph; compressed the design and results paragraphs; merged the duplicated "single image cannot reveal velocity" sentences; removed numbers from the contribution list.
W4-3. Related work: merged the two VP2 paragraphs; Bilinear World Models' moving-goal filter is mentioned once; the benchmark-list sentence was removed and the task sources are cited in Section 3.2 (DMC Reacher, OGBench Cube, PushT, TwoRoom). D4RL is no longer cited.
W4-4. Section 3 has three subsections; determinism and reference-condition text folded in; five-macro-step CEM horizon stated; the evidence-level paragraph names the fresh-case history replication; new label sec:evidence.
W4-5. Section 4: new opening sentence; the unsupported claim about "states that the planner visits" was replaced by a pointer to Section 5; stream B has its own paragraph ("would give an interval entirely above zero"); the text notes that original-case comparisons use stream A only.
W4-6. Section 5: 86.3% -> 7.8% is stated once; the context-matched control is one sentence with its baseline (relative to the refit); TwoRoom and Cube single-frame h=5 reductions (90.8%, 31.8%) are given; "elbow" -> "wrist" (DMC Reacher joint names are shoulder and wrist).
W4-7. Section 6: removed the failure-rate sentence (redundant with the success rates); compressed the frame/action confound sentence; rewrote the full-history paragraph without repeating -0.5.
W4-8. Section 8: 8.1 is a synthesis without repeated numbers (the stale first-plan sentence is gone); 8.2 has the new opening and the objective-mismatch/VP2 sentence; 8.3 says both main endpoints "were both replicated in preregistered tests on fresh cases"; Table 4 is a ragged-right tabularx table and uses PushT 4.2% (deployment query) instead of 1.3%; limitations compressed; the conclusion says the one-frame query "has 6-74 times the prediction error" instead of "is 6-74 times harder".
W4-9. Tables 2 and 3 use the Section 3.3 evidence vocabulary (prespecified, preregistered, post hoc, descriptive); known-case rows are post hoc.
W4-10. Figures 2-5 regenerated from the same sealed tables: Fig. 2 short labels (no overlap) and a band labelled "Reacher history gain, fresh cases" whose limits are read from the R6 endpoint; Fig. 4 in two rows, legend of panel b outside the axes, E2 contrasts read from the R8-FIX table and shown as a colored inset; Fig. 5d light/dark bars with labelled contrasts read from the M4 tables (hand-typed values removed); Fig. 3 wider spacing. The Linux build uses Liberation Sans because Helvetica is unavailable there.
W4-11. Reference title "stable-worldmodel"; Data availability names the GitHub URL and Zenodo DOI placeholders.
W4-12. Float parameters and emergencystretch added; Figures 4 and 5 may use float pages, so no float is deferred to the end.
W4-13. Data availability now cites the archived dataset (new reference wang2026data, "[dataset]", DOI placeholder) as required by the journal's data policy; a separate title page (title_page.docx) is provided in the submission package because the guide asks for one.
W4-14. Reference polish: MuJoCo is now an @inproceedings entry (fixes the missing space before its DOI) and proper nouns keep their capitals in sentence-case titles (MuJoCo, DeepMind Control Suite, OGBench, RL, TD-MPC2, Atari, Go).
W4-15. Inserted the GitHub URL (https://github.com/2GyounnnG/lewm-planning-audit) and the reserved Zenodo DOI (10.5281/zenodo.23134310) in Data availability and in the dataset reference; the reference now carries the DOI as a link.

## W5 repositioning after the second novelty check (Claude, 2026-10-04)

W5-1. A repeated novelty search found concurrent 2026 analyses of LeWM-style planners (You et al.; Singh; Li et al.; Gao et al. IMWM; Bai and Xiong; Wang et al.; Obst and Stolzenburg), the stable-worldmodel paper and World-in-World. Ten references were added after checking each abstract page; Wei et al. (objective-mismatch review) was added as well.
W5-2. Abstract and introduction now state that prediction error is already known to be a poor guide to planning success, with different proposed causes, and pose the open question as where a genuine improvement is lost inside one planner and whether the loss can be recovered without training. A new introduction paragraph states the synthesis: no single bottleneck explains the failure. Contributions 1, 2 and 4 were reworded (interventional design; random streams).
W5-3. Related work has a new paragraph on analyses of latent world-model planners, a review citation, World-in-World in the control-centric sentence, and Terver et al.'s two-frame velocity remark.
W5-4. Discussion 8.1 places each task's result next to the concurrent account (in-distribution information gap vs off-manifold queries; PushT scoring vs candidate-rescoring diagnostics; TwoRoom vs objective and search accounts); 8.2 and 8.3 cite the concurrent work; Limitations names a second model family as the most direct test of generality; the Conclusion states the task-specific locations.
W5-5. Factual correction: Sections 3.1 and 5 no longer state that TwoRoom used a one-frame training configuration or that its three-frame diagnostic equals the one-frame computation. The R5 code applies the same three-frame window to all tasks, and the TwoRoom single- and three-frame errors differ by about 1%; the text now reports only the measured ratio.


## W6 revision after the external review (Claude, 2026-10-04)

Sources: the external review (KBS readiness about 70%, novelty about 6/10), new post hoc analyses of sealed raw tables (`scripts/w6_posthoc.py`, `scripts/w6_family_sensitivity.py`, outputs in `evidence/w6/`), code inspection of le-wm 8edfeb3 and stable-worldmodel history, archived run identities, and an independent read-only check of the revised text by a separate agent. No trajectory was rerun and no model was trained.

W6-1. Two research questions. The introduction and abstract now ask (i) how much of a measured predictor improvement survives the deployed planner and (ii) whether a training-free query change improves control and lets that improvement through. "Each points to a single culprit" was removed.
W6-2. Predictor x history interaction (new, post hoc). Refit history gain minus official history gain: original Reacher -1.6 [-6.7, +3.7]; fresh history set -3.6 [-8.8, +1.4]; PushT -0.6 [-2.8, +1.7]. Reported in Section 6.2 and Table 3. With real history both predictors reach 95-98%, which limits the test; the text says the advantage "does not detectably grow" rather than claiming equivalence.
W6-3. Pooled single-frame refit contrast (new, post hoc). Per set: original +3.7, fresh history set +4.7 [+0.4, +9.1], fresh refit set +4.1; pooled over 300 cases +4.1 [+1.4, +6.8] (exact 4.148; an earlier draft rounded twice to 4.2). PushT pool over 200 cases +2.4 [-0.3, +5.1]. The abstract, introduction, Section 4, Table 2, Fig. 2b (open diamond) and conclusion now say that no prespecified or preregistered test detects a gain and that the pooled Reacher gain is about four points. Title kept by the author (option A); Section 4 states what the title summarizes.
W6-4. Narrowed claims: "the gain comes from motion" -> real observation-action history versus a static context; "the single-frame query lies within the training distribution" -> one-frame prediction is supervised in training, so the gap is not an unseen context length, and online off-manifold shifts are not tested; probes measure decodability, not non-identifiability; Section 7 retitled "Fixed-menu simulator rescoring improves PushT control"; "manufacture/fake" -> a stream-specific difference that other streams do not reproduce; "no effect/unchanged" -> "no detectable change"; the budget-100 negative refit contrast is stated in Section 8.2.
W6-5. Budget mechanism: the history gain tracks the budget rather than the offset, but a larger budget adds steps, calls and chances to succeed and single-frame success is 98-100% (offset 25) and 96-98% (offset 50) per predictor at budget 100; extra replanning is not identified as the mechanism. Offset doubling is a doubled start-goal time interval, not distance.
W6-6. Definitions and reproducibility (new Section 3.3 and Supplement): action alignment (frames t-10, t-5, t; two executed blocks prepended; same five future blocks and endpoint t+25); every later call uses real history (budget 100: calls at 25, 50, 75); fresh-set construction, reuse and sealing times; meaning of preregistered and post hoc; denominators (no missing cell in Tables 2-3); family-cluster sensitivity for PushT ([-3.0, +9.2], [-0.9, +6.5], [+4.0, +13.9]); D_total, J_task and both menu compositions in Section 7.
W6-7. Literature: ARC-Bench, AD-WM and Aim Short to Reach Far added after abstract-level checks; Terver et al. and TRM described more fully; the stable-worldmodel documentation is cited for its July 2026 history option (commit accd0a1).
W6-8. Software version correction: run identities and the R3 compatibility manifest record stable-worldmodel commit abdced49 (20 May 2026); the reference previously cited 63988116 (22 September 2026). The reference and Section 3.3 now give abdced49 and state that the library's history option postdates it and was not evaluated.
W6-9. Figures: Fig. 1 result chip and Fig. 2b (pooled row) updated; Figs. 3-5 unchanged.
W6-10. Interaction at the other operating points (post hoc, from the budget-100, offset-50 and stress raw tables): -0.3 [-4.9, +4.2] at budget 50, offset 50 (real-history success 94-95%); +1.1 [+0.4, +1.9] at budget 100, offset 25 (official predictor at 100%); +0.6 [-0.8, +1.8] at budget 100, offset 50; PushT stress -0.7 [-3.4, +2.1]. One sentence in Section 6.2; full table in the Supplement (`scripts/w6_interaction_operating_points.py`).


## W7: R10 history-completion results merged (Claude, 2026-10-04)

Source: R10 delivery (four preregistered modules, 9,000 new trajectories, no training), sealed tables copied to `evidence/r10/` with `SHA256SUMS.txt`.

W7-1. Library option (module A): under stable-worldmodel 63988116, `history_len=3` minus `history_len=1` for the official predictor on the Reacher history set is +12.0 [+7.7, +16.7] (84.7% -> 96.7%); single-frame success under that version is +4.3 [-0.7, +9.7] above the sealed value (descriptive). Section 3.3 now states what the version actually did at the first call (one frame, no action history), replacing the "padding" description and "not evaluated". Reported in the abstract, Sections 1, 6.1, 8.2, 9, Table 3 and Fig. 4a.
W7-2. TwoRoom and Cube history (modules B, C): -0.4 [-1.1, +0.1] and -0.2 [-1.0, +0.6] on the fresh refit sets; the preregistered prediction from the one-frame error ratio (Section 5) was met; 26% and 36% of runs reach the second call. Reported in the abstract, Sections 1, 5, 6.1, 8, 9, Table 3, Table 4 and Figs. 1 and 4a.
W7-3. Frame-action separation (module D): real frames with null actions -0.2 [-2.0, +1.6]; current-frame copies with real executed actions +11.6 [+8.5, +14.8]. Section 6.3 retitled "What the extra context supplies"; both arms are stated to be unpaired, off-distribution inputs that rank the components without isolating a mechanism. Limitations and Table 4 updated.
W7-4. Same-machine controls: cross-machine first-plan drift of about 1e-6 led to complete same-machine single-frame controls for B, C and D (preregistered); documented in Section 3.3 and the Supplement.
W7-5. Abstract rewritten to 210 words so that it and the keywords stay on page 1; highlights 3 and 4 replaced (12-16 points; actions carry the gain); cover letter contribution 2 updated; Fig. 4 enlarged to a 5.3-inch height with five new forest rows; Fig. 1 result chip updated.
