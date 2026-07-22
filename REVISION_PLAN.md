# PROAKTIV: reviewer-driven project and manuscript revision plan

## Purpose
Turn PROAKTIV from a promising proof of concept into a reproducible, mutation-aware kinase bioactivity study whose claims are no broader than its evidence. Address every point from Reviewers 1 and 2 through new validation, corrected analyses, transparent data curation, reproducible code, revised figures, and a point-by-point response.

Source reports: [`REVIEWER_COMMENTS.md`](REVIEWER_COMMENTS.md).

## Non-negotiable principles
- Preserve `master`; all work occurs on a dedicated revision branch.
- Do not alter or overwrite raw ChEMBL/UniProt source files.
- Freeze provenance, software versions, seeds, split manifests, and outputs.
- Treat pIC50 prediction as the primary task. Call it selectivity prediction only if a valid matched-compound differential-selectivity analysis can be supported.
- Treat NSCLC as a biological case study, not clinical validation.
- Report RMSE/MSE/MAE as primary performance metrics; correlation is supplementary.
- Do not claim uncertainty tracks error unless quantitative tests support it.
- Do not claim predicted resistance patterns reproduce experiments unless actual held-out experimental values are plotted alongside predictions.
- Separate computational validation, biological concordance, and clinical utility. They are not interchangeable.

## Initial reconnaissance findings
1. The manuscript relies on a 70/10/20 random row split and foregrounds PCC, so leakage through similar ligands, recurrent variants, and repeated assays is plausible.
2. Figures 2C and 3C are described as error-versus-uncertainty plots, but the manuscript states that uncertainty tracks error without reporting a coefficient, calibration metric, confidence interval, or high-error detection performance.
3. Figure 4 contains model predictions only, yet its legend and main text make experimental/clinical concordance claims.
4. The manuscript calls 25,412 entries “high-quality” without reporting extraction accuracy, mutation-normalization accuracy, unit checks, replicate variability, censoring, or an independently reviewed gold set.
5. `dataset.py` filters ChEMBL `potential_duplicate` records but does not establish a defensible replicate/assay aggregation policy. Mutation extraction is heavily EGFR-specific, despite the multi-kinase claim.
6. The training scripts assume random splitting. One script uses hard-coded HPC paths; the README advertises commands/configuration files that are not present in the repository.
7. The ESM2 pipeline truncates sequences at 1,024 tokens. Full-length ALK is longer, and clinically important ALK substitutions around residues 1,196–1,269 may therefore be absent from the encoded sequence. This must be resolved before making ALK claims.
8. The ESM2 training code contains duplicated early-stopping/checkpoint logic and labels an RMSE column as “MSE”; these require code-level verification.
9. The central defensible novelty is likely the mutation-aware curation and benchmarking framework, not a new neural architecture. The manuscript currently does not state that distinction clearly.

## Phase 0 — approval, preservation, and computational feasibility
### Actions after approval
1. Verify the local clone of `HaroldMate1/proaktiv` at `/opt/data/projects/proaktiv`, including remote, default branch, commit, tags, and working tree.
2. Create a branch such as `revision/reviewer-validation`; do not modify `master` directly.
3. Archive the submitted PDF, reviewer comments, current figures, data checksums, environment files, and current repository commit as the immutable baseline.
4. Inventory available CPU, RAM, disk, GPU/VRAM, existing checkpoints, logs, and intermediate artifacts.
5. Decide whether the 3B ESM2 model can be retrained locally, must use the original HPC system, or should be evaluated from an existing checkpoint. No new 3B result will be claimed without a reproducible execution record.
### Deliverable
- Baseline manifest and feasibility memo.
### Approval gate
- Confirm compute route and whether the intended venue still requires an Application Note/Short Report format.

## Phase 1 — repository audit and reproducibility repair
### Actions
1. Run a complete code/data inventory and map each manuscript result to its generating script, input file, checkpoint, and figure artifact.
2. Reconcile README commands with actual files. Either implement the documented CLIs/configs or rewrite README usage to match reality.
3. Replace hard-coded paths with configuration and command-line arguments.
4. Establish a reproducible structure, provisionally:
   - `data/raw/` immutable source exports
   - `data/interim/` parsed records
   - `data/processed/` validated modeling tables
   - `configs/` data, split, model, and experiment definitions
   - `src/proaktiv/curation`, `splits`, `models`, `evaluation`, `figures`
   - `tests/` actual unit/integration tests
   - `results/<experiment_id>/` metrics, predictions, split manifests, logs
5. Pin dependencies and record Python, CUDA, ChEMBL, UniProt, RDKit, DeepPurpose, PyTorch, Transformers, and ESM versions.
6. Add lightweight CI for parser/unit/split tests; GPU training remains a separately documented workflow.
### Tests to add
- Unit and relation conversion for IC50 records.
- Mutation parsing and canonicalization for EGFR, ALK, and BRAF.
- Reference-residue validation before applying substitutions.
- Insertions, deletions, compound mutations, and sequence-length changes.
- Duplicate/replicate aggregation.
- Scaffold and mutation group isolation across splits.
- Deterministic split manifests and seeds.
- Metric calculations and confidence intervals.
### Acceptance criteria
- A clean environment can reproduce the curated summary table and a small CPU smoke-test experiment from documented commands.
- README contains no nonexistent command or file.

## Phase 2 — validate and rebuild the curated dataset
### 2A. Provenance and eligibility
1. Freeze exact ChEMBL/UniProt releases and source identifiers.
2. Define inclusion/exclusion rules before examining model performance:
   - human EGFR, ALK, BRAF targets with verified identifiers;
   - IC50 only, with explicit unit handling;
   - accepted relation policy (`=`, `<`, `>`) and treatment of censored values;
   - valid structures and canonicalized compounds;
   - valid mutation labels and canonical sequence coordinates;
   - assay-format categories retained rather than silently pooled.
3. Preserve source row IDs, assay IDs, document IDs/DOIs, target IDs, molecule IDs, original values, transformations, and exclusion reasons.

### 2B. Mutation curation validation
1. Build a manually reviewed gold set stratified by kinase, wild type/mutant, mutation class, assay source, and common/rare variants.
2. Compare automated extraction and normalization against the gold set.
3. Report precision, recall, F1, exact-match accuracy, and error categories.
4. If possible, use a second annotator for a subset and report inter-rater agreement; otherwise document single-annotator adjudication honestly.
5. Expand ALK and BRAF parsing rather than routing their records through EGFR-centred regular expressions.
6. Reject or quarantine records where the stated reference residue disagrees with the canonical sequence.

### 2C. Assay heterogeneity and duplicates
1. Quantify records by biochemical/cell-based/other assay, BAO format, source document, year, unit, relation, and kinase.
2. Define biological duplicate keys explicitly, rather than relying only on ChEMBL’s `potential_duplicate` flag.
3. For repeated compound–variant measurements, report replicate counts and within-group dispersion in pIC50.
4. Compare reasonable aggregation rules, provisionally median pIC50 with replicate count and dispersion retained.
5. Run sensitivity analyses by assay class and exclude incompatible assay mixtures from the primary endpoint if necessary.

### 2D. Dataset reporting
Create a principal data table with, per kinase:
- raw and retained measurements;
- unique compounds and Bemis–Murcko scaffolds;
- unique variants, wild-type records, single and compound variants;
- assay-type counts;
- duplicate/replicate counts;
- pIC50 distribution and missingness;
- train/validation/test counts under each split.

### Acceptance criteria
- Every processed row is traceable to a source record and transformation log.
- Automated mutation curation has measured performance.
- No unit, relation, duplicate, or assay policy remains implicit.

## Phase 3 — define the scientific claim and novelty
### Literature/context analysis
1. Conduct a focused review of mutation-aware EGFR/TKI and broader drug–target affinity models, including D3EGFR, DeepPurpose and relevant deep/graph/structure-based DTA methods.
2. Build a comparison table covering task, targets, mutation representation, dataset, split strategy, metrics, external validation, uncertainty, public code, and intended use.
3. Distinguish:
   - contextual comparison: published performance on non-identical datasets;
   - fair head-to-head comparison: models rerun on PROAKTIV’s frozen data and identical splits.
4. Implement reproducible baselines on the same splits:
   - train-set mean/kinase mean;
   - simple ligand-only Morgan regression or tree model;
   - ligand plus explicit variant identity baseline;
   - Morgan + CNN–RNN benchmark;
   - Morgan + ESM2 model if compute permits.
5. If a published tool cannot be executed fairly, state that and compare design/validation rather than placing incomparable scores in one ranking.

### Proposed novelty statement
PROAKTIV should be positioned, if supported, as a reproducible mutation-aware curation and benchmarking framework spanning EGFR, ALK, and BRAF, with explicit evaluation of ligand/scaffold and unseen-variant generalization. It should not be presented as inventing a fundamentally new deep-learning architecture.

### Acceptance criteria
- Introduction reviews the relevant field.
- Novelty is stated in one precise paragraph and supported by a comparison table.

## Phase 4 — rigorous validation and leakage control
### Frozen evaluation schemes
Retain the random split only as a historical/optimistic baseline, then add:
1. **Bemis–Murcko scaffold split**: no scaffold shared across train/validation/test.
2. **Leave-one-mutation-out or grouped unseen-variant evaluation**: held-out variant identities do not appear in training.
3. **Combined scaffold + variant grouped split**, if sample sizes permit, as the hardest internal test.
4. **Temporal/external validation**: train on the frozen historical release and test on eligible records added in a later ChEMBL release or on a manually curated post-cutoff literature set.
5. Optional leave-one-kinase-out analysis as exploratory only; with three kinases it must not be oversold.

### Leakage audit
For every split, report overlap in:
- exact compounds;
- Bemis–Murcko scaffolds;
- canonical variant labels;
- protein sequences;
- assay/document IDs;
- duplicate groups.
A split fails acceptance if forbidden groups cross partitions.

### Fair training protocol
1. Use identical split manifests for all models.
2. Fit preprocessing and choose hyperparameters on training/validation only.
3. Keep the test set sealed until the pipeline is frozen.
4. Use repeated seeds where computationally feasible; report mean and confidence intervals, not the best single run.
5. Prevent Optuna or manual tuning from seeing test labels.

### Primary metrics
- RMSE and MSE in pIC50 units;
- MAE;
- R²;
- Pearson and Spearman as supplementary association metrics;
- bootstrap 95% confidence intervals;
- stratified results by kinase, assay type, common/rare variant, and split difficulty.

### Acceptance criteria
- Main conclusions survive at least scaffold and unseen-variant testing.
- If they do not, the manuscript is narrowed rather than cosmetically defended.

## Phase 5 — repair the protein representation and multi-kinase analysis
1. Verify the exact sequence segment seen by each encoder.
2. Resolve the 1,024-token truncation problem. The preferred comparison is a biologically defined kinase-domain sequence, with documented UniProt boundaries and adequate flanking context, versus the previous full-length/truncated representation.
3. Confirm that EGFR, ALK, and BRAF mutation positions are actually represented in every encoded input.
4. Produce per-kinase metrics under every split.
5. Add a substantive ALK or BRAF case study with held-out experimental observations, not prediction-only claims.
6. If ALK/BRAF sample size or curation quality remains inadequate, narrow the title and principal claims to EGFR and present ALK/BRAF as exploratory demonstrations.

### ESM2 rationale test
Compare CNN–RNN and ESM2 specifically on:
- unseen variants;
- low-frequency variants;
- compound mutations;
- scaffold-held-out compounds;
- each kinase separately.
The ESM2 model remains emphasized only if it adds value on the mutation-generalization task or provides a clearly justified complementary benefit.

## Phase 6 — decide and validate the selectivity claim
1. Define the proposed use case before analysis:
   - absolute pIC50 prediction;
   - within-variant inhibitor ranking;
   - mutation-induced resistance/sensitivity shift, ΔpIC50 relative to wild type;
   - cross-kinase selectivity, ΔpIC50 for the same compound across targets.
2. Audit whether sufficiently matched compounds and assay-compatible measurements exist for each task.
3. If data are sufficient, evaluate ranking/selectivity using held-out experimental pairs and report rank correlation, top-k retrieval, pairwise ordering accuracy, and ΔpIC50 error as appropriate.
4. Add a prediction-versus-measurement selectivity/resistance figure with uncertainty and residuals.
5. If matched data are insufficient, remove “selectivity prediction” claims and describe PROAKTIV strictly as variant-specific bioactivity prediction.

## Phase 7 — uncertainty analysis that can withstand review
1. Correct MC-dropout implementation and ensure stochastic dropout is active without introducing unintended training-state effects.
2. Increase and justify the number of stochastic passes; assess stability of estimated uncertainty.
3. Quantify, separately for validation and sealed test sets:
   - Pearson and Spearman association between uncertainty and absolute error;
   - confidence intervals and p-values;
   - high-error detection AUROC/AUPRC at a predeclared error threshold;
   - risk–coverage curve and area under the risk–coverage curve;
   - empirical interval coverage/calibration after calibration on validation data only.
4. Compare MC dropout against at least a simple baseline, such as ensemble variance or distance-to-training-domain, if computationally feasible.
5. Rewrite Figures 2C/3C and text according to evidence:
   - retain “tracked error” only if supported quantitatively;
   - otherwise state that MC-dropout uncertainty was weakly associated or uninformative and present it as preliminary.

## Phase 8 — rebuild figures and tables
### Proposed main items
- **Figure 1:** study design and curation/validation flow, including leakage-safe splits.
- **Table 1:** dataset composition by kinase, variant, ligand, scaffold, assay and replicate status.
- **Table 2:** comparison with prior mutation-aware and DTA methods.
- **Figure 2:** model performance across random, scaffold, unseen-variant and temporal/external splits using RMSE/MAE with confidence intervals.
- **Figure 3:** per-kinase and variant-frequency-stratified performance; explicit ESM2 versus CNN–RNN comparison.
- **Figure 4:** measured versus predicted drug–variant sensitivity/resistance for a genuinely held-out EGFR case study, plus ALK or BRAF if defensible.
- **Figure 5 or supplement:** uncertainty calibration, uncertainty–error association and risk–coverage.
- **Supplementary tables:** parser gold-set results, assay heterogeneity, replicate dispersion, split overlap audit, full hyperparameters and per-seed results.

### Figure rules
- Experimental points and predictions use unmistakably different symbols/layers.
- Every performance panel states sample size, split, metric units and confidence interval.
- No heatmap of predictions is described as experimental validation.
- Correlation never substitutes for error magnitude.

## Phase 9 — rewrite the manuscript
### Title and abstract
1. Narrow title and conclusions to demonstrated capabilities.
2. State dataset validation, hard splits, primary RMSE/MAE, external validation and limitations.
3. Remove unsupported clinical-decision language and any claim that predictions “consistently aligned” without held-out measurements.

### Introduction
1. Add a concise review of existing kinase/DTA and EGFR mutation-specific tools.
2. Identify the precise unresolved gap: reproducible mutation-aware curation plus hard-split evaluation across selected kinases.
3. Explain why EGFR, ALK and BRAF are biologically informative case studies without implying patient-level validation.

### Methods
Expand data provenance, assay eligibility, relation/unit handling, mutation gold-set validation, replicate aggregation, sequence windows, split algorithms, leakage checks, model selection, metrics, confidence intervals, uncertainty calibration, baselines and reproducibility.

### Results
Follow the evidence in this order:
1. validated dataset;
2. leakage audit;
3. baseline and model performance under increasingly difficult splits;
4. stratified kinase results;
5. ESM2 added-value test;
6. held-out biological case study;
7. uncertainty results.

### Discussion
1. Lead with what survives hard validation.
2. Compare fairly with prior work.
3. Discuss assay noise, limited kinases, sparse variants, sequence-only modeling, ChEMBL bias, model applicability domain and lack of clinical validation.
4. Reframe future clinical use as hypothesis generation requiring experimental and prospective validation.

### Editorial cleanup
- Correct Pearson capitalization and percentage/scalar reporting.
- Remove duplicated declarations/author-contribution text.
- Ensure figure references and captions match the actual panels.
- Correct citation numbering and add omitted methodological references.
- Use consistent pIC50 units and terminology.

## Phase 10 — point-by-point response to reviewers
Create a response matrix with one row per reviewer comment:
- reviewer quotation;
- response and whether we agree;
- analysis/code/data change;
- numerical evidence;
- manuscript section, page and line numbers;
- figure/table/supplement reference;
- repository path/commit reference.

The response will acknowledge valid limitations directly. Where the evidence does not support the former claim, the claim will be withdrawn rather than argued into submission.

## Phase 11 — final verification and release
1. Run unit/integration tests and CPU smoke test.
2. Reproduce every table and figure from a clean environment or documented container.
3. Validate that no test data informed tuning.
4. Cross-check all manuscript numbers against machine-readable metrics files.
5. Check citations against source papers and verify every DOI/title.
6. Have a second scientific reviewer inspect the mutation gold set and biological case-study interpretations if available.
7. Build final manuscript PDF, supplement, response letter, data dictionary and repository release.
8. Open a pull request for review; merge only after your approval.

## Reviewer-to-workstream map
| Reviewer concern | Primary response |
|---|---|
| Novelty and existing models | Phase 3 literature/comparison table and fair baselines |
| NSCLC context unsupported | Narrow claims; use NSCLC as case study only |
| No selectivity use case | Phase 6 matched-data selectivity gate or remove claim |
| Correlation instead of RMSE/MSE | Phase 4 primary error metrics and confidence intervals |
| Random split inflation | Scaffold, unseen-variant, combined and temporal/external splits |
| Unvalidated curation | Phase 2 gold set, provenance, parser metrics and replicate audit |
| Figures 2C/3C mismatch | Phase 7 quantitative uncertainty evaluation and rewritten claims |
| Figure 4 predictions-only | Held-out measured-versus-predicted case study or relabel as exploratory |
| ALK/BRAF underdeveloped | Per-kinase analysis and added case study, or narrower framing |
| Assay heterogeneity | Assay-stratified audit, aggregation policy and sensitivity analyses |
| ESM2 rationale unclear | Direct unseen-variant/rare-variant comparison with CNN–RNN |
| Missing dataset summary | Main per-kinase dataset table |

## Recommended execution order
1. Preserve baseline and inspect compute.
2. Repair curation and validate the dataset.
3. Freeze split manifests and leakage checks.
4. Run lightweight baselines and CNN–RNN experiments.
5. Decide whether ESM2 3B retraining is feasible and warranted.
6. Run hard-split, stratified, selectivity and uncertainty analyses.
7. Freeze results.
8. Rebuild figures/tables.
9. Rewrite manuscript and supplement.
10. Draft reviewer response.
11. Reproduce cleanly, open PR, and obtain final author approval.

## Decisions requested before implementation
1. Confirm whether the target is resubmission to the same venue or preparation for a different journal.
2. Confirm access to the original GPU/HPC environment and any trained checkpoints/logs not in GitHub.
3. Confirm whether a co-author can independently annotate a subset of mutation records.
4. Confirm whether you prefer the strongest full revision above or a narrower EGFR-first manuscript if ALK/BRAF validation proves too weak.

## Current status
The reviewer comments and this revision plan have been added as documentation. No data, model, figure, analysis or manuscript revision has begun.
