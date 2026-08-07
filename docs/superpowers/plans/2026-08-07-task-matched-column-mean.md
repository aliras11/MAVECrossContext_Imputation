# Task-Matched Column Mean Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Use ephemeral test files outside the repository because this publication repository must not contain unit tests.

**Goal:** Score Column Mean on the exact held-out values used by W, B1, B0, and no-added-source-missingness B1; include it in every corresponding comparison family and paper-facing winner calculation.

**Architecture:** Existing Column Mean predictions remain target-only. Regular predictions are re-scored using masks and source availability; no-double predictions are generated with the existing imputer inside each target-specific split tree. The statistics package consumes explicit task-matched loss rows, forms one all-method family per task and rate, and figures keep Column Mean as a dashed line that is eligible to win and receive a star.

**Tech Stack:** Python 3.10+, pandas, NumPy, SciPy, matplotlib, Slurm shell wrappers.

## Global Constraints

- Work only in `MAVE-Imputation-Pipeline`, whose remote is `git@github.com:aliras11/MAVECrossContext_Imputation.git`.
- Do not add unit-test files, test fixtures, development-only data, DOCX/PDF generation code, or generated scientific outputs to this repository.
- Use the target mask (`mask[TGT_score] == 1`) as the source of truth for deliberate target holdout.
- Define B1 as target held out and source available in the training split; define B0 as target held out and source unavailable in the training split.
- Exclude naturally missing target truth and non-finite predictions from scoring. Preserve all ordered non-self source-to-target pairs.
- Pool RMSE through additive SSE and N; do not weight context pairs equally.
- Retain two-sided unpaired Mann-Whitney U tests on 50 split-level pooled RMSE values.
- Bonferroni-correct all choose-two method comparisons within each dataset, task, and rate.
- Column Mean stays a dashed figure reference but is a full competitor and may receive the gold star.

---

### Task 1: Task-matched Column Mean loss production

**Files:**
- Modify: `imputation/measure_loss_on_splits_colmean.py`
- Modify: `imputation/colmean_imputer_runonsplits.py` only if required for target-specific invocation
- Create: `imputation/loss_measure_colmean_no_double_missing.py`
- Create: `cluster/submit_nodouble_colmean.sh`
- Modify: `cluster/submit_nodouble_all.sh`
- Modify: `cluster/submit_nodouble_measure_losses.sh`
- Modify: `cluster/submit_measure_losses.sh`

**Produces:**
- `$RUN_ROOT/splits_results_0506/column_mean_task_losses_regular.csv`
- `$RUN_ROOT/splits_results_0506/column_mean_task_losses_no_double.csv`

Both files use columns `dataset,model,rate,split,src,tgt,shift_type,loss_type,rmse,n_points,sse,prediction_file,train_file,mask_file`. Paths are relative to the run root. Regular output contains W, B1, and B0; no-double output contains B1 only.

- [ ] Create an ephemeral controlled fixture and a failing behavioral test proving the current W-only loss script cannot emit B1/B0 rows or distinguish mask holdouts from natural missingness.
- [ ] Implement one-to-one `hgvs_pro` alignment and task classification from mask plus training data.
- [ ] Compute N, SSE, and RMSE from finite truth/prediction pairs and reject zero-point rows or duplicate logical keys.
- [ ] Generate regular W/B1/B0 rows for eight W maps and 56 ordered between-map pairs.
- [ ] Run the existing Column Mean imputer inside each no-double `tgt_TGT` tree and generate no-double B1 rows.
- [ ] Add the no-double Column Mean Slurm array to the dependency graph and both loss stages to their established wrappers.
- [ ] Verify the ephemeral test fails before the change and passes after it; do not commit the test file.
- [ ] Commit the production changes.

### Task 2: Unified statistics and gold-star behavior

**Files:**
- Modify: `statistics/code/mave_statistics/constants.py`
- Modify: `statistics/code/mave_statistics/loaders.py`
- Modify: `statistics/code/mave_statistics/pipeline.py`
- Modify: `statistics/code/mave_statistics/inference.py`
- Modify: `statistics/code/mave_statistics/reporting.py`
- Modify: `statistics/code/figures/figure_helpers.py`
- Modify: the Figure 2-5 bar-chart modules and downstream active ranking/winner modules as required

**Interfaces:** `mave-stats` keeps its existing CLI. The two task-loss files from Task 1 become required inputs. `vs_colmean_*.csv` views inherit p-values from the unified all-method pairwise families rather than running separate tests.

- [ ] Create ephemeral failing tests showing Column Mean is absent from current W/B1/B0 pairwise families and the star helper accepts a tested set smaller than the displayed set.
- [ ] Load the new task-loss schemas and include Column Mean in W, regular B1, regular B0, and no-double B1 summaries.
- [ ] Build unified headline families of 10 W tests, 45 B1 tests, 10 B0 tests, 45 no-double tests at the first four rates, and 21 no-double tests at 0.1% saturation.
- [ ] Include Column Mean in context-level families and derive `vs_colmean` views from the corresponding unified rows.
- [ ] Make figure generation validate exact equality between displayed and tested method sets.
- [ ] Preserve the dashed line; choose the lowest mean across bars plus Column Mean and place the star on the line if it wins all adjusted comparisons.
- [ ] Ensure Figures 7 and 8 and every active best/ranking consumer consider the corrected Column Mean rows when their scope claims all evaluated methods.
- [ ] Verify the ephemeral tests fail before implementation and pass after it; do not commit test files.
- [ ] Commit the production changes.

### Task 3: Runtime documentation and production validation

**Files:**
- Modify: `README.md`

- [ ] Document the two task-loss files, their schemas, event definitions, target-only nature, fallback mean rule, and no-double invocation.
- [ ] Document the unified comparison families and exact expected counts.
- [ ] Make clear that masks identify deliberate target holdout, source availability comes from the training split, and pair-level rows are pooled by SSE and N.
- [ ] Run syntax/help/import checks and controlled end-to-end statistics/figure generation against a temporary synthetic run tree.
- [ ] Confirm no secrets, credentials, absolute local paths, generated outputs, or test files are tracked.
- [ ] Commit the documentation changes.

### Task 4: Cluster run and paper-facing outputs

- [ ] Merge the reviewed branch to `main`, push to the configured GitHub remote, and pull that exact commit on Pitt.
- [ ] Reuse regular Column Mean predictions, run the regular task-loss scorer, generate no-double Column Mean predictions, and run the no-double scorer.
- [ ] Run `mave-stats`, verify expected row/family counts, and download the two raw task-loss files plus regenerated statistics.
- [ ] Regenerate all active figures from the corrected results and visually inspect Figures 2-5, 7, and 8.
- [ ] Regenerate main Tables 1-2 and supplemental Tables S1-S3 from the corrected summaries; keep S4-S6 descriptive.
- [ ] Produce a Markdown discrepancy/provenance report and review changed winners, stars, p-values, and manuscript claims before surgical document edits.
