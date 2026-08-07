# MAVE Imputation Pipeline

This runtime-only repository reproduces the MTHFR missing-data workflow: split and mask generation, imputation, loss measurement, variance decomposition, manuscript statistics, the focused Trp165 analysis, figure generation, and Pitt Slurm submission. It contains the executable Python, R, and shell code; one authoritative full-data CSV; archival source-score CSVs; required runtime metadata and resources; packaging metadata for the statistics commands; the license; and this guide.


## Repository layout

| Path | Contents |
| --- | --- |
| `data/` | The authoritative MTHFR full-data CSV and archival source-score exports. |
| `imputation/` | Split generators, model runners, loss scripts, shared model/preprocessing code, `runtime_paths.py`, and the bundled BLOSUM100 matrix. |
| `imputation/decomposition/` | Prediction-layout metadata, event definitions, sufficient-statistics builder, summaries, validation, and local/Slurm orchestration. |
| `statistics/` | Installable statistics, Trp165, and figure commands plus their Python implementation. |
| `cluster/` | Pitt Slurm wrappers for the regular and no-double-missing pipelines. |

The checkout must be an immediate child of the run root. The imputation loss scripts derive this layout from their own location:

```text
RUN_ROOT/
├── MAVE-Imputation-Pipeline/       # this checkout
├── data_splits/                    # regular splits and predictions
├── data_splits_no_double_missing/  # no-double splits and predictions
├── splits_results_0506/            # all 13 statistics input CSVs
└── full_data/                       # staged copy for decomposition/Trp165
```

The established Pitt values are:

```bash
export RUN_ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
export REPO="$RUN_ROOT/MAVE-Imputation-Pipeline"
```

On another system, set `RUN_ROOT` and `REPO` to the analogous absolute paths for direct commands. The bundled `cluster/*.sh` wrappers retain the Pitt path above, the `dept_cpu` partition, and the `struct` environment; adapt those wrapper constants and Slurm resources before using them elsewhere.

## Software setup

Use Python 3.10 or newer. The imputation and statistics code uses PyTorch, NumPy, pandas, SciPy, scikit-learn, Biopython, matplotlib, and seaborn. Install versions suitable for the target CPU/GPU environment. From the repository root, install the packaged statistics commands with:

```bash
cd "$REPO"
python -m pip install -e ./statistics
```

This installs only `mave-stats`, `mave-trp165`, and `mave-figures` and their declared statistics dependencies. The flat scripts under `imputation/` are not a Python package and still require their scientific and machine-learning dependencies, including PyTorch, scikit-learn, and Biopython, in the active environment.

The R runners require R plus `optparse`, `data.table`, `mice`, `lme4`, `miceRanger`, `ranger`, `R6`, and `class`. Some runners attempt to install missing packages at run time; that requires network and library write access. Preinstalling all packages in the cluster environment is more reliable.

`old_base` (historical local work) and `struct` (Pitt wrappers) record the environments originally used; this repository does not create either environment. A complete 50-split run is computationally intensive and was run through Pitt Slurm.

## Authoritative input and terminology

`data/mthfr_crossAllcontext_domainannotation.csv` is the sole authoritative runtime full-data CSV. It is 4,069,375 bytes, with 13,192 data rows and 24 columns. Its SHA-256 is:

```text
ed503e4cba97c46cfa5c1be7f2084d9f6ab535bbfe7db101faff146b75a81e4a
```

`data/source_scores/` contains 51 archival CSVs supplied with the MTHFR score
processing workspace: 24 wild-type-background exports, 24 A222V-background
exports, and three combined or intermediate tables. These files are retained
for data provenance and are not read by the split, imputation, decomposition,
or statistics commands. In particular,
`data/source_scores/mthfr_crossAllcontext_domainannotation_new.csv` is a
byte-for-byte copy of the authoritative CSV above. The runtime pipeline
continues to use only `data/mthfr_crossAllcontext_domainannotation.csv`.

The source-data copy deliberately excludes duplicate ZIP archives, notebooks,
helper scripts, and operating-system metadata. See `data/source_scores/README.md`
for the inventory and provenance boundary.

The exact header order is:

```text
hgvs_pro,
wt12_score,wt12_se,wt25_score,wt25_se,wt100_score,wt100_se,wt200_score,wt200_se,
av25_score,av25_se,av100_score,av100_se,av200_score,av200_se,av12_score,av12_se,
aa_pos,alt_aminoacid,wt_aminoacid,pos_aminoacid,str_aa_wt,str_aa_mut,domain
```

The eight assay contexts are `av12`, `av25`, `av100`, `av200`, `wt12`, `wt25`, `wt100`, and `wt200`; each has a score and reported standard error (SE). The eight identity/annotation columns are exactly `hgvs_pro`, `aa_pos`, `alt_aminoacid`, `wt_aminoacid`, `pos_aminoacid`, `str_aa_wt`, `str_aa_mut`, and `domain`.

Terms used by the pipeline:

- **Natural missingness** is an absent value in the authoritative CSV. A **deliberately held-out value** was observed in that CSV and was replaced with missing data by a split generator for evaluation.
- A **mask file** is the source of truth for deliberate holdout: `1` in a score or paired SE field means that field was deliberately held out. Naturally missing entries remain `0` and are not evaluation points. Identity/annotation fields are carried through the mask.
- **Missingness rate** is the fraction of originally observed values deliberately hidden; **saturation** is `100% - missingness rate`.
- A **split** is one seeded realization of deliberate missingness. The established design uses 50 splits per rate; split `S` uses NumPy seed `42 + (S - 1)`.
- **W** (within-map) predicts a held-out target from the same assay map.
- **B1** (source-observed between-map) predicts a held-out target where the source score is available in that split.
- **B0** (source-missing between-map) predicts a held-out target where the source score is unavailable. Decomposition further distinguishes deliberately injected from naturally missing source values.

The regular generator injects the requested fraction independently into every score column and its paired SE, while preserving natural missingness. It writes labels such as `10` for a rate supplied as `10`, `10%`, or `0.1`. The established regular wrappers use `10,20,40,60,80,90`.

The no-double generator injects missingness into one target context at a time; other potential source maps are unchanged but may still contain natural missingness. Its established wrapper supplies raw rates `10,40,80,99,99.9`, producing labels `10,40,80,99,999`. Label `999` therefore means 99.9% missingness and 0.1% saturation; model and loss runners consume the label `999`, while split generation must receive raw `99.9`.

## Run the pipeline

The stages below are ordered. All commands assume the `RUN_ROOT`/`REPO` layout defined above.

### 1. Stage the external full-data copy

Imputation and loss code reads the tracked CSV directly from `data/`. Decomposition and Trp165 preserve their historical `$RUN_ROOT/full_data/...` input contract, so stage an identical external copy before those stages:

```bash
mkdir -p "$RUN_ROOT/full_data"
cp -p "$REPO/data/mthfr_crossAllcontext_domainannotation.csv" \
  "$RUN_ROOT/full_data/mthfr_crossAllcontext_domainannotation.csv"
sha256sum "$RUN_ROOT/full_data/mthfr_crossAllcontext_domainannotation.csv"
```

The digest must be `ed503e4cba97c46cfa5c1be7f2084d9f6ab535bbfe7db101faff146b75a81e4a`. This is a runtime copy, not a second tracked source.

### 2. Generate regular splits and masks

Run from the repository root:

```bash
cd "$REPO"
python imputation/split_generator.py \
  --full-data data/mthfr_crossAllcontext_domainannotation.csv \
  --n-splits 50 \
  --rate 10 --rate 20 --rate 40 --rate 60 --rate 80 --rate 90 \
  --output-dir "$RUN_ROOT/data_splits" \
  --seed 42
```

### 3. Generate no-double-missing splits and masks

Run from the repository root. The raw `99.9` argument is intentionally converted to the `999` path label.

```bash
cd "$REPO"
python imputation/split_generator_nodouble_missing.py \
  --full-data data/mthfr_crossAllcontext_domainannotation.csv \
  --n-splits 50 \
  --rate 10 --rate 40 --rate 80 --rate 99 --rate 99.9 \
  --output-dir "$RUN_ROOT/data_splits_no_double_missing" \
  --seed 42
```

Use repeated `--target CONTEXT` options to generate only selected targets; the default is all eight.

### 4. Submit the Pitt dependency graphs

Both master scripts change into their own directory, but invoking them from the repository root is clearest:

```bash
cd "$REPO"
bash cluster/submit_all.sh
bash cluster/submit_nodouble_all.sh
```

`submit_all.sh` generates regular splits, runs eight model jobs, and then runs regular loss measurement. It evaluates both KNN similarity modes and the required loss stage explicitly measures direct KNN. `submit_nodouble_all.sh` generates all five no-double labels (`10,40,80,99,999`), runs five between-map model families as 40-task arrays (five rates × eight targets), and measures their losses in the same dependency graph.

Individual wrapper mapping:

| Wrapper | Stage |
| --- | --- |
| `cluster/submit_split_generator.sh` | Regular split/mask generation. |
| `cluster/submit_single_ae.sh`, `submit_dual_ae.sh` | Regular SingleAE and DualAE. |
| `cluster/submit_colmean.sh`, `submit_knn.sh`, `submit_pca.sh` | Regular Column Mean, KNN (`diff` and `direct`), and PCA. |
| `cluster/submit_mice.sh`, `submit_mice_rf.sh`, `submit_lmm.sh` | Regular MICE-PMM, MICE-RF, and the R linear/mixed models. |
| `cluster/submit_measure_losses.sh` | Eight required regular loss CSVs, including direct KNN. |
| `cluster/submit_measure_knn_losses.sh` | Optional standalone loss measurement for both KNN modes. |
| `cluster/submit_nodouble_split_generator.sh` | Five-rate no-double split/mask generation. |
| `cluster/submit_nodouble_single_ae.sh`, `submit_nodouble_dual_ae.sh` | No-double SingleAE and DualAE. |
| `cluster/submit_nodouble_mice.sh`, `submit_nodouble_mice_rf.sh`, `submit_nodouble_lmm.sh` | No-double MICE-PMM, MICE-RF, and R linear/mixed models. |
| `cluster/submit_nodouble_measure_losses.sh` | Five required no-double loss CSVs. |

### 5. Run individual models directly

The Python runners accept the options shown below. Run them from the repository root so local imports and output paths are unambiguous:

```bash
cd "$REPO"

# One regular rate; each runner processes splits 1..50.
python imputation/single_ae_runon_splits.py \
  --base-dir "$RUN_ROOT/data_splits" --num-splits 50 --missing-rate 10
python imputation/dual_ae_runon_splits.py \
  --base-dir "$RUN_ROOT/data_splits" --num-splits 50 --missing-rate 10
python imputation/colmean_imputer_runonsplits.py \
  --base-dir "$RUN_ROOT/data_splits" --num-splits 50 --missing-rate 10
python imputation/knn_runon_splits.py \
  --base-dir "$RUN_ROOT/data_splits" --num-splits 50 --missing-rate 10 \
  --matrix BLOSUM100 --k 5 --pos-window 0 --sim_mode direct
python imputation/pca_runon_splits.py \
  --base-dir "$RUN_ROOT/data_splits" --num-splits 50 --missing-rate 10 \
  --n-components 1 2 4 10 20 22

# One no-double target/rate label.
python imputation/single_ae_runon_nodouble_splits.py \
  --base-dir "$RUN_ROOT/data_splits_no_double_missing" \
  --num-splits 50 --missing-rate 999 --target av12
python imputation/dual_ae_runon_nodouble_splits.py \
  --base-dir "$RUN_ROOT/data_splits_no_double_missing" \
  --num-splits 50 --missing-rate 999 --target av12
```

`knn_runon_splits.py` automatically uses `imputation/blosum100.iij` when `--matrix BLOSUM100` is selected and `--matrix-file` is omitted. `--sim_mode diff` ranks by the difference from the wild-type→mutant substitution score; `direct` ranks candidate amino acids directly against the mutant. The statistics pipeline requires the direct-mode loss file.

The R interfaces are positional for MICE and option-based for the linear/mixed runner:

```bash
cd "$REPO"

Rscript imputation/mice_runon_splits.R "$RUN_ROOT/data_splits" 10
Rscript imputation/mice_rf_run_onsplits.R "$RUN_ROOT/data_splits" 10
Rscript imputation/lmm_runon_splits.R \
  --splits_dir "$RUN_ROOT/data_splits" --rate 10 \
  --out_dir "$RUN_ROOT/data_splits"

Rscript imputation/mice_runon_nodouble_splits.R \
  "$RUN_ROOT/data_splits_no_double_missing" 999 av12
Rscript imputation/mice_rf_runon_nodouble_splits.R \
  "$RUN_ROOT/data_splits_no_double_missing" 999 av12
Rscript imputation/lmm_runon_nodouble_splits.R \
  --splits_dir "$RUN_ROOT/data_splits_no_double_missing" \
  --rate 999 --target av12
```

### 6. Measure and collate losses

The regular loss scripts have fixed six-rate scans; the no-double scripts default to all five no-double labels. They read the tracked full-data CSV and the sibling split directories through `imputation/runtime_paths.py`, then publish fixed filenames together in `$RUN_ROOT/splits_results_0506`. Run from the repository root:

```bash
cd "$REPO"

python imputation/singleae_loss_measure.py
python imputation/doubleae_loss_measure.py
python imputation/measure_loss_on_splits_mice.py
python imputation/measure_loss_on_splits_RFmice2.py
python imputation/measure_loss_on_splits_linearmodels.py
python imputation/measure_loss_on_knn_imputer.py --sim_mode direct
python imputation/measure_loss_on_splits_colmean.py
python imputation/pca_loss_measure.py

python imputation/loss_measure_singleae_no_double_missing.py
python imputation/loss_measure_dualae_no_double_missing.py
python imputation/loss_measure_mice_no_double_missing.py
python imputation/loss_measure_micerf_no_double_missing.py
python imputation/loss_measure_linearmodels_no_double_missing.py
```

The no-double loss scripts also support `--base-dir`, `--target` (one or more values), and `--rates` (one or more integer labels). There is no separate collator: successful execution of these scripts creates the 13 files expected by statistics in the common directory.

### 7. Run variance decomposition

Decomposition reads `$RUN_ROOT/full_data`, regular masks/splits, and the regular prediction layouts. The consolidated wrapper must be invoked through `imputation/decomposition/submit_decomposition.sh`; it internally runs the Python files as `decomposition` modules from `imputation/`.

The following selects the eight primary layouts: SingleAE, DualAE, MICE-PMM, MICE-RF, basic linear, Column Mean, PCA k=1, and direct KNN. `GENERATION_COMMIT` must be the exact commit whose code generated the SingleAE, DualAE, and PCA predictions. The current consolidated history accepts corrected root `09bf3c9fc7e150947015c8ed1c428483936d97e3` or a descendant (and recognizes the legacy corrected roots recorded under Provenance).

```bash
cd "$REPO"
export GENERATION_COMMIT="$(git rev-parse HEAD)"
export DECOMP_OUT="$RUN_ROOT/decomposition_outputs/mar6_primary"
export SPLITS_1_50="$(seq -s, 1 50)"

bash imputation/decomposition/submit_decomposition.sh \
  --run-root "$RUN_ROOT" \
  --output-root "$DECOMP_OUT" \
  --run-id mar6_primary \
  --models single_ae:default,dual_ae:default,mice_pmm:default,mice_rf:default,lmm:basic_linear,column_mean:default,pca:k1,knn:direct \
  --rates 10,20,40,60,80,90 \
  --splits "$SPLITS_1_50" \
  --generation-commit "$GENERATION_COMMIT" \
  --max-concurrent 64
```

Without `--local`, the wrapper submits a shard array, finalizer, and summary job with dependencies. Add `--local` to execute them synchronously. The default output root is `$RUN_ROOT/decomposition_outputs/RUN_ID`; `--output-root` makes the destination explicit. `--development-only` permits unapproved pre-fix output provenance but marks the artifacts non-scientific and is not appropriate for the primary run.

The wrapper automatically executes both summaries. To rerun them manually, work from `imputation/` and invoke modules, not direct file paths:

```bash
cd "$REPO/imputation"
python -m decomposition.summarize_decomposition \
  --stats "$DECOMP_OUT/decomposition_sufficient_stats.csv" \
  --validation "$DECOMP_OUT/decomposition_file_validation.csv" \
  --output-root "$DECOMP_OUT"
python -m decomposition.summarize_split_variability \
  --stats "$DECOMP_OUT/decomposition_sufficient_stats.csv" \
  --output-root "$DECOMP_OUT" --expected-splits 50
```

### 8. Generate manuscript statistics

Both input options point to the same co-located directory because it contains all eight regular and five no-double files. Use a dedicated output directory, not the raw-results directory or its parent:

```bash
cd "$REPO"
mave-stats \
  --results-dir "$RUN_ROOT/splits_results_0506" \
  --nodouble-results-dir "$RUN_ROOT/splits_results_0506" \
  --output-dir "$RUN_ROOT/statistics_outputs" \
  --expected-splits 50 \
  --minimum-completeness 0.95
```

`--results-dir`, `--nodouble-results-dir`, and `--output-dir` are required. `--expected-splits` and `--minimum-completeness` default to `50` and `0.95`.

Without an editable install, the equivalent is:

```bash
PYTHONPATH=statistics/code python -m cli.generate_statistics \
  --results-dir "$RUN_ROOT/splits_results_0506" \
  --nodouble-results-dir "$RUN_ROOT/splits_results_0506" \
  --output-dir "$RUN_ROOT/statistics_outputs" \
  --expected-splits 50 --minimum-completeness 0.95
```

For model `m`, rate `r`, and split `s`, headline RMSE pools eligible held-out cells across context rows before comparing models:

```text
RMSE(m,r,s) = sqrt(sum_i[n_i * RMSE_i^2] / sum_i[n_i])
```

Thus a context pair with more eligible values receives more weight; context pairs are not equally weighted. The runtime performs unpaired, two-sided Mann–Whitney U tests on split-level RMSE arrays. Headline and Column-Mean comparisons use Bonferroni correction within each dataset × loss type × rate family. Context tables report both Bonferroni correction within each context and across all valid comparisons at that rate. These are deliberately unpaired tests even when split identifiers overlap.

### 9. Run the focused Trp165 analysis

Trp165 reads the staged `$RUN_ROOT/full_data` copy plus regular masks and predictions. The output directory must not already exist.

```bash
cd "$REPO"
mave-trp165 \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/trp165_wt_extreme" \
  --rates 10 20 40 60 80 90 \
  --splits 1-50 \
  --repo-commit "$(git rev-parse HEAD)" \
  --source-checksum \
    "data/mthfr_crossAllcontext_domainannotation.csv=ed503e4cba97c46cfa5c1be7f2084d9f6ab535bbfe7db101faff146b75a81e4a"
```

Only `--run-root` and `--output-dir` are required. Rates default to the six regular labels, splits default to `1-50`, and the commit/checksum metadata is optional but recommended for an audited run.

Each `--source-checksum` is `SAFE_RELATIVE_PATH=64_HEX_SHA256` metadata; repeat it to record additional staged source files. Without installation:

```bash
PYTHONPATH=statistics/code python -m cli.analyze_trp165_wt_extremes \
  --run-root "$RUN_ROOT" --output-dir "$RUN_ROOT/trp165_wt_extreme" \
  --rates 10 20 40 60 80 90 --splits 1-50 \
  --repo-commit "$(git rev-parse HEAD)"
```

The Pitt wrapper `statistics/cluster/submit_trp165_wt_extreme_analysis.sh` instead accepts positional `STAGE_ROOT RUN_ROOT OUTPUT_DIR REPO_COMMIT [RATES_CSV] [SPLITS]`, computes checksums for its staged statistics source, and submits the same analysis.

### 10. Generate figures

Figures consume the raw loss directory, no-double loss directory, and the 16 statistics CSVs:

```bash
cd "$REPO"
mave-figures \
  --results-dir "$RUN_ROOT/splits_results_0506" \
  --nodouble-results-dir "$RUN_ROOT/splits_results_0506" \
  --statistics-dir "$RUN_ROOT/statistics_outputs" \
  --output-dir "$RUN_ROOT/figures"
```

All four directory options are required.

Without installation:

```bash
PYTHONPATH=statistics/code python -m cli.generate_figures \
  --results-dir "$RUN_ROOT/splits_results_0506" \
  --nodouble-results-dir "$RUN_ROOT/splits_results_0506" \
  --statistics-dir "$RUN_ROOT/statistics_outputs" \
  --output-dir "$RUN_ROOT/figures"
```

## Generated file contracts

`R` below is a rate label, `S` is a 1-based split, `SRC`/`TGT` are context names, and `K` is a PCA component count.

### Splits and masks

| Producer | Filename/pattern | Granularity | Key contents |
| --- | --- | --- | --- |
| `split_generator.py` | `$RUN_ROOT/data_splits/test_frac_R/train_split_rR_sS.csv` | Rate × split | Same 24-column schema as full data; deliberate score/SE holdouts are missing. |
| `split_generator.py` | `$RUN_ROOT/data_splits/test_frac_R/mask_rR_sS.csv` | Rate × split | `hgvs_pro` and annotations plus 0/1 score/SE fields; `1` identifies deliberate holdout. |
| `split_generator_nodouble_missing.py` | `$RUN_ROOT/data_splits_no_double_missing/tgt_TGT/test_frac_R/train_split_rR_sS.csv` | Target × rate × split | Same 24 columns; deliberate missingness only in `TGT_score` and `TGT_se`. |
| `split_generator_nodouble_missing.py` | `$RUN_ROOT/data_splits_no_double_missing/tgt_TGT/test_frac_R/mask_rR_sS.csv` | Target × rate × split | Only the selected target score/SE can contain holdout markers; natural missingness is not marked. |

### Model predictions

Regular outputs are under `$RUN_ROOT/data_splits`; no-double outputs for the five between-map families are under `$RUN_ROOT/data_splits_no_double_missing/tgt_TGT` with the same model directory patterns.

| Producer | Directory and filename | Granularity | Key contents |
| --- | --- | --- | --- |
| `colmean_imputer_runonsplits.py` | `mean_imputed_R/split_S/mean_imputed_splitS.csv` | Rate × split, wide | Original columns with missing numeric score/SE fields filled by `pos_aminoacid` group mean, then global column mean. |
| `knn_runon_splits.py` | `blosum_knn_R_BLOSUM100_k5_w0[_direct]/split_S/blosum_knn_imputed_splitS_rR.csv` | Rate × mode × split, wide | Original data plus `CONTEXT_score_imputed`, `CONTEXT_se_imputed`, and corresponding `_method` columns. |
| `pca_runon_splits.py` | `pca_kK_testfracR/split_S/pca_kK_splitS.csv` | Components × rate × split, wide | Original columns with eight score columns reconstructed; SE columns are not PCA targets. |
| `single_ae_runon*_splits.py` | `single_AE3_testfracR/split_S/SRC_to_TGT_singleAE.csv` | Rate × split × ordered pair | `hgvs_pro`, source score, `TGT_imputed`, source/target training flags, relation, and model. Regular includes self-pairs; no-double uses seven sources per target. |
| `dual_ae_runon*_splits.py` | `Dual_AE3_testfracR/split_S/SRC_to_TGT_DualAE.csv` | Rate × split × between-map pair | `hgvs_pro`, source/target train-set scores, source and target imputations, training flags, relation, and model. |
| `mice_runon*_splits.R` | `mice_test_frac_R/split_S/mice_imputed_SRC_to_TGT_splitS_rateR.csv` | Rate × split × between-map pair | `hgvs_pro`, covariates, completed pair score/SE fields, and `map_pair`. |
| `mice_rf_run*split*.R` | `mice_test_rf2_frac_R/split_S/miceRF_imputed_SRC_to_TGT_splitS_rateR.csv` | Rate × split × ordered pair | `hgvs` identifier, covariates, completed pair score/SE fields, and `map_pair`; regular output includes self-pairs and no-double uses seven sources per target. |
| `lmm_runon*_splits.R` | `linear_model_output_R/split_S/{basic_linear,oneparam_linear,full_interaction_linear,full_interaction_mixed,mixed_random}_SRC_score_to_TGT_score_sS_rR.csv` | Model variant × rate × split × between-map pair | `hgvs_pro`, retained covariates, source/target score and available SE fields, filled target predictions, and `map_pair`. |

### Loss inputs required by statistics

All files are written to `$RUN_ROOT/splits_results_0506`. Rows are generally context/pair × rate × split summaries and include the eligible point counts used for pooled RMSE.

| Producer | Fixed filename | Granularity | Key contents/columns |
| --- | --- | --- | --- |
| `singleae_loss_measure.py` | `single_AE3_rmse_results.csv` | Pair × regular rate × split | `test`, `regression_test_loss`, `double_missing` where applicable; `n_*`, `rate`, `split`, `src-tgt`. |
| `doubleae_loss_measure.py` | `dual_AE3_rmse_results.csv` | Pair × regular rate × split | Regression/double-missing RMSE and counts, `rate`, `split`, `src-tgt`. |
| `measure_loss_on_splits_mice.py` | `mice_loss_measurements_all_splits_rates2.csv` | Pair × regular rate × split | `rmse_regression`, `rmse_double_missing`, counts, `rate`, `split`, `src-tgt`, `model`. |
| `measure_loss_on_splits_RFmice2.py` | `mice_loss_measurements_all_splits_ratesrf2.csv` | Pair × regular rate × split | Adds within-map target RMSE for self-pairs plus regression/double-missing RMSE and counts. |
| `measure_loss_on_splits_linearmodels.py` | `linear_model_loss_measurements_all_splits_rates2.csv` | Linear variant × pair × rate × split | `rmse_test`, `rmse_double_missing`, counts, model, rate, split, and `src-tgt`. |
| `measure_loss_on_knn_imputer.py --sim_mode direct` | `blosum_knn_direct_rmse_all_splits.csv` | Column × regular rate × split | `test_fraction`, `split`, `model_file`, `column`, `loss`, `n_test`, `n_training`. |
| `measure_loss_on_splits_colmean.py` | `col_mean_imputed_results.csv` | Column × regular rate × split | Same column-oriented loss/count schema. |
| `pca_loss_measure.py` | `pca_rmse_results_all.csv` | Context × components × regular rate × split | `test`, `training`, counts, `rate`, `split`, `src-tgt`, `n_components`. |
| `loss_measure_singleae_no_double_missing.py` | `single_AE3_rmse_no_double_missing.csv` | Pair × no-double target/rate/split | Regression RMSE/count plus identifiers and diagnostic buckets. |
| `loss_measure_dualae_no_double_missing.py` | `dual_AE3_rmse_no_double_missing.csv` | Pair × no-double target/rate/split | Regression RMSE/count plus identifiers and diagnostic buckets. |
| `loss_measure_mice_no_double_missing.py` | `mice_loss_no_double_missing.csv` | Pair × no-double target/rate/split | `rmse_regression`, counts, `model`, `target_context`, `rate`, `split`, `src-tgt`. |
| `loss_measure_micerf_no_double_missing.py` | `mice_rf_loss_no_double_missing.csv` | Pair × no-double target/rate/split | Same MICE-style RMSE/count fields for RF. |
| `loss_measure_linearmodels_no_double_missing.py` | `linear_model_loss_no_double_missing.csv` | Linear variant × pair × no-double target/rate/split | `rmse_test`, counts, model, target, rate, split, and `src-tgt`. |

`pca_loss_measure.py` also writes per-component `pca_kK_rmse_results.csv` files, and standalone difference-mode KNN loss writes `blosum_knn_rmse_all_splits.csv`; neither is among the 13 required statistics inputs.

### Variance decomposition

| Producer | Filename | Granularity | Key contents |
| --- | --- | --- | --- |
| decomposition finalizer | `decomposition_sufficient_stats.csv` | Model variant × rate × split × task/subtype × pair × variant class | Additive `N`, error sums, `SSE`, sum of squared reported target SEs (`Q`), and truth/prediction cross-moments. |
| decomposition finalizer | `decomposition_file_validation.csv` | Logical prediction input | Source path/size/header signature, eligible counts, and `validation_status`. |
| decomposition finalizer | `decomposition_method_counts.csv` | Sufficient-statistic group × KNN method | `prediction_method` and `N`; empty for layouts without method labels. |
| decomposition finalizer/completer | `decomposition_run_manifest.json` | Run | Run root, commits, layout checksum, requested coordinates, shard counts, development flag, timestamps, and status. |
| `summarize_decomposition` | `decomposition_primary_pooled.csv` | Primary model × rate × task/subtype | Cell-pooled additive moments; MSE/RMSE; the combined `tau2_plus_b2_raw` term; its `VR` and `rho` error-component ratios; empirical mean and squared mean error; and calibration metrics. The output does not identify prediction variance and squared bias separately. |
| `summarize_decomposition` | `decomposition_pair_map_pooled.csv` | Model × rate × task/subtype × pair/map | Pair/map-level pooled moments and derived metrics. |
| `summarize_decomposition` | `decomposition_reconciliation.csv` | Requested model/rate/task grouping | Expected versus observed counts and `validation_status`. |
| `summarize_split_variability` | `decomposition_pair_map_by_split.csv` | Model × rate × split × task/subtype × pair/map | Pair/map metrics derived separately per split. |
| `summarize_split_variability` | `decomposition_pair_map_split_summary.csv` | Pair/map stratum across splits | Split count and mean, SD, variance, MCSE, quantiles, and extrema for each metric. |
| `summarize_split_variability` | `decomposition_pooled_by_split.csv` | Model × rate × split × task/subtype | Cell-pooled metrics derived separately per split. |
| `summarize_split_variability` | `decomposition_pooled_split_summary.csv` | Pooled stratum across splits | Across-split mean, SD, variance, MCSE, quantiles, and extrema. |

The decomposition output root also contains an internal task manifest, shard CSVs, and logs; the 11 files above are the final scientific bundle.

### Statistics CSVs

| Producer | Filename | Granularity | Key contents |
| --- | --- | --- | --- |
| `mave-stats` | `nodouble_model_rate_completeness.csv` | Dataset/loss/model/rate | Observed/expected rows and completeness fraction. |
| `mave-stats` | `pairwise_mwu_regression_test.csv` | Rate × model pair | Split counts, RMSE means/medians/differences, U, raw/Bonferroni p, family metadata. |
| `mave-stats` | `pairwise_mwu_double_missing.csv` | Rate × model pair | Same headline schema for regular B0 loss. |
| `mave-stats` | `pairwise_mwu_within_map.csv` | Rate × model pair | Same headline schema for W loss. |
| `mave-stats` | `pairwise_mwu_nodouble_regression_test.csv` | No-double rate × model pair | Same headline schema after completeness filtering. |
| `mave-stats` | `pairwise_mwu_by_context_regression_test.csv` | Rate × context/pair × model pair | Context completeness, U/raw p, within-context and rate-wide Bonferroni p, test status. |
| `mave-stats` | `pairwise_mwu_by_context_double_missing.csv` | Rate × context/pair × model pair | Same context schema for regular B0 loss. |
| `mave-stats` | `pairwise_mwu_by_context_within_map.csv` | Rate × map × model pair | Same context schema for W loss. |
| `mave-stats` | `pairwise_mwu_by_context_nodouble_regression_test.csv` | No-double rate × context/pair × model pair | Same context schema after completeness filtering. |
| `mave-stats` | `rmse_summary_regression_test.csv` | Model × regular rate | Split count, mean/SE RMSE, 95% normal interval, and formatted mean ± SE. |
| `mave-stats` | `rmse_summary_double_missing.csv` | Model × regular rate | Same summary schema for B0 loss. |
| `mave-stats` | `rmse_summary_within_map.csv` | Model × regular rate | Same summary schema for W loss. |
| `mave-stats` | `rmse_summary_nodouble_regression_test.csv` | Model × no-double rate | Same summary schema after completeness filtering. |
| `mave-stats` | `vs_colmean_regression_test.csv` | Rate × model versus Column Mean | Model/baseline split counts, mean/median differences, percent improvement, U and Bonferroni inference. |
| `mave-stats` | `vs_colmean_double_missing.csv` | Rate × model versus Column Mean | Same baseline-comparison schema for B0 loss. |
| `mave-stats` | `vs_colmean_within_map.csv` | Rate × model versus Column Mean | Same baseline-comparison schema for W loss. |

### Trp165 products

| Producer | Filename | Granularity | Key contents |
| --- | --- | --- | --- |
| `mave-trp165` | `trp165_wt_extreme_events.csv` | Method × rate × direction × split × Trp165 variant | Prediction, truth, residual, squared error, saturation, and source file. |
| `mave-trp165` | `trp165_wt_extreme_by_split.csv` | Method × rate × direction × split | `N`, `SSE`, `MSE`, `RMSE`, and status. |
| `mave-trp165` | `trp165_wt_extreme_primary.csv` | Method × rate × direction | Pooled `N`, `SSE`, `MSE`, `RMSE`, rank, and status. |
| `mave-trp165` | `trp165_wt_extreme_column_mean_diagnostics.csv` | Rate × split × target | Observed target count, position/whole-map means, fallback flag, and reference value. |
| `mave-trp165` | `trp165_wt_extreme_manifest.json` | Run | Arguments, commits/checksums, input inventory and coverage, reconciliation, artifact hashes, and timestamps. |

### Figures

`mave-figures` writes every stem as both PNG and SVG and writes `figure_manifest.json` with input/output inventories, hashes, raw-to-published names, and timestamps.

| Producer | Filename pattern | Granularity | Key contents |
| --- | --- | --- | --- |
| `mave-figures` | `fig2_within_map_barchart.{png,svg}` | One figure, two formats | Within-map RMSE bars. |
| `mave-figures` | `fig3_regression_test_barchart.{png,svg}` | One figure, two formats | Source-observed between-map RMSE bars. |
| `mave-figures` | `fig4_double_missing_barchart.{png,svg}` | One figure, two formats | Source-missing between-map RMSE bars. |
| `mave-figures` | `nodouble_regression_test_barchart.{png,svg}` | One figure, two formats | No-double between-map RMSE bars. |
| `mave-figures` | `fig5_coverage_vs_accuracy.{png,svg}` | One figure, two formats | Between-map coverage/accuracy trajectory. |
| `mave-figures` | `fig5b_coverage_vs_accuracy_panels.{png,svg}` | One figure, two formats | Coverage/accuracy panels. |
| `mave-figures` | `fig5c_risk_coverage_curves.{png,svg}` | One figure, two formats | Risk-coverage curves. |
| `mave-figures` | `fig5d_accuracy_composition_panels.{png,svg}` | One figure, two formats | Accuracy-composition panels. |
| `mave-figures` | `fig5e_regime_dominance_heatmap.{png,svg}` | One figure, two formats | Regime-dominance heatmap. |
| `mave-figures` | `fig6_point_composition.{png,svg}` | One figure, two formats | Between-map point composition. |
| `mave-figures` | `fig7_degradation_by_rate.{png,svg}` | One figure, two formats | Performance degradation by rate. |
| `mave-figures` | `fig8_best_rmse_per_saturation.{png,svg}` | One figure, two formats | Best RMSE at each saturation. |
| `mave-figures` | `figS1_pca_k_sensitivity.{png,svg}` | One figure, two formats | PCA component sensitivity. |
| `mave-figures` | `figS2a_pct_improvement_regression_test.{png,svg}` | One figure, two formats | Percent improvement for B1 loss. |
| `mave-figures` | `figS2b_pct_improvement_double_missing.{png,svg}` | One figure, two formats | Percent improvement for B0 loss. |
| `mave-figures` | `figS2c_pct_improvement_within_map.{png,svg}` | One figure, two formats | Percent improvement for W loss. |
| `mave-figures` | `figS3a_regression_test_scatter.{png,svg}` | One figure, two formats | B1 comparison scatter. |
| `mave-figures` | `figS3b_double_missing_scatter.{png,svg}` | One figure, two formats | B0 comparison scatter. |
| `mave-figures` | `figS3c_within_map_scatter.{png,svg}` | One figure, two formats | W comparison scatter. |
| `mave-figures` | `figure_manifest.json` | One run | Figure inventory and provenance record. |

## Provenance and limitations

- Imputation and decomposition were consolidated from commit `e6a62ccacfb4340baa3827394cad2add23ec73ad`.
- Statistics were consolidated from base commit `6ec548979781495f5cea6e39b2e106c8464cfc8d` plus the intentional production working-tree updates present at consolidation time.
- Decomposition recognizes the exact legacy corrected commits `aa9ad790d3cd1001f785f16cf3856c3e1ea67755` and `e6a62ccacfb4340baa3827394cad2add23ec73ad`. In this fresh history it also recognizes consolidated corrected root `09bf3c9fc7e150947015c8ed1c428483936d97e3` and its descendants.
- The authoritative CSV checksum is `ed503e4cba97c46cfa5c1be7f2084d9f6ab535bbfe7db101faff146b75a81e4a`; verify the staged external copy against it.
- This repository has a clean new history; earlier development histories were not imported. Configure the desired remote explicitly before publishing or cloning it into a production run root.
- Generated scientific outputs are intentionally external and ignored. Pitt scheduler logs are also ignored but are written under `cluster/` or `cluster/logs/` by the bundled wrappers. The wrappers preserve established run-root, environment, partition, memory, time, and output conventions; adapt them for other systems.
- Split generation uses base seed 42 plus the zero-based split index. SingleAE and DualAE pass PyTorch seed 42 for each fit. MICE-PMM and MICE-RF use R seed 123. Other methods are deterministic conditional on their inputs and library behavior. Seeds improve repeatability but do not promise bitwise-identical results across package versions, hardware, thread schedules, or CPU/GPU backends.
- The no-double master includes all five required labels in one graph, with `999` generated only from raw `99.9%` missingness. The regular master generates both KNN modes and explicitly evaluates direct KNN. All 13 required loss CSVs are co-located in `$RUN_ROOT/splits_results_0506`.
