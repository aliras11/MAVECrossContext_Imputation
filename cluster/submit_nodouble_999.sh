#!/bin/bash
# Submit no-double-missing pipeline for rate=999 (99.9% missingness / 0.1% saturation)
# Stage 1: Generate splits
# Stage 2: Run all 5 between-map models (8 targets each)
# Stage 3: Measure losses

set -e
export PATH=/opt/slurm/bin:$PATH
mkdir -p logs

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation
DATA_SPLITS=$ROOT/data_splits_no_double_missing
FULL_CSV=$ROOT/MAVE-Imputation-Pipeline/data/mthfr_crossAllcontext_domainannotation.csv

TARGETS="av12 av25 av100 av200 wt12 wt25 wt100 wt200"

# --- Stage 1: Generate splits ---
echo "Stage 1: Submitting split generator for rate=99.9%..."
SPLIT_JOB=$(sbatch --parsable <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=nd999_splits
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=dept_cpu
#SBATCH --output=logs/nd999_splits_%j.out

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
python $ROOT/MAVE-Imputation-Pipeline/imputation/split_generator_nodouble_missing.py \
  --full-data $ROOT/MAVE-Imputation-Pipeline/data/mthfr_crossAllcontext_domainannotation.csv \
  --n-splits 50 \
  --rate 99.9 \
  --output-dir $ROOT/data_splits_no_double_missing \
  --seed 42
SBATCH
)
echo "  Split job: $SPLIT_JOB"

# --- Stage 2: Model runs (array=0-7, one per target) ---
echo "Stage 2: Submitting model runs..."

J1=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=nd999_lmm
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --array=0-7
#SBATCH --output=logs/nd999_lmm_%A_%a.out

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
TARGETS=(av12 av25 av100 av200 wt12 wt25 wt100 wt200)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
echo "LMM Rate=999 Target=$TARGET"

Rscript $ROOT/MAVE-Imputation-Pipeline/imputation/lmm_runon_nodouble_splits.R \
  --splits_dir $ROOT/data_splits_no_double_missing \
  --rate 999 \
  --target $TARGET
SBATCH
)
echo "  LMM: $J1"

J2=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=nd999_mice
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --array=0-7
#SBATCH --output=logs/nd999_mice_%A_%a.out

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
TARGETS=(av12 av25 av100 av200 wt12 wt25 wt100 wt200)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
echo "MICE PMM Rate=999 Target=$TARGET"

Rscript $ROOT/MAVE-Imputation-Pipeline/imputation/mice_runon_nodouble_splits.R \
  $ROOT/data_splits_no_double_missing 999 $TARGET
SBATCH
)
echo "  MICE PMM: $J2"

J3=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=nd999_micerf
#SBATCH --time=72:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --array=0-7
#SBATCH --output=logs/nd999_micerf_%A_%a.out

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
TARGETS=(av12 av25 av100 av200 wt12 wt25 wt100 wt200)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
echo "MICE RF Rate=999 Target=$TARGET"

Rscript $ROOT/MAVE-Imputation-Pipeline/imputation/mice_rf_runon_nodouble_splits.R \
  $ROOT/data_splits_no_double_missing 999 $TARGET
SBATCH
)
echo "  MICE RF: $J3"

J4=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=nd999_sae
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --array=0-7
#SBATCH --output=logs/nd999_sae_%A_%a.out

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
TARGETS=(av12 av25 av100 av200 wt12 wt25 wt100 wt200)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
echo "SingleAE Rate=999 Target=$TARGET"

python $ROOT/MAVE-Imputation-Pipeline/imputation/single_ae_runon_nodouble_splits.py \
  --base-dir $ROOT/data_splits_no_double_missing \
  --num-splits 50 \
  --missing-rate 999 \
  --target $TARGET
SBATCH
)
echo "  SingleAE: $J4"

J5=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=nd999_dae
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --array=0-7
#SBATCH --output=logs/nd999_dae_%A_%a.out

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
TARGETS=(av12 av25 av100 av200 wt12 wt25 wt100 wt200)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
echo "DualAE Rate=999 Target=$TARGET"

python $ROOT/MAVE-Imputation-Pipeline/imputation/dual_ae_runon_nodouble_splits.py \
  --base-dir $ROOT/data_splits_no_double_missing \
  --num-splits 50 \
  --missing-rate 999 \
  --target $TARGET
SBATCH
)
echo "  DualAE: $J5"

# --- Stage 3: Loss measurement ---
echo "Stage 3: Submitting loss measurement..."
LOSS_JOB=$(sbatch --parsable --dependency=afterok:$J1:$J2:$J3:$J4:$J5 <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=nd999_loss
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --output=logs/nd999_losses_%j.out

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation

echo "=== Linear models ==="
python $SCRIPTS/loss_measure_linearmodels_no_double_missing.py --rates 999
echo "=== MICE PMM ==="
python $SCRIPTS/loss_measure_mice_no_double_missing.py --rates 999
echo "=== MICE RF ==="
python $SCRIPTS/loss_measure_micerf_no_double_missing.py --rates 999
echo "=== SingleAE ==="
python $SCRIPTS/loss_measure_singleae_no_double_missing.py --rates 999
echo "=== DualAE ==="
python $SCRIPTS/loss_measure_dualae_no_double_missing.py --rates 999

echo "Done."
SBATCH
)
echo "  Loss measurement: $LOSS_JOB"

echo ""
echo "Pipeline submitted for rate=999 (0.1% saturation):"
echo "  Splits:   $SPLIT_JOB"
echo "  LMM:      $J1 (8 tasks)"
echo "  MICE PMM: $J2 (8 tasks)"
echo "  MICE RF:  $J3 (8 tasks)"
echo "  SingleAE: $J4 (8 tasks)"
echo "  DualAE:   $J5 (8 tasks)"
echo "  Losses:   $LOSS_JOB"
