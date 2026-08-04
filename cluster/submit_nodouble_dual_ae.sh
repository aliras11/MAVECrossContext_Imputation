#!/bin/bash
#SBATCH --job-name=nodouble_dual_ae
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --array=0-31
#SBATCH --output=logs/nodouble_dual_ae_%A_%a.out

# 4 rates x 8 targets = 32 array tasks
eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation
DATA_SPLITS=$ROOT/data_splits_no_double_missing

RATES=(10 40 80 99)
TARGETS=(av12 av25 av100 av200 wt12 wt25 wt100 wt200)

RATE_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
TGT_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
RATE=${RATES[$RATE_IDX]}
TARGET=${TARGETS[$TGT_IDX]}

echo "Rate=$RATE Target=$TARGET"

python $SCRIPTS/dual_ae_runon_nodouble_splits.py \
  --base-dir $DATA_SPLITS \
  --num-splits 50 \
  --missing-rate $RATE \
  --target $TARGET
