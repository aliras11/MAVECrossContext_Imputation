#!/bin/bash
#SBATCH --job-name=nodouble_colmean
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --array=0-39
#SBATCH --output=logs/nodouble_colmean_%A_%a.out

set -e

# 5 rates x 8 targets = 40 array tasks
eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation
DATA_SPLITS=$ROOT/data_splits_no_double_missing

RATES=(10 40 80 99 999)
TARGETS=(av12 av25 av100 av200 wt12 wt25 wt100 wt200)

RATE_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
TGT_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
RATE=${RATES[$RATE_IDX]}
TARGET=${TARGETS[$TGT_IDX]}

echo "Rate=$RATE Target=$TARGET"

python $SCRIPTS/colmean_imputer_runonsplits.py \
  --base-dir $DATA_SPLITS/tgt_$TARGET \
  --num-splits 50 \
  --missing-rate $RATE
