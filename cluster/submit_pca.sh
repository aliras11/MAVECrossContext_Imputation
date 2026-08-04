#!/bin/bash
#SBATCH --job-name=pca
#SBATCH --output=pca_%A_%a.out
#SBATCH --error=pca_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --array=0-5

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation
DATA_SPLITS=$ROOT/data_splits

RATES=(10 20 40 60 80 90)
RATE=${RATES[$SLURM_ARRAY_TASK_ID]}

python $SCRIPTS/pca_runon_splits.py \
  --base-dir $DATA_SPLITS \
  --num-splits 50 \
  --missing-rate $RATE
