#!/bin/bash
#SBATCH --job-name=lmm
#SBATCH --output=lmm_%A_%a.out
#SBATCH --error=lmm_%A_%a.err
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

Rscript $SCRIPTS/lmm_runon_splits.R \
  --splits_dir $DATA_SPLITS \
  --rate $RATE \
  --out_dir $DATA_SPLITS
