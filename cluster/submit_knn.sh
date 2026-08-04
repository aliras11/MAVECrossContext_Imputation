#!/bin/bash
#SBATCH --job-name=knn
#SBATCH --output=knn_%A_%a.out
#SBATCH --error=knn_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --array=0-11

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation
DATA_SPLITS=$ROOT/data_splits

RATES=(10 20 40 60 80 90)
MODES=(diff direct)

RATE_IDX=$(( SLURM_ARRAY_TASK_ID % 6 ))
MODE_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
RATE=${RATES[$RATE_IDX]}
MODE=${MODES[$MODE_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: rate=$RATE sim_mode=$MODE"

python $SCRIPTS/knn_runon_splits.py \
  --base-dir $DATA_SPLITS \
  --num-splits 50 \
  --missing-rate $RATE \
  --sim_mode $MODE
