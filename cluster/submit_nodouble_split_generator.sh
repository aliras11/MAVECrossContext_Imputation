#!/bin/bash
#SBATCH --job-name=nodouble_splits
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=dept_cpu
#SBATCH --output=logs/nodouble_splits_%j.out

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation
FULL_CSV=$ROOT/MAVE-Imputation-Pipeline/data/mthfr_crossAllcontext_domainannotation.csv
OUT_DIR=$ROOT/data_splits_no_double_missing

python $SCRIPTS/split_generator_nodouble_missing.py \
  --full-data $FULL_CSV \
  --n-splits 50 \
  --rate 10 --rate 40 --rate 80 --rate 99 --rate 99.9 \
  --output-dir $OUT_DIR \
  --seed 42
