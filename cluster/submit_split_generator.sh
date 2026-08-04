#!/bin/bash
#SBATCH --job-name=split_gen
#SBATCH --output=split_gen_%j.out
#SBATCH --error=split_gen_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=dept_cpu

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation
FULL_CSV=$ROOT/MAVE-Imputation-Pipeline/data/mthfr_crossAllcontext_domainannotation.csv
DATA_SPLITS=$ROOT/data_splits

python $SCRIPTS/split_generator.py \
  --full-data $FULL_CSV \
  --n-splits 50 \
  --rate 10 --rate 20 --rate 40 --rate 60 --rate 80 --rate 90 \
  --output-dir $DATA_SPLITS
