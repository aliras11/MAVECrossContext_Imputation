#!/bin/bash
#SBATCH --job-name=knn_losses
#SBATCH --output=knn_losses_%j.out
#SBATCH --error=knn_losses_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation

python $SCRIPTS/measure_loss_on_knn_imputer.py
python $SCRIPTS/measure_loss_on_knn_imputer.py --sim_mode direct
