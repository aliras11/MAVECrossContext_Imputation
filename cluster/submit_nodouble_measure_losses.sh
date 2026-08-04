#!/bin/bash
#SBATCH --job-name=nodouble_losses
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu
#SBATCH --output=logs/nodouble_losses_%j.out

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation

echo "=== Linear models ==="
python $SCRIPTS/loss_measure_linearmodels_no_double_missing.py

echo "=== MICE PMM ==="
python $SCRIPTS/loss_measure_mice_no_double_missing.py

echo "=== MICE RF ==="
python $SCRIPTS/loss_measure_micerf_no_double_missing.py

echo "=== SingleAE ==="
python $SCRIPTS/loss_measure_singleae_no_double_missing.py

echo "=== DualAE ==="
python $SCRIPTS/loss_measure_dualae_no_double_missing.py

echo "All loss measurements complete."
