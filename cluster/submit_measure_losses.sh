#!/bin/bash
#SBATCH --job-name=measure_losses
#SBATCH --output=measure_losses_%j.out
#SBATCH --error=measure_losses_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dept_cpu

eval "$(conda shell.bash hook)"
conda activate struct

ROOT=/net/dali/home/roth/ser219/imputation_mthfr_run_mar6
SCRIPTS=$ROOT/MAVE-Imputation-Pipeline/imputation

python $SCRIPTS/singleae_loss_measure.py
python $SCRIPTS/doubleae_loss_measure.py
python $SCRIPTS/measure_loss_on_splits_colmean.py
python $SCRIPTS/measure_loss_on_knn_imputer.py
python $SCRIPTS/pca_loss_measure.py
python $SCRIPTS/measure_loss_on_splits_mice.py
python $SCRIPTS/measure_loss_on_splits_RFmice2.py
python $SCRIPTS/measure_loss_on_splits_linearmodels.py
