#!/bin/bash
# Master submission script — runs all jobs in dependency order
# Usage: bash submit_all.sh

cd "$(dirname "$0")"

# Step 1: Generate splits
SPLIT_JOB=$(sbatch --parsable submit_split_generator.sh)
echo "Submitted split_generator: $SPLIT_JOB"

# Step 2: All imputation models (parallel, depend on splits)
J1=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_single_ae.sh)
J2=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_dual_ae.sh)
J3=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_colmean.sh)
J4=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_knn.sh)
J5=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_pca.sh)
J6=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_mice.sh)
J7=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_mice_rf.sh)
J8=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_lmm.sh)
echo "Submitted models: $J1 $J2 $J3 $J4 $J5 $J6 $J7 $J8"

# Step 3: Loss measurement (after all models finish)
J9=$(sbatch --parsable --dependency=afterok:$J1:$J2:$J3:$J4:$J5:$J6:$J7:$J8 submit_measure_losses.sh)
echo "Submitted measure_losses: $J9"

echo "All jobs submitted. Run 'squeue -u \$USER' to monitor."
