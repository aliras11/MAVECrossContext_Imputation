#!/bin/bash
# Master script: submit the full no-double-missing pipeline with SLURM dependencies
# Stage 1: Generate splits
# Stage 2: Run all 5 between-map models in parallel (after splits)
# Stage 3: Measure losses (after all models complete)

set -e
mkdir -p logs

echo "Stage 1: Submitting split generator..."
SPLIT_JOB=$(sbatch --parsable submit_nodouble_split_generator.sh)
echo "  Split generator job: $SPLIT_JOB"

echo "Stage 2: Submitting model runs (depend on splits)..."
J1=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_nodouble_lmm.sh)
echo "  LMM job array: $J1"

J2=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_nodouble_mice.sh)
echo "  MICE PMM job array: $J2"

J3=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_nodouble_mice_rf.sh)
echo "  MICE RF job array: $J3"

J4=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_nodouble_single_ae.sh)
echo "  SingleAE job array: $J4"

J5=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB submit_nodouble_dual_ae.sh)
echo "  DualAE job array: $J5"

echo "Stage 3: Submitting loss measurement (depend on all models)..."
LOSS_JOB=$(sbatch --parsable --dependency=afterok:$J1:$J2:$J3:$J4:$J5 submit_nodouble_measure_losses.sh)
echo "  Loss measurement job: $LOSS_JOB"

echo ""
echo "Pipeline submitted:"
echo "  Splits:   $SPLIT_JOB"
echo "  LMM:      $J1 (32 tasks)"
echo "  MICE PMM: $J2 (32 tasks)"
echo "  MICE RF:  $J3 (32 tasks)"
echo "  SingleAE: $J4 (32 tasks)"
echo "  DualAE:   $J5 (32 tasks)"
echo "  Losses:   $LOSS_JOB (after all above)"
