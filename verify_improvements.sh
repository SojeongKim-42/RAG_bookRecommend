#!/bin/bash

echo "Starting Retrieval Verification..."
echo "Environment: bkms"

# Ensure we are using the right python
source activate bkms || conda activate bkms || echo "Could not activate bkms, assuming active"

# 1. Ablation Study: Adaptive K Improvement (No Cross Encoder)
echo "----------------------------------------------------------------"
echo "Running Experiment: ablation_cross_encoder"
echo "Goal: Verify if improved Adaptive K (Relative Drop) works better than baseline"
python run_experiments.py --preset ablation_cross_encoder --no-save-individual

# 2. Improved Retrieval: + Cross Encoder
echo "----------------------------------------------------------------"
echo "Running Experiment: improved_retrieval"
echo "Goal: Verify impact of Cross-Encoder reranking"
python run_experiments.py --preset improved_retrieval --no-save-individual

echo "----------------------------------------------------------------"
echo "Verification Complete."
