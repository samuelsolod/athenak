#!/bin/bash

# Restrict to GPUs 4, 5, 6, 7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# This script runs Athena++ simulation and sends an email notification.

EXECUTABLE="/data/solod/simulations/grTorus/torusAMR/simulation_8/athena"
INPUT=gr_chakrabarti_torus_sane_8_4.athinput

INPUT_FILE="/data/solod/simulations/grTorus/torusAMR/simulation_8/inputs/$INPUT"

# Use 4 MPI ranks (1 per GPU assumed here)
mpirun -n 4 --mca pml ucx -x UCX_TLS=sm,cuda,cuda_copy,gdr_copy,cuda_ipc \
  -x CUDA_VISIBLE_DEVICES \
  $EXECUTABLE -i $INPUT_FILE




