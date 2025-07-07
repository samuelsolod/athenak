#!/bin/bash

# Run Athena++ from restart file
EXECUTABLE="/data/solod/simulations/grTorus/torusAMR/simulation_25/athena"
RESTART_FILE="/data/solod/simulations/grTorus/torusAMR/simulation_25/rst/torus.00014.rst"
INPUT=gr_chakrabarti_torus_sane_8_4_adaptive.athinput
INPUT_FILE="/data/solod/simulations/grTorus/torusAMR/simulation_25/inputs/$INPUT"

mpirun -n 64 $EXECUTABLE -r $RESTART_FILE -i $INPUT_FILE

