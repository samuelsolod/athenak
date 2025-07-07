#!/bin/bash

# This script executes an executable created in the home build directory
# and runs a simulation with the specified input files and parameters.

# Define the base directory for simulations
BASE_SIM_DIR="/data/solod/simulations/powerAMR_density"

# Path to Athena++ executable (assuming it's built in the BASE_SIM_DIR)
ATHENA_EXECUTABLE="athena"

# Input file of interest (assuming it's in the BASE_SIM_DIR)
INPUT_FILENAME="kh2d.athinput"

# Path to the plotting script
PLOT_SCRIPT_PATH="/home/solod/athenak/vis/python/plot_slice.py"

# Define arrays for alpha_refine and alpha_coarsen parameters
# Ensure these arrays have the same number of elements
ALPHA_REFINE_PARAMS=(4 5 6 7 4 5 6 7 4)
ALPHA_COARSEN_PARAMS=(5 6 7 8 6 7 8 9 7)

# Define array for stencil_order parameters
STENCIL_ORDER_PARAMS=(3 5)

# Check if the alpha parameter arrays have the same length
if [ ${#ALPHA_REFINE_PARAMS[@]} -ne ${#ALPHA_COARSEN_PARAMS[@]} ]; then
    echo "Error: ALPHA_REFINE_PARAMS and ALPHA_COARSEN_PARAMS arrays must have the same number of elements."
    exit 1
fi

# Initialize simulation counter
SIM_COUNTER=98

# Loop through the stencil_order parameters
for CURRENT_STENCIL_ORDER in "${STENCIL_ORDER_PARAMS[@]}"; do
    # Loop through the alpha_refine and alpha_coarsen parameters
    for i in "${!ALPHA_REFINE_PARAMS[@]}"; do
        CURRENT_ALPHA_REFINE=${ALPHA_REFINE_PARAMS[$i]}
        CURRENT_ALPHA_COARSEN=${ALPHA_COARSEN_PARAMS[$i]}

        # Determine the simulation number
        SIM_DIR="$BASE_SIM_DIR/simulation_$SIM_COUNTER"

        echo "-----------------------------------------------------"
        echo "Starting simulation_$SIM_COUNTER with:"
        echo "  alpha_refine = $CURRENT_ALPHA_REFINE"
        echo "  alpha_coarsen = $CURRENT_ALPHA_COARSEN"
        echo "  stencil_order = $CURRENT_STENCIL_ORDER"
        echo "-----------------------------------------------------"

        # Create the simulation directory
        mkdir -p "$SIM_DIR"

        # *** MODIFIED SECTION START ***
        # Always copy necessary files from the BASE_SIM_DIR
        cp "$BASE_SIM_DIR/$ATHENA_EXECUTABLE" "$SIM_DIR/"
        cp "$BASE_SIM_DIR/$INPUT_FILENAME" "$SIM_DIR/"
        # *** MODIFIED SECTION END ***

        # Go into the simulation directory
        cd "$SIM_DIR" || { echo "Error: Could not enter $SIM_DIR"; exit 1; }
        

        # Modify kh2d.athinput
        echo "Modifying $INPUT_FILENAME..."
        sed -i "s/alpha_refine = [0-9]*/alpha_refine = $CURRENT_ALPHA_REFINE/" "$INPUT_FILENAME"
        sed -i "s/alpha_coarsen = [0-9]*/alpha_coarsen = $CURRENT_ALPHA_COARSEN/" "$INPUT_FILENAME"
        # Add or modify stencil_order line
        if grep -q "stencil_order" "$INPUT_FILENAME"; then
            sed -i "s/stencil_order = [0-9]*/stencil_order = $CURRENT_STENCIL_ORDER/" "$INPUT_FILENAME"
        else
            # If stencil_order line doesn't exist, append it.
            # You might want to adjust where it's inserted based on your input file structure.
            echo "stencil_order = $CURRENT_STENCIL_ORDER" >> "$INPUT_FILENAME"
            echo "Warning: 'stencil_order' line not found, appended to end of $INPUT_FILENAME. Please verify placement."
        fi

        # Run the simulation
        echo "Running simulation..."
        mpirun -n 8 --mca pml ucx -x UCX_TLS=sm,cuda,cuda_copy,gdr_copy,cuda_ipc "./$ATHENA_EXECUTABLE" -i "$INPUT_FILENAME"

        # Define directories for images and binary files within the current simulation directory
        BIN_DIR="$SIM_DIR/bin"
        OUT_DIR="$SIM_DIR/imgs"

        # Create the output directory for images if it doesn't exist
        mkdir -p "$OUT_DIR"

        # Generate images from binary data
        echo "Generating images from binary data..."
        # Loop over all .bin files in the bin directory
        for file in "$BIN_DIR"/*.bin; do
            if [ -f "$file" ]; then # Check if the file exists
                # Extract the filename from the full path
                filename=$(basename "$file")

                # Define the output PNG filename
                output_file="$OUT_DIR/${filename%.bin}.png"

                # Run the Python plotting script
                python3 "$PLOT_SCRIPT_PATH" "$file" dens -l 0.5 --grid -c Oranges "$output_file"
            fi
        done
        echo "Images generated in $OUT_DIR"

        # Go back to the powerAMR directory
        cd "$BASE_SIM_DIR" || { echo "Error: Could not go back to $BASE_SIM_DIR"; exit 1; }

        echo "Finished simulation_$SIM_COUNTER."
        # Increment simulation counter for the next run
        SIM_COUNTER=$((SIM_COUNTER + 1))
    done
done

echo "All simulations completed."