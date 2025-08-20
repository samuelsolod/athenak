# !/bin/bash

# Directories
BIN_DIR="/data/solod/simulations/grTorus/torusAMR/simulation_25/bin"
OUT_DIR_DENS="/data/solod/simulations/grTorus/torusAMR/simulation_25/imgs_dens"
OUT_DIR_SIGMA="/data/solod/simulations/grTorus/torusAMR/simulation_25/imgs_sigma"
SCRIPT_PATH="/home/solod/athenak/vis/python/plot_slice.py"

# Create output directories if they don't exist
mkdir -p "$OUT_DIR_DENS"
mkdir -p "$OUT_DIR_SIGMA"

# Loop over all .bin files in the bin directory
for file in "$BIN_DIR"/*.bin; do
    # Extract the filename from the full path
    filename=$(basename "$file")
    index=$(echo "$filename" | grep -oE '[0-9]+' | tail -1)

    # Only plot if the index is greater than a threshold
    if [ "$index" -gt 42 ]; then
        # Output filenames
        output_dens="$OUT_DIR_DENS/${filename%.bin}.png"
        output_sigma="$OUT_DIR_SIGMA/${filename%.bin}.png"

        # Plot density
        python3 "$SCRIPT_PATH" "$file" -d x -n log \
            --horizon -l 0.0 --grid --grid_color black -c jet dens "$output_dens"

        # Plot sigma
        python3 "$SCRIPT_PATH" "$file" -d x -n log \
            --horizon -l 0.0 --grid --grid_color black --vmin 1e-7 --vmax 10 -c jet \
            derived:sigmah_rel "$output_sigma"
    fi
done

# Define input/output paths
VIDEO_INPUT="/data/solod/simulations/grTorus/torusAMR/simulation_13/imgs_dens/torus.mhd_w_bcc.%05d.png"
VIDEO_OUTPUT="/data/solod/simulations/grTorus/torusAMR/simulation_13/torus_dens.mp4"

# Create the video starting from image 00043
ffmpeg -framerate 10 -start_number 40 -i "$VIDEO_INPUT" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "$VIDEO_OUTPUT"



