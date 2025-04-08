#!/bin/bash

# Configuration variables
SOURCE_PREFIX="individual_cases_simple_"
TARGET_SUFFIX="_with_concepts"

# Set library path to include conda environment libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Get Hugging Face username
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Logged in as: $HF_USER"

# Ensure we have permission to USB ports (for potential hardware access)
sudo chmod a+rw /dev/ttyACM0 2>/dev/null || true
sudo chmod a+rw /dev/ttyACM1 2>/dev/null || true

echo "Creating concept-enhanced datasets..."



# Run the add_features_from_metadata.py script with our config
python lerobot/scripts/add_features_from_metadata.py \
    --prefix "$SOURCE_PREFIX" \
    --target-suffix "$TARGET_SUFFIX" 
    
echo "Dataset enhancement completed!"

# List the generated datasets
echo "Generated datasets:"
ls -la ~/.cache/huggingface/lerobot/"$HF_USER"/ | grep "${SOURCE_PREFIX}${TARGET_SUFFIX}" || echo "No datasets found. Check for errors above." 