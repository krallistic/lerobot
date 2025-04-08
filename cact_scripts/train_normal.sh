#!/bin/bash

# Set library path to include conda environment libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Get Hugging Face username
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Logged in as: $HF_USER"

echo "Collecting Datasets"
# List directories, grep for _simple, and remove trailing slashes
DATASETS_NAME=$(ls -l ~/.cache/huggingface/lerobot/${HF_USER}/ | grep _simple | awk '{print $NF}' | sed 's/\/$//')

echo "Building dataset list"
# Build a comma-separated list of datasets with the user prefix
DATASET_LIST=""
for dataset in $DATASETS_NAME; do
  if [[ -z "$DATASET_LIST" ]]; then
    DATASET_LIST="${HF_USER}/${dataset}"
  else
    DATASET_LIST="${DATASET_LIST},${HF_USER}/${dataset}"
  fi
done

echo "Dataset list: $DATASET_LIST"

echo "Training"
python lerobot/scripts/train.py \
--dataset.repo_id=$DATASET_LIST \
--policy.type=act \
--output_dir=outputs/train/act_so100_normal \
--job_name=act_so100_normal \
--policy.device=cuda \
--wandb.enable=false