#!/bin/bash

# Set library path to include conda environment libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Get Hugging Face username
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Logged in as: $HF_USER"

BASE_JOB_NAME="act_normal_so100_more_data"
BASE_OUTPUT_DIR="outputs/train/${BASE_JOB_NAME}"

# Random seeds to loop over
SEEDS=(42 123 456)

echo "Collecting Datasets"
# List directories, grep for _simple, and remove trailing slashes
DATASETS_NAME=$(ls -l ~/.cache/huggingface/lerobot/${HF_USER}/ | grep _simple |  grep -v concept | awk '{print $NF}' | sed 's/\/$//')

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
ENABLE_WANDB=true  # Set to true to enable Weights & Biases logging

LEARNING_RATE=1e-5
BATCH_SIZE=8
STEPS=100000

# Loop over each seed
for SEED in "${SEEDS[@]}"; do
  echo "Starting training with seed $SEED"
  
  # Update job name and output directory to include seed
  JOB_NAME="${BASE_JOB_NAME}_seed${SEED}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}_seed${SEED}"
  
  # Set up wandb flag
  WANDB_FLAG="--wandb.enable=false"
  if [ "$ENABLE_WANDB" = true ]; then
      WANDB_FLAG="--wandb.enable=true --wandb.disable_artifact=false --wandb.run_id=${JOB_NAME}"
      echo "Weights & Biases logging enabled"
  fi

  echo "Training with seed $SEED"
  python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_LIST \
    --policy.type=act \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${JOB_NAME} \
    --policy.device=cuda \
    --policy.optimizer_lr=$LEARNING_RATE \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --log_freq=1000 \
    --seed=$SEED \
    $WANDB_FLAG
    
  echo "Completed training with seed $SEED"
done

echo "All training runs completed!"
