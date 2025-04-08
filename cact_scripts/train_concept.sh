#!/bin/bash

# Configuration variables
DATASET_PREFIX="individual_cases_simple_with_concepts"
OUTPUT_DIR="outputs/train/concept_act_so100"
JOB_NAME="concept_act_so100"
DEVICE="cuda"  # Use "cuda" for GPU or "cpu" for CPU
LEARNING_RATE=1e-5
BATCH_SIZE=64
STEPS=100000
CONCEPT_WEIGHT=1.0  # Weight for concept loss component
ENABLE_WANDB=false  # Set to true to enable Weights & Biases logging

# Set library path to include conda environment libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Get Hugging Face username
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Logged in as: $HF_USER"

echo "Collecting concept datasets"
# List directories matching our prefix and remove trailing slashes
DATASETS_NAME=$(ls -l ~/.cache/huggingface/lerobot/${HF_USER}/ | grep $DATASET_PREFIX | awk '{print $NF}' | sed 's/\/$//')

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

# Set up wandb flag
WANDB_FLAG="--wandb.enable=false"
if [ "$ENABLE_WANDB" = true ]; then
    WANDB_FLAG="--wandb.enable=true"
    echo "Weights & Biases logging enabled"
fi

echo "Starting training with ConceptACT policy"
python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_LIST \
    --policy.type=concept_act \
    --output_dir=$OUTPUT_DIR \
    --job_name=$JOB_NAME \
    --policy.device=$DEVICE \
    --policy.concept_weight=$CONCEPT_WEIGHT \
    --policy.optimizer_lr=$LEARNING_RATE \
    --train.batch_size=$BATCH_SIZE \
    --train.steps=$STEPS \
    --policy.use_concept_learning=true \
    --policy.concept_method=prediction_head \
    $WANDB_FLAG

echo "Training completed!" 