#!/bin/bash

# Configuration variables
BASE_JOB_NAME="concept_act_so100_transformer_bce_v2"
BASE_OUTPUT_DIR="outputs/train"
DEVICE="cuda"
ROBOT_TYPE="so100"
N_EPISODES=8

DATASET_PREFIX="individual_cases_simple_with_concepts"

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

echo $DATASET_LIST

echo "Looking for trained models with prefix: $BASE_JOB_NAME"

# Find all directories matching our training pattern
TRAIN_DIRS=$(find $BASE_OUTPUT_DIR -maxdepth 1 -type d -name "${BASE_JOB_NAME}_seed*" | sort)

if [ -z "$TRAIN_DIRS" ]; then
    echo "Error: No training directories found matching pattern ${BASE_JOB_NAME}_seed*"
    echo "Looking in: $BASE_OUTPUT_DIR"
    exit 1
fi

echo "Found training directories:"
echo "$TRAIN_DIRS"

# Use the first (or you could modify to use latest) training directory
SELECTED_DIR=$(echo "$TRAIN_DIRS" | head -n 1)
echo "Selected training directory: $SELECTED_DIR"

# Look for checkpoint directories
CHECKPOINT_DIR="$SELECTED_DIR/checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoints directory not found at $CHECKPOINT_DIR"
    exit 1
fi

# Find the latest checkpoint (highest numbered directory)
LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "[0-9]*" | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint directories found in $CHECKPOINT_DIR"
    exit 1
fi

# Construct the full model path
MODEL_PATH="$LATEST_CHECKPOINT/pretrained_model"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Pretrained model not found at $MODEL_PATH"
    exit 1
fi

echo "Using model from: $MODEL_PATH"

# Run the evaluation
echo "Starting evaluation..."
python cact_scripts/eval.py \
    --policy.path="$MODEL_PATH" \
    --robot.type="$ROBOT_TYPE" \
    --n_episodes="$N_EPISODES" \
    --policy.device="$DEVICE" \
    --dataset.repo_id=$DATASET_LIST

echo "Evaluation completed!"