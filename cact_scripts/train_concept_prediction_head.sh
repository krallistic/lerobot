#!/bin/bash

# Configuration variables
DATASET_PREFIX="so101_individual_cases_new_with_concepts"


BASE_JOB_NAME="predictionhead_act_so101_10epoch"
BASE_OUTPUT_DIR="outputs/train/${BASE_JOB_NAME}"
DEVICE="cuda"  # Use "cuda" for GPU or "cpu" for CPU

# Random seeds to loop over
SEEDS=(42 123 456 100 101 102 103 104 105 106)
PERCENTS=(0.33 0.66 1.0)

CONCEPT_WEIGHT=0.2  # Weight for concept loss component
ENABLE_WANDB=true  # Set to true to enable Weights & Biases logging

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

LEARNING_RATE=3e-5
BATCH_SIZE=8
STEPS=30000

# Loop over each seed
for SEED in "${SEEDS[@]}"; do
  for PERCENT in "${PERCENTS[@]}"; do
    echo "Starting training with seed $SEED and $PERCENT"

    # Update job name and output directory to include seed
    JOB_NAME="${BASE_JOB_NAME}_percent_${PERCENT}_seed_${SEED}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_percent_${PERCENT}_seed_${SEED}"

    # Set up wandb flag
    WANDB_FLAG="--wandb.enable=false"
    if [ "$ENABLE_WANDB" = true ]; then
        WANDB_FLAG="--wandb.enable=true --wandb.disable_artifact=true --wandb.run_id=${JOB_NAME}"
        echo "Weights & Biases logging enabled"
    fi

    echo "Starting training with ConceptACT policy using seed $SEED"
    python lerobot/scripts/train.py \
        --dataset.repo_id=$DATASET_LIST \
        --policy.type=concept_act \
        --output_dir=$OUTPUT_DIR \
        --job_name=$JOB_NAME \
        --policy.device=$DEVICE \
        --policy.concept_weight=$CONCEPT_WEIGHT \
        --policy.optimizer_lr=$LEARNING_RATE \
        --batch_size=$BATCH_SIZE \
        --epochs=10 \
        --dataset_percent=$PERCENT \
        --policy.use_concept_learning=true \
        --policy.concept_method=prediction_head \
        --policy.n_heads=16 \
        --log_freq=10000 \
        --save_freq=0 \
        --seed=$SEED \
        $WANDB_FLAG

      echo "Completed training with seed $SEED and $PERCENT"
    done
  done

echo "All training runs completed!" 