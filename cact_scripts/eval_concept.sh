#!/bin/bash

# Configuration variables
BASE_JOB_NAME="concept_act_so100_more_data"
BASE_OUTPUT_DIR="outputs/train/${BASE_JOB_NAME}"
ROBOT_TYPE="so100"  # Set to your robot type
DEVICE="cuda"

# Get Hugging Face username
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Logged in as: $HF_USER"


EVAL_DATASET_PREFIX="${HF_USER}/eval_${BASE_JOB_NAME}"

# Local dataset storage location
DATASET_ROOT="/home/robolab-server/.cache/huggingface/lerobot"  # Change this to your preferred local storage location



# Evaluation parameters
EVAL_EPISODES=3
FPS=30
TASK_DESCRIPTION="Grasp a lego block and put it in the bin."
WARMUP_TIME_S=10
EPISODE_TIME_S=30
RESET_TIME_S=10
PUSH_TO_HUB=false  # Set to false since we're using local datasets

# Random seeds to loop over (same as training script)
SEEDS=(42 123 456)

# Set library path to include conda environment libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


echo "Ensure we have permission to USB ports (for potential hardware access)"
sudo chmod a+rw /dev/ttyACM0 2>/dev/null || true
sudo chmod a+rw /dev/ttyACM1 2>/dev/null || true

echo "Starting evaluation of trained models on real robot for ${BASE_JOB_NAME}"





# Loop over each seed
for SEED in "${SEEDS[@]}"; do
  echo "====================================="
  echo "Evaluating model with seed $SEED on real robot"

  # Construct paths
  MODEL_DIR="${BASE_OUTPUT_DIR}_seed${SEED}/checkpoints"
  EVAL_DATASET_NAME="${EVAL_DATASET_PREFIX}_seed${SEED}"
  EVAL_DATASET_DIR="${DATASET_ROOT}/${EVAL_DATASET_NAME}"

  # Find the final checkpoint (highest number)
  FINAL_CHECKPOINT=$(find ${MODEL_DIR} -type d -name "[0-9]*" | sort -n | tail -1)

  if [ -z "$FINAL_CHECKPOINT" ]; then
    echo "No checkpoints found in ${MODEL_DIR}. Skipping seed ${SEED}."
    continue
  fi

  # Extract the step number from the checkpoint path
  STEP=$(basename ${FINAL_CHECKPOINT})
  CHECKPOINT_MODEL="${FINAL_CHECKPOINT}/pretrained_model"

  # Check if the model exists
  if [ ! -f "${CHECKPOINT_MODEL}/model.safetensors" ] || [ ! -f "${CHECKPOINT_MODEL}/config.json" ]; then
    echo "Warning: Incomplete model at ${CHECKPOINT_MODEL}, skipping evaluation for seed ${SEED}"
    continue
  fi

  echo "Evaluating final checkpoint at step ${STEP} for seed ${SEED}"

  # Check if evaluation dataset exists and delete it if it does
  if [ -d "${EVAL_DATASET_DIR}" ]; then
    echo "Dataset directory ${EVAL_DATASET_DIR} exists. Deleting..."
    rm -rf "${EVAL_DATASET_DIR}"
    echo "Deleted dataset directory ${EVAL_DATASET_DIR}"
  else
    echo "Dataset directory ${EVAL_DATASET_DIR} does not exist. Will create new in next step."
  fi

  # Run the evaluation using control_robot.py
  echo "Running evaluation with policy from step ${STEP}, seed ${SEED}"
  python lerobot/scripts/control_robot.py \
    --robot.type=${ROBOT_TYPE} \
    --control.type=record \
    --control.fps=${FPS} \
    --control.single_task="${TASK_DESCRIPTION}" \
    --control.repo_id=${EVAL_DATASET_NAME} \
    --control.num_episodes=${EVAL_EPISODES} \
    --control.warmup_time_s=${WARMUP_TIME_S} \
    --control.episode_time_s=${EPISODE_TIME_S} \
    --control.reset_time_s=${RESET_TIME_S} \
    --control.push_to_hub=${PUSH_TO_HUB} \
    --control.policy.path=${CHECKPOINT_MODEL} \
    --control.policy.device=${DEVICE}

  echo "Completed evaluation on real robot for seed ${SEED}"
done

echo "All real-world evaluations completed!"
