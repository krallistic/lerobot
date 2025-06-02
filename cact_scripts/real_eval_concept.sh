#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

sudo chmod a+rw /dev/ttyACM1
sudo chmod a+rw /dev/ttyACM0

# Configuration variables
CHECKPOINTS=true  # Set to true to evaluate all checkpoints, false for latest checkpoint only
BASE_JOB_NAME="concept_act_so100_30krun"
BASE_JOB_NAME="act_normal_so100_30krun"
BASE_JOB_NAME="concept_act_prediction_head_so100_30krun"

#BASE_JOB_NAME="act_normal_so100_more_heads"
BASE_OUTPUT_DIR="outputs/train"
DEVICE="cuda"
ROBOT_TYPE="so100"
N_EPISODES=10

echo "=================================="
echo "Robot Evaluation Script"
echo "=================================="
echo "Base model pattern: $BASE_JOB_NAME"
echo "Checkpoint evaluation mode: $CHECKPOINTS"
echo "=================================="

DATASET_PREFIX="individual_cases_simple_with_concepts"

# Get Hugging Face username
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "Logged in as: $HF_USER"

echo "Collecting concept datasets"
# List directories matching our prefix and remove trailing slashes
DATASETS_NAME=$(ls -l ~/.cache/huggingface/lerobot/${HF_USER}/ | grep $DATASET_PREFIX | awk '{print $NF}' | sed 's/\/$//' | head -n 3)

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

# Create base evaluation output directory (organized by base job name only)
EVAL_BASE_DIR="outputs/eval/${BASE_JOB_NAME}"

echo "Using evaluation output directory: $EVAL_BASE_DIR"
mkdir -p "$EVAL_BASE_DIR"

# Find all directories matching our training pattern
TRAIN_DIRS=$(find $BASE_OUTPUT_DIR -maxdepth 1 -type d -name "${BASE_JOB_NAME}*" | sort)

if [ -z "$TRAIN_DIRS" ]; then
    echo "Error: No training directories found matching pattern ${BASE_JOB_NAME}*"
    echo "Looking in: $BASE_OUTPUT_DIR"
    exit 1
fi

echo "Found training directories:"
echo "$TRAIN_DIRS"
echo ""

# Count available models
MODEL_COUNT=$(echo "$TRAIN_DIRS" | wc -l)
echo "Found $MODEL_COUNT base model(s) to evaluate."
echo ""

# Function to extract model suffix (everything after BASE_JOB_NAME)
get_model_suffix() {
    local full_path="$1"
    local basename=$(basename "$full_path")
    # Remove the BASE_JOB_NAME prefix and any leading underscore
    echo "${basename#${BASE_JOB_NAME}}" | sed 's/^_//'
}

# Function to run evaluation for a specific model and checkpoint
run_single_evaluation() {
    local model_path="$1"
    local job_suffix="$2"
    local checkpoint_name="$3"

    # Construct output filename
    local output_filename="eval_result"
    if [ -n "$job_suffix" ]; then
        output_filename="${output_filename}_${job_suffix}"
    fi
    if [ -n "$checkpoint_name" ] && [ "$CHECKPOINTS" = true ]; then
        output_filename="${output_filename}_${checkpoint_name}"
    fi
    output_filename="${output_filename}.json"

    # Check if evaluation already exists (DUPLICATE CHECK)
    local output_file_path="$EVAL_BASE_DIR/$output_filename"
    if [ -f "$output_file_path" ]; then
        echo "⏭️  SKIPPING: $output_filename already exists at $output_file_path"
        echo "   To re-run this evaluation, delete the existing file first."
        echo ""
        return 0
    fi

    echo "----------------------------------------"
    echo "🚀 Running NEW evaluation:"
    echo "  Model: $model_path"
    echo "  Job suffix: '$job_suffix'"
    echo "  Checkpoint: $checkpoint_name"
    echo "  Output file: $output_filename"
    echo "  Full path: $output_file_path"
    echo "----------------------------------------"

    # Construct job name for display purposes
    local job_name="eval_${BASE_JOB_NAME}"
    if [ -n "$job_suffix" ]; then
        job_name="${job_name}_${job_suffix}"
    fi
    if [ -n "$checkpoint_name" ] && [ "$CHECKPOINTS" = true ]; then
        job_name="${job_name}_checkpoint_${checkpoint_name}"
    fi

    echo "Job name: $job_name"

    # Run the evaluation
    python cact_scripts/eval.py \
        --policy.path="$model_path" \
        --robot.type="$ROBOT_TYPE" \
        --policy.device="$DEVICE" \
        --job_name="$job_name" \
        --dataset.repo_id=$DATASET_LIST \
        --output_dir="$EVAL_BASE_DIR" \
        --output_filename="$output_filename"

    # Check if evaluation was successful
    if [ -f "$output_file_path" ]; then
        echo "✅ Evaluation completed successfully: $output_filename"
    else
        echo "❌ Evaluation failed: $output_filename was not created"
    fi
    echo ""
}

# Main evaluation loop
TOTAL_EVALUATIONS=0
SKIPPED_EVALUATIONS=0
COMPLETED_EVALUATIONS=0

for SELECTED_DIR in $TRAIN_DIRS; do
    echo "📁 Processing model: $(basename "$SELECTED_DIR")"

    # Extract model suffix for job naming
    MODEL_SUFFIX=$(get_model_suffix "$SELECTED_DIR")
    echo "Model suffix: '$MODEL_SUFFIX'"

    # Look for checkpoint directories
    CHECKPOINT_DIR="$SELECTED_DIR/checkpoints"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "⚠️  Warning: Checkpoints directory not found at $CHECKPOINT_DIR"
        echo "Skipping $(basename "$SELECTED_DIR")"
        continue
    fi

    # Find checkpoint directories (numbered directories)
    CHECKPOINTS_FOUND=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "[0-9]*" | sort -V)

    if [ -z "$CHECKPOINTS_FOUND" ]; then
        echo "⚠️  Warning: No checkpoint directories found in $CHECKPOINT_DIR"
        echo "Skipping $(basename "$SELECTED_DIR")"
        continue
    fi

    if [ "$CHECKPOINTS" = true ]; then
        # Evaluate all checkpoints
        echo "📊 Checkpoint mode: Evaluating all checkpoints"
        CHECKPOINT_COUNT=$(echo "$CHECKPOINTS_FOUND" | wc -l)
        echo "Found $CHECKPOINT_COUNT checkpoint(s) to evaluate:"
        echo "$CHECKPOINTS_FOUND" | sed 's/.*\///g' | sed 's/^/  - /'
        echo ""

        for CHECKPOINT_PATH in $CHECKPOINTS_FOUND; do
            CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")
            MODEL_PATH="$CHECKPOINT_PATH/pretrained_model"

            if [ ! -d "$MODEL_PATH" ]; then
                echo "⚠️  Warning: Pretrained model not found at $MODEL_PATH"
                echo "Skipping checkpoint $CHECKPOINT_NAME"
                continue
            fi

            TOTAL_EVALUATIONS=$((TOTAL_EVALUATIONS + 1))

            # Check if this evaluation would be skipped before calling the function
            output_filename="eval_result"
            if [ -n "$MODEL_SUFFIX" ]; then
                output_filename="${output_filename}_${MODEL_SUFFIX}"
            fi
            output_filename="${output_filename}_${CHECKPOINT_NAME}.json"

            if [ -f "$EVAL_BASE_DIR/$output_filename" ]; then
                SKIPPED_EVALUATIONS=$((SKIPPED_EVALUATIONS + 1))
            else
                COMPLETED_EVALUATIONS=$((COMPLETED_EVALUATIONS + 1))
            fi

            run_single_evaluation "$MODEL_PATH" "$MODEL_SUFFIX" "$CHECKPOINT_NAME"
        done

    else
        # Evaluate only the latest checkpoint (old behavior)
        echo "📊 Latest checkpoint mode: Evaluating most recent checkpoint only"
        LATEST_CHECKPOINT=$(echo "$CHECKPOINTS_FOUND" | tail -n 1)
        CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
        MODEL_PATH="$LATEST_CHECKPOINT/pretrained_model"

        echo "Latest checkpoint: $CHECKPOINT_NAME"

        if [ ! -d "$MODEL_PATH" ]; then
            echo "❌ Error: Pretrained model not found at $MODEL_PATH"
            echo "Skipping $(basename "$SELECTED_DIR")"
            continue
        fi

        TOTAL_EVALUATIONS=$((TOTAL_EVALUATIONS + 1))

        # Check if this evaluation would be skipped
        output_filename="eval_result"
        if [ -n "$MODEL_SUFFIX" ]; then
            output_filename="${output_filename}_${MODEL_SUFFIX}"
        fi
        output_filename="${output_filename}_${CHECKPOINT_NAME}.json"

        if [ -f "$EVAL_BASE_DIR/$output_filename" ]; then
            SKIPPED_EVALUATIONS=$((SKIPPED_EVALUATIONS + 1))
        else
            COMPLETED_EVALUATIONS=$((COMPLETED_EVALUATIONS + 1))
        fi

        run_single_evaluation "$MODEL_PATH" "$MODEL_SUFFIX" "$CHECKPOINT_NAME"
    fi

    echo "✅ Completed processing model: $(basename "$SELECTED_DIR")"
    echo ""
done

echo "🎉 All evaluations completed!"
echo ""
echo "📊 SUMMARY:"
echo "=================================="
echo "📁 Results directory: $EVAL_BASE_DIR"
echo "📈 Total potential evaluations: $TOTAL_EVALUATIONS"
echo "⏭️  Skipped (already exist): $SKIPPED_EVALUATIONS"
echo "🚀 Newly completed: $COMPLETED_EVALUATIONS"
echo "📄 Total files in directory: $(find "$EVAL_BASE_DIR" -name "*.json" | wc -l)"
echo "=================================="
echo ""
echo "📝 To extract results into a CSV:"
echo "   python extract_eval_results.py --base-job $BASE_JOB_NAME"
echo ""
echo "📊 To plot results:"
echo "   python plot_eval_results.py --input evaluation_results.csv"