#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

sudo chmod a+rw /dev/ttyACM1
sudo chmod a+rw /dev/ttyACM0

# Combined robot learning script for 3-object sorting training and testing
# Usage: ./robot_learning.sh [train|test]

mode="${1:-train}"  # Default to training if no mode specified

# Define training cases
# Format: "pos1_color_shape pos2_color_shape pos3_color_shape" (desired final ordering)
declare -a training_cases=(
"red_rectangle green_cube yellow_cube"
"red_rectangle red_cube green_cylinder"
"red_rectangle red_cube blue_cylinder"
"green_rectangle green_cube red_cube"
"red_rectangle green_rectangle green_cube"
"red_rectangle red_cube green_cube"
"green_rectangle red_rectangle blue_cylinder"
"red_rectangle blue_rectangle blue_cylinder"
"green_rectangle red_cube blue_cylinder"
"green_rectangle green_cube green_cylinder"
"green_rectangle blue_rectangle blue_cylinder"
"red_rectangle green_cube green_cylinder"
"green_rectangle green_cube blue_cylinder"
"green_rectangle red_rectangle red_cube"
"green_rectangle red_rectangle red_cylinder"
"green_rectangle red_rectangle yellow_cube"
"green_rectangle green_cube yellow_cube"
"green_rectangle red_cube red_cylinder"
"red_rectangle green_rectangle green_cylinder"
"red_rectangle green_rectangle yellow_cube"
)

# Define test cases
declare -a test_cases=(
"green_rectangle green_cube red_cylinder"
"red_rectangle green_cube blue_cylinder"
"red_rectangle red_cube yellow_cube"
"red_rectangle red_cube red_cylinder"
"green_rectangle red_cube yellow_cube"
"red_rectangle green_rectangle blue_cylinder"
)

# Function to check if a case is a test case
is_test_case() {
    local case_string="$1"
    for test_case in "${test_cases[@]}"; do
        if [ "$case_string" = "$test_case" ]; then
            return 0
        fi
    done
    return 1
}

# Parse case string into desired sorted order
parse_case() {
    local case_string="$1"
    echo $case_string  # Returns space-separated objects in desired order
}

# Create a description of the sorting task
create_sorting_description() {
    local objects=("$@")
    local description=""

    for i in "${!objects[@]}"; do
        local obj="${objects[$i]}"
        local color="${obj%_*}"
        local shape="${obj#*_}"
        local position=$((i + 1))

        if [ $i -eq 0 ]; then
            description="$color $shape in position $position"
        else
            description="$description, then $color $shape in position $position"
        fi
    done

    echo "$description"
}

# Process a single case based on mode
process_case() {
    local case_string="$1"
    local case_index="$2"
    local is_test=$(is_test_case "$case_string" && echo true || echo false)

    # Parse the case into desired sorted order
    local objects=($(parse_case "$case_string"))
    local obj1="${objects[0]}"  # Should be in position 1
    local obj2="${objects[1]}"  # Should be in position 2
    local obj3="${objects[2]}"  # Should be in position 3

    if [ "$mode" = "train" ] && [ "$is_test" = "false" ]; then
        # Training mode and not a test case
        echo "Training Case $case_index: Sort to [$obj1, $obj2, $obj3]"

        # Create description for speech
        local sorting_desc=$(create_sorting_description "${objects[@]}")

        python lerobot/scripts/say.py "Training case $case_index. Sort three objects: $sorting_desc."

        # Create repo ID with sorted order
        repo_id="sorting_case_${case_index}_${obj1}_${obj2}_${obj3}"

        python -m lerobot.record \
              --robot.type=so101_follower \
              --robot.port=/dev/ttyACM0 \
              --robot.id=so101_follower \
              --robot.cameras="{ hand: {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: -90}, scene: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
              --teleop.type=so101_leader \
              --teleop.port=/dev/ttyACM1 \
              --teleop.id=so101_leader \
              --display_data=False \
              --play_sounds=False \
              --dataset.fps=30 \
              --dataset.repo_id=${HF_USER}/so101_${repo_id} \
              --dataset.push_to_hub=False \
              --dataset.episode_time_s=60 \
              --dataset.reset_time_s=30 \
              --dataset.num_episodes=3 \
              --dataset.tags="['three_object_sorting', '$obj1', '$obj2', '$obj3', 'position_1', 'position_2', 'position_3']" \
              --dataset.single_task="Sort three randomly placed objects into correct order: $sorting_desc"

        python lerobot/scripts/say.py "Training case $case_index completed"
        return 0

    elif [ "$mode" = "test" ] && [ "$is_test" = "true" ]; then
        # Test mode and is a test case
        echo "Testing Case: Expected sorting [$obj1, $obj2, $obj3]"
        local sorting_desc=$(create_sorting_description "${objects[@]}")
        echo python eval_sorting.py "$obj1" "$obj2" "$obj3" "$sorting_desc"
        return 0
    fi

    return 1  # Don't count this case
}

# Initialize counters
total_count=0
case_index=1

# Print appropriate header
if [ "$mode" = "train" ]; then
    echo "Starting robot training for 3-object sorting..."
    cases_to_process=("${training_cases[@]}")
elif [ "$mode" = "test" ]; then
    echo "Starting evaluation with 3-object sorting test cases..."
    cases_to_process=("${test_cases[@]}")
else
    echo "Invalid mode. Use 'train' or 'test'"
    exit 1
fi

# Process all cases
for case_string in "${cases_to_process[@]}"; do
    if [ "$mode" = "train" ]; then
        # In training mode, only process training cases
        if process_case "$case_string" "$case_index"; then
            total_count=$((total_count + 1))
        fi
    else
        # In test mode, only process test cases
        if process_case "$case_string" "$case_index"; then
            total_count=$((total_count + 1))
        fi
    fi
    case_index=$((case_index + 1))
done

# Print appropriate summary
if [ "$mode" = "train" ]; then
    echo "Training completed with $total_count three-object sorting examples!"
else
    echo "Evaluation completed with $total_count three-object sorting test cases!"
fi