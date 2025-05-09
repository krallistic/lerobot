#!/bin/bash

HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


sudo chmod a+rw /dev/ttyACM1
sudo chmod a+rw /dev/ttyACM0
# Combined robot learning script for both training and testing
# Usage: ./robot_learning.sh [train|test]

mode="${1:-train}"  # Default to training if no mode specified

# Function to determine if a case is a test case
is_test_case() {
  color=$1
  shape=$2

  # List of test cases (20% of total combinations)
  if [ "$color" = "red" ] && [ "$shape" = "cube" ]; then return 0; fi
  if [ "$color" = "yellow" ] && [ "$shape" = "rectangle" ]; then return 0; fi

  return 1  # Not a test case
}

# Function to determine dropoff location based on new rule
# New rule: Object goes to Drop-off Location A if:
#  (The object is red) OR (The object is a rectangle AND NOT yellow)
determine_dropoff() {
  color=$1
  shape=$2
  location=$3  # Still included for parameter consistency but not used

  # Check if the object is red
  if [ "$color" = "red" ]; then
    echo "A"
    return
  fi

  # Check if the object is a rectangle AND not yellow
  if [ "$shape" = "rectangle" ] && [ "$color" != "yellow" ]; then
    echo "A"
    return
  fi

  # Otherwise, go to Location B
  echo "B"
}

# Process a single case based on mode
process_case() {
  color=$1
  shape=$2
  location=$3
  is_test=$(is_test_case "$color" "$shape" && echo true || echo false)

  if [ "$mode" = "train" ] && [ "$is_test" = "false" ]; then
    # Training mode and not a test case
    dropoff=$(determine_dropoff "$color" "$shape" "$location")
    echo "Training: $color $shape at location $location → $dropoff"
    python lerobot/scripts/say.py "Training: Color $color.  Shape $shape. at location $location. Dropoff in $dropoff. Dropoff in $dropoff."
    repo_id="individual_cases_simple_$color-$shape-$location-$dropoff"
    python lerobot/scripts/control_robot.py \
          --robot.type=so100 \
          --control.type=record \
          --control.single_task="Grasping Color: $color Shape: $shape Location: $location Dropoff: $dropoff" \
          --control.fps=30 \
          --control.repo_id=${HF_USER}/so100_${repo_id} \
          --control.tags='["$color", "$shape", "$location", "$dropoff"]' \
          --control.warmup_time_s=5 \
          --control.episode_time_s=30 \
          --control.reset_time_s=20 \
          --control.num_episodes=3 \
          --control.push_to_hub=false \
          --control.resume=true
    python lerobot/scripts/say.py "Done"
    return 0  # Count this case
  elif [ "$mode" = "test" ] && [ "$is_test" = "true" ]; then
    # Test mode and is a test case
    expected=$(determine_dropoff "$color" "$shape" "$location")
    echo "Testing: $color $shape at location $location → Expected: Location $expected"
    echo python eval.py "$color" "$shape" "$location" "$expected"
    return 0  # Count this case
  fi

  return 1  # Don't count this case
}

# Available inventory
cube_colors=("red" "green" "yellow")
rectangle_colors=("red" "blue" "green" "yellow")
cylinder_colors=("red" "blue" "green")
locations=(1 2 3 4 5)

# Initialize counters
total_count=0

# Print appropriate header
if [ "$mode" = "train" ]; then
  echo "Starting robot training..."
elif [ "$mode" = "test" ]; then
  echo "Starting evaluation with test cases..."
else
  echo "Invalid mode. Use 'train' or 'test'"
  exit 1
fi

# Process all valid combinations in a single loop
for shape in "cube" "rectangle" "cylinder"; do
  # Select appropriate colors based on shape
  if [ "$shape" = "cube" ]; then
    colors=("${cube_colors[@]}")
  elif [ "$shape" = "rectangle" ]; then
    colors=("${rectangle_colors[@]}")
  else  # cylinder
    colors=("${cylinder_colors[@]}")
  fi

  for color in "${colors[@]}"; do
    for location in "${locations[@]}"; do
      # Process the case and increment counter if processed
      if process_case "$color" "$shape" "$location"; then
        total_count=$((total_count + 1))
      fi
    done
  done
done

# Print appropriate summary
if [ "$mode" = "train" ]; then
  echo "Training completed with $total_count examples!"
else
  echo "Evaluation completed with $total_count test cases!"
fi