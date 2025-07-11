python lerobot/scripts/configure_motor.py \
  --port /dev/ttyACM1 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 4


export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
HF_USER=$(huggingface-cli whoami | head -n 1)


sudo chmod a+rw /dev/ttyACM0 2>/dev/null || true
sudo chmod a+rw /dev/ttyACM1 2>/dev/null || true

python lerobot/scripts/find_motors_bus_port.py

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'

  python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=teleoperate

  python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.single_task="Red block grasp test" \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/so100_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=10 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.resume=true


  python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so100_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so100_test \
  --job_name=act_so100_test \
  --policy.device=cuda \
  --wandb.enable=false


python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/so100_individual_cases_simple_with_concepts_blue-cylinder-2-B 



python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/so100_individual_cases_simple_blue-cylinder-2-B 

  python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/eval_so100_test \
  --control.single_task="Red block grasp eval" \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=false \
  --control.policy.path=outputs/train/act_so100_test/checkpoints/last/pretrained_model


python -m lerobot.calibrate --teleop.type=so100_leader --teleop.port=/dev/tty.ACM1  --teleop.id=so_100_leader 

python -m lerobot.teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so_100_follower \
    --robot.cameras="{ hand: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: 90}, scene: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so_100_leader \
    --display_data=true



python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so_100_follower \
    --robot.cameras="{ hand: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: 90}, scene: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so_100_leader \
    --display_data=False \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.push_to_hub=False \
    --dataset.num_episodes=2 \
    --dataset.single_task="Test"


python examples/backward_compatibility/replay.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so_100_follower \
    --dataset.repo_id=krallistic/so100_individual_cases_simple_with_concepts_yellow-cube-3-B \
    --dataset.episode=2