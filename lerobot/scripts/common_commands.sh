python lerobot/scripts/configure_motor.py \
  --port /dev/ttyACM1 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 4


export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
HF_USER=$(huggingface-cli whoami | head -n 1)


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